from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

USEMO_GP_CONFIG: dict[str, object] = {
    "normalize_x": False,
    "standardize_y": False,
    "winsorize_y_quantile": None,
    "kernel": "rbf",
    "nu": 0.0,
    "ard": False,
    "output_model": "independent_gp_per_objective",
    "likelihood": "gaussian",
    "noise_constraint": [1e-5, 5e-2],
    "initial_noise": 1e-4,
    "lengthscale_constraint": [0.05, 2.0],
    "initial_lengthscale": 0.30,
    "outputscale_constraint": [0.05, 20.0],
    "max_fit_iter": 80,
    "num_restarts": 4,
    "jitter": 1e-6,
    "surrogate_objective": "mean",
    "lcb_beta": 0.0,
    "cheap_solver": "NSGA-II",
    "surrogate_nsga_steps": 50,
    "pop_size": 80,
    "n_restarts": 1,
    "candidate_pool_size": 80,
    "keep_top_k": 80,
    "filter_invalid": False,
}


@dataclass(frozen=True)
class USEMOResult:
    x: np.ndarray
    y: np.ndarray
    sigma: np.ndarray


def _suppress_gp_warnings() -> None:
    def warn(*args, **kwargs):
        return None

    import warnings

    warnings.warn = warn


@dataclass
class GPFixedInternalSurrogateModel:
    n_var: int
    n_obj: int
    seed: int = 0
    gps: list[GaussianProcessRegressor] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _suppress_gp_warnings()
        self.n_var = int(self.n_var)
        self.n_obj = int(self.n_obj)
        self.seed = int(self.seed)
        self.gps = []

        for obj_id in range(self.n_obj):
            kernel = (
                ConstantKernel(
                    constant_value=1.0,
                    constant_value_bounds="fixed",
                )
                * RBF(
                    length_scale=0.5,
                    length_scale_bounds="fixed",
                )
                + WhiteKernel(
                    noise_level=1e-5,
                    noise_level_bounds="fixed",
                )
            )

            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                optimizer=None,
                random_state=self.seed + obj_id,
            )
            self.gps.append(model)

    @property
    def models(self) -> list[GaussianProcessRegressor]:
        return self.gps

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GPFixedInternalSurrogateModel":
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        for obj_id, gp in enumerate(self.gps):
            gp.fit(x_arr, y_arr[:, obj_id])
        return self

    def predict_mean(self, x: np.ndarray, device: str | None = None) -> np.ndarray:
        del device
        x_arr = np.asarray(x, dtype=np.float64)
        preds = []
        for gp in self.gps:
            mean = gp.predict(x_arr, return_std=False)
            preds.append(np.asarray(mean, dtype=np.float32).reshape(-1))
        out = np.stack(preds, axis=1).astype(np.float32)
        return np.maximum(out, 0.0).astype(np.float32)

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        preds = []
        for gp in self.gps:
            _, std = gp.predict(x_arr, return_std=True)
            preds.append(np.asarray(std, dtype=np.float32).reshape(-1))
        return np.stack(preds, axis=1).astype(np.float32) + 1e-6


class _ConfiguredUSEMOGP:
    def __init__(self, *, xl: np.ndarray, xu: np.ndarray):
        self.xl = np.asarray(xl, dtype=np.float64).reshape(1, -1)
        self.xu = np.asarray(xu, dtype=np.float64).reshape(1, -1)
        self.x_span = np.clip(self.xu - self.xl, 1e-12, None)
        self.y_low: np.ndarray | None = None
        self.y_high: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_scale: np.ndarray | None = None
        self.model = None

    def _normalize_x(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        if bool(USEMO_GP_CONFIG["normalize_x"]):
            return np.clip((x_arr - self.xl) / self.x_span, 0.0, 1.0)
        return x_arr

    def _winsorize_y(self, y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.float64)
        q_value = USEMO_GP_CONFIG["winsorize_y_quantile"]
        if q_value is None:
            self.y_low = np.min(y_arr, axis=0)
            self.y_high = np.max(y_arr, axis=0)
            return y_arr
        q = float(q_value)
        if q <= 0.0:
            self.y_low = np.min(y_arr, axis=0)
            self.y_high = np.max(y_arr, axis=0)
            return y_arr
        self.y_low = np.quantile(y_arr, q, axis=0)
        self.y_high = np.quantile(y_arr, 1.0 - q, axis=0)
        return np.clip(y_arr, self.y_low, self.y_high)

    def _standardize_y(self, y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.float64)
        if bool(USEMO_GP_CONFIG["standardize_y"]):
            self.y_mean = np.mean(y_arr, axis=0)
            self.y_scale = np.std(y_arr, axis=0)
            self.y_scale = np.where(self.y_scale < 1e-8, 1.0, self.y_scale)
            return (y_arr - self.y_mean) / self.y_scale
        self.y_mean = np.zeros(y_arr.shape[1], dtype=np.float64)
        self.y_scale = np.ones(y_arr.shape[1], dtype=np.float64)
        return y_arr

    def fit(self, archive_x: np.ndarray, archive_y: np.ndarray, *, seed: int) -> "_ConfiguredUSEMOGP":
        x_arr = np.asarray(archive_x, dtype=np.float64)
        y_arr = np.asarray(archive_y, dtype=np.float64)
        self.y_low = np.min(y_arr, axis=0)
        self.y_high = np.max(y_arr, axis=0)
        self.y_mean = np.zeros(y_arr.shape[1], dtype=np.float64)
        self.y_scale = np.ones(y_arr.shape[1], dtype=np.float64)
        self.model = GPFixedInternalSurrogateModel(
            n_var=int(x_arr.shape[1]),
            n_obj=int(y_arr.shape[1]),
            seed=int(seed),
        ).fit(x_arr, y_arr)
        return self

    def _restore_mean(self, mean: np.ndarray) -> np.ndarray:
        mean_arr = np.asarray(mean, dtype=np.float64)
        assert self.y_mean is not None
        assert self.y_scale is not None
        restored = mean_arr * self.y_scale.reshape(1, -1) + self.y_mean.reshape(1, -1)
        return restored.astype(np.float32)

    def _restore_std(self, std: np.ndarray) -> np.ndarray:
        std_arr = np.asarray(std, dtype=np.float64)
        assert self.y_scale is not None
        restored = std_arr * self.y_scale.reshape(1, -1)
        return np.maximum(restored, 1e-6).astype(np.float32)

    def predict_mean(self, x: np.ndarray) -> np.ndarray:
        assert self.model is not None
        x_arr = self._normalize_x(np.asarray(x, dtype=np.float64))
        mean = np.asarray(self.model.predict_mean(x_arr), dtype=np.float64)
        return self._restore_mean(mean)

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        assert self.model is not None
        x_arr = self._normalize_x(np.asarray(x, dtype=np.float64))
        std = np.asarray(self.model.predict_std(x_arr), dtype=np.float64)
        return self._restore_std(std)

    def incumbent(self) -> np.ndarray:
        assert self.y_low is not None
        assert self.y_high is not None
        assert self.y_mean is not None
        assert self.y_scale is not None
        y_ref = np.clip(self.y_mean.reshape(1, -1), self.y_low.reshape(1, -1), self.y_high.reshape(1, -1))
        return y_ref.reshape(-1).astype(np.float32)


def _dominates(lhs: np.ndarray, rhs: np.ndarray) -> bool:
    lhs_arr = np.asarray(lhs, dtype=np.float32).reshape(-1)
    rhs_arr = np.asarray(rhs, dtype=np.float32).reshape(-1)
    return bool(np.all(lhs_arr <= rhs_arr) and np.any(lhs_arr < rhs_arr))


def _pareto_mask(values: np.ndarray) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float32)
    if values_arr.ndim != 2:
        raise ValueError(f"values must be 2D, got shape={values_arr.shape}.")
    keep = np.ones(int(values_arr.shape[0]), dtype=bool)
    for i in range(int(values_arr.shape[0])):
        if not keep[i]:
            continue
        for j in range(int(values_arr.shape[0])):
            if i == j:
                continue
            if _dominates(values_arr[j], values_arr[i]):
                keep[i] = False
                break
    return keep


def _expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    incumbent: np.ndarray,
) -> np.ndarray:
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32).clip(min=1e-12)
    incumbent_arr = np.asarray(incumbent, dtype=np.float32).reshape(1, -1)
    improvement = incumbent_arr - mean_arr
    z = improvement / std_arr
    ei = improvement * norm.cdf(z) + std_arr * norm.pdf(z)
    return np.asarray(ei, dtype=np.float32)


class _USEMOProblem(Problem):
    def __init__(
        self,
        gp_suite,
        incumbent: np.ndarray,
        n_var: int,
        n_obj: int,
        xl: np.ndarray,
        xu: np.ndarray,
        acquisition: str,
        beta: float,
    ):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
        self.gp_suite = gp_suite
        self.incumbent = np.asarray(incumbent, dtype=np.float32)
        self.acquisition = str(acquisition).lower()
        self.beta = float(beta)

    def _evaluate(self, X, out, *args, **kwargs):
        x_arr = np.asarray(X, dtype=np.float32)
        mean = np.asarray(self.gp_suite.predict_mean(x_arr), dtype=np.float32)
        std = np.asarray(self.gp_suite.predict_std(x_arr), dtype=np.float32)
        if self.acquisition == "lcb":
            out["F"] = (mean - self.beta * std).astype(np.float32)
            return
        if self.acquisition == "ei":
            ei = _expected_improvement(mean, std, self.incumbent)
            out["F"] = (-ei).astype(np.float32)
            return
        raise ValueError(f"Unsupported USEMO acquisition: {self.acquisition}")


def _build_initial_population(archive_x: np.ndarray, pop_size: int, seed: int) -> np.ndarray:
    init_x = np.asarray(archive_x, dtype=np.float64)
    if init_x.shape[0] >= int(pop_size):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(init_x.shape[0], size=int(pop_size), replace=False)
        return init_x[idx].copy()
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, init_x.shape[0], size=int(pop_size))
    return init_x[idx].copy()


def _filter_valid_candidates(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    valid = np.all(np.isfinite(x), axis=1)
    valid &= np.all(np.isfinite(mean), axis=1)
    valid &= np.all(np.isfinite(std), axis=1)
    valid &= np.all(std > 0.0, axis=1)
    return valid


def run_surrogate_usemo(
    problem,
    archive_x,
    archive_y,
    pop_size,
    surrogate_nsga_steps=32,
    seed=0,
    n_gen=None,
    acquisition="ei",
    beta=None,
):
    acquisition_name = str(acquisition).lower()
    if acquisition_name not in {"lcb", "ei"}:
        raise ValueError(f"USEMO solver supports only acquisition in {{'lcb', 'ei'}}, got {acquisition}.")

    if n_gen is not None:
        surrogate_nsga_steps = n_gen
    if surrogate_nsga_steps is None:
        surrogate_nsga_steps = int(USEMO_GP_CONFIG["surrogate_nsga_steps"])
    effective_beta = float(USEMO_GP_CONFIG["lcb_beta"] if beta is None else beta)

    archive_x_arr = np.asarray(archive_x, dtype=np.float64)
    archive_y_arr = np.asarray(archive_y, dtype=np.float64)
    xl = np.asarray(problem.xl, dtype=np.float32)
    xu = np.asarray(problem.xu, dtype=np.float32)
    gp_suite = _ConfiguredUSEMOGP(xl=xl, xu=xu).fit(
        archive_x=archive_x_arr,
        archive_y=archive_y_arr,
        seed=int(seed),
    )

    surrogate_problem = _USEMOProblem(
        gp_suite=gp_suite,
        incumbent=np.min(np.asarray(archive_y_arr, dtype=np.float32), axis=0),
        n_var=int(problem.n_var),
        n_obj=int(problem.n_obj),
        xl=xl,
        xu=xu,
        acquisition=acquisition_name,
        beta=effective_beta,
    )

    all_x: list[np.ndarray] = []
    all_acq: list[np.ndarray] = []
    n_restarts = int(USEMO_GP_CONFIG["n_restarts"])
    for restart_id in range(n_restarts):
        init_x = _build_initial_population(archive_x_arr, int(pop_size), int(seed) + int(restart_id))
        algorithm = NSGA2(
            pop_size=int(pop_size),
            sampling=init_x,
            eliminate_duplicates=True,
        )
        res = pymoo_minimize(
            surrogate_problem,
            algorithm,
            termination=get_termination("n_gen", int(surrogate_nsga_steps)),
            seed=int(seed) + int(restart_id),
            verbose=False,
            save_history=False,
        )
        all_x.append(np.asarray(res.X, dtype=np.float32))
        all_acq.append(np.asarray(res.F, dtype=np.float32))

    x_res = np.vstack(all_x).astype(np.float32)
    acq_res = np.vstack(all_acq).astype(np.float32)
    if int(USEMO_GP_CONFIG["candidate_pool_size"]) > 0 and x_res.shape[0] > int(USEMO_GP_CONFIG["candidate_pool_size"]):
        x_res = x_res[: int(USEMO_GP_CONFIG["candidate_pool_size"])].copy()
        acq_res = acq_res[: int(USEMO_GP_CONFIG["candidate_pool_size"])].copy()

    mean_res = np.asarray(gp_suite.predict_mean(x_res), dtype=np.float32)
    std_res = np.asarray(gp_suite.predict_std(x_res), dtype=np.float32)
    if bool(USEMO_GP_CONFIG["filter_invalid"]):
        valid_mask = _filter_valid_candidates(x_res, mean_res, std_res)
        x_res = x_res[valid_mask]
        acq_res = acq_res[valid_mask]
        mean_res = mean_res[valid_mask]
        std_res = std_res[valid_mask]

    mask = _pareto_mask(acq_res)
    x_res = np.asarray(x_res[mask], dtype=np.float32)
    mean_res = np.asarray(mean_res[mask], dtype=np.float32)
    std_res = np.asarray(std_res[mask], dtype=np.float32)

    keep_top_k = int(USEMO_GP_CONFIG["keep_top_k"])
    if keep_top_k > 0 and x_res.shape[0] > keep_top_k:
        uncertainty = np.linalg.norm(std_res, axis=1)
        top_idx = np.argsort(-uncertainty)[:keep_top_k]
        x_res = x_res[top_idx]
        mean_res = mean_res[top_idx]
        std_res = std_res[top_idx]

    return (
        x_res,
        mean_res,
        std_res,
    )
