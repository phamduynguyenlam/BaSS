from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


@dataclass(frozen=True)
class USEMOResult:
    x: np.ndarray
    y: np.ndarray
    sigma: np.ndarray


class _SingleObjectiveGaussianProcess:
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.x_values: list[np.ndarray] = []
        self.y_values: list[float] = []
        sqrd_exp = RBF(length_scale=1.0)
        self.model = GaussianProcessRegressor(kernel=sqrd_exp, n_restarts_optimizer=10)

    def add_sample(self, x: np.ndarray, y: float) -> None:
        self.x_values.append(np.asarray(x, dtype=np.float64).reshape(-1))
        self.y_values.append(float(y))

    def fit(self) -> None:
        x_arr = np.asarray(self.x_values, dtype=np.float64)
        y_arr = np.asarray(self.y_values, dtype=np.float64)
        self.model.fit(x_arr, y_arr)

    def predict_mean_std(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float64)
        mean, std = self.model.predict(x_arr, return_std=True)
        return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


class _USEMOGaussianProcessSuite:
    def __init__(self, n_var: int, n_obj: int):
        self.n_var = int(n_var)
        self.n_obj = int(n_obj)
        self.models = [_SingleObjectiveGaussianProcess(n_var) for _ in range(n_obj)]

    def fit(self, archive_x: np.ndarray, archive_y: np.ndarray) -> "_USEMOGaussianProcessSuite":
        x_arr = np.asarray(archive_x, dtype=np.float64)
        y_arr = np.asarray(archive_y, dtype=np.float64)
        if x_arr.ndim != 2:
            raise ValueError(f"archive_x must be 2D, got shape={x_arr.shape}.")
        if y_arr.ndim != 2:
            raise ValueError(f"archive_y must be 2D, got shape={y_arr.shape}.")
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"archive_x/archive_y row mismatch: {x_arr.shape} vs {y_arr.shape}.")

        for obj_id, model in enumerate(self.models):
            for x_row, y_val in zip(x_arr, y_arr[:, obj_id], strict=False):
                model.add_sample(x_row, float(y_val))
            model.fit()
        return self

    def predict_mean_std(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float64)
        means: list[np.ndarray] = []
        stds: list[np.ndarray] = []
        for model in self.models:
            mean, std = model.predict_mean_std(x_arr)
            means.append(mean.reshape(-1))
            stds.append(std.reshape(-1))
        mean_arr = np.stack(means, axis=1).astype(np.float32)
        std_arr = np.stack(stds, axis=1).astype(np.float32)
        return mean_arr, std_arr


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
        gp_suite: _USEMOGaussianProcessSuite,
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
        mean, std = self.gp_suite.predict_mean_std(x_arr)
        if self.acquisition == "lcb":
            out["F"] = (mean - self.beta * std).astype(np.float32)
            return
        if self.acquisition == "ei":
            ei = _expected_improvement(mean, std, self.incumbent)
            out["F"] = (-ei).astype(np.float32)
            return
        raise ValueError(f"Unsupported USEMO acquisition: {self.acquisition}")


def run_surrogate_usemo(
    problem,
    archive_x,
    archive_y,
    pop_size,
    surrogate_nsga_steps=100,
    seed=0,
    n_gen=None,
    acquisition="lcb",
    beta=1.0,
):
    acquisition_name = str(acquisition).lower()
    if acquisition_name not in {"lcb", "ei"}:
        raise ValueError(f"USEMO solver supports only acquisition in {{'lcb', 'ei'}}, got {acquisition}.")

    if n_gen is not None:
        surrogate_nsga_steps = n_gen

    archive_x_arr = np.asarray(archive_x, dtype=np.float64)
    archive_y_arr = np.asarray(archive_y, dtype=np.float64)
    gp_suite = _USEMOGaussianProcessSuite(
        n_var=int(problem.n_var),
        n_obj=int(problem.n_obj),
    ).fit(archive_x_arr, archive_y_arr)

    init_x = np.asarray(archive_x_arr, dtype=np.float64)
    if init_x.shape[0] >= int(pop_size):
        init_x = init_x[: int(pop_size)].copy()
    else:
        rng = np.random.default_rng(int(seed))
        idx = rng.integers(0, init_x.shape[0], size=int(pop_size))
        init_x = init_x[idx].copy()

    surrogate_problem = _USEMOProblem(
        gp_suite=gp_suite,
        incumbent=np.min(archive_y_arr, axis=0),
        n_var=int(problem.n_var),
        n_obj=int(problem.n_obj),
        xl=np.asarray(problem.xl, dtype=np.float32),
        xu=np.asarray(problem.xu, dtype=np.float32),
        acquisition=acquisition_name,
        beta=float(beta),
    )

    algorithm = NSGA2(
        pop_size=int(pop_size),
        sampling=init_x,
        eliminate_duplicates=True,
    )

    res = minimize(
        surrogate_problem,
        algorithm,
        termination=get_termination("n_gen", int(surrogate_nsga_steps)),
        seed=int(seed),
        verbose=False,
        save_history=False,
    )

    x_res = np.asarray(res.X, dtype=np.float32)
    mean_res, std_res = gp_suite.predict_mean_std(x_res)
    return x_res, np.asarray(mean_res, dtype=np.float32), np.asarray(std_res, dtype=np.float32)
