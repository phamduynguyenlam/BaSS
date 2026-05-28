from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination
from scipy.stats import norm
from surrogate.gp import USEMO_GP_CONFIG, fit_gp_surrogates
from surrogate.surrogate_model import fit_tabpfn_surrogate


@dataclass(frozen=True)
class USEMOResult:
    x: np.ndarray
    y: np.ndarray
    sigma: np.ndarray


def _resolve_surrogate_nsga_steps(
    surrogate_nsga_steps: Any,
    *,
    n_gen: int | None = None,
) -> int:
    if n_gen is not None:
        return int(n_gen)

    if surrogate_nsga_steps is None:
        return int(USEMO_GP_CONFIG["surrogate_nsga_steps"])

    if isinstance(surrogate_nsga_steps, (int, np.integer)):
        return int(surrogate_nsga_steps)

    if isinstance(surrogate_nsga_steps, dict):
        if surrogate_nsga_steps.get("gp_nsga_steps", None) is not None:
            return int(surrogate_nsga_steps["gp_nsga_steps"])
        if surrogate_nsga_steps.get("surrogate_nsga_steps", None) is not None:
            return int(surrogate_nsga_steps["surrogate_nsga_steps"])

    if isinstance(surrogate_nsga_steps, argparse.Namespace) or hasattr(surrogate_nsga_steps, "__dict__"):
        gp_steps = getattr(surrogate_nsga_steps, "gp_nsga_steps", None)
        if gp_steps is not None:
            return int(gp_steps)
        base_steps = getattr(surrogate_nsga_steps, "surrogate_nsga_steps", None)
        if base_steps is not None:
            return int(base_steps)

    return int(surrogate_nsga_steps)


def _resolve_usemo_option(
    explicit_value: Any,
    fallback_source: Any,
    *,
    key: str,
    default: Any,
) -> Any:
    if explicit_value is not None and not isinstance(explicit_value, (argparse.Namespace, dict)):
        return explicit_value

    for source in (explicit_value, fallback_source):
        if source is None:
            continue
        if isinstance(source, dict) and key in source and source[key] is not None:
            return source[key]
        if isinstance(source, argparse.Namespace) or hasattr(source, "__dict__"):
            value = getattr(source, key, None)
            if value is not None:
                return value
    return default


def _build_usemo_surrogate(
    *,
    problem: Any,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    seed: int,
    config_source: Any,
    surrogate_model: Any | None = None,
    device: Any | None = None,
):
    surrogate_name = str(
        _resolve_usemo_option(
            surrogate_model,
            config_source,
            key="surrogate_model",
            default="gp",
        )
    ).lower()
    resolved_device = str(
        _resolve_usemo_option(
            device,
            config_source,
            key="device",
            default="cpu",
        )
    )

    if surrogate_name == "gp":
        gp_nu = float(_resolve_usemo_option(None, config_source, key="gp_nu", default=5.0))
        return fit_gp_surrogates(
            archive_x=archive_x,
            archive_y=archive_y,
            seed=int(seed),
            nu=gp_nu,
            variant="gp",
        )

    if surrogate_name == "gp2":
        return fit_gp_surrogates(
            archive_x=archive_x,
            archive_y=archive_y,
            seed=int(seed),
            variant="gp2",
        )

    if surrogate_name == "gp3":
        gp3_nu = float(_resolve_usemo_option(None, config_source, key="gp3_nu", default=USEMO_GP_CONFIG["nu"]))
        return fit_gp_surrogates(
            archive_x=archive_x,
            archive_y=archive_y,
            variant="gp3",
            xl=np.asarray(problem.xl, dtype=np.float32),
            xu=np.asarray(problem.xu, dtype=np.float32),
            seed=int(seed),
            nu=gp3_nu,
        )

    if surrogate_name == "tabpfn":
        n_estimators = int(_resolve_usemo_option(None, config_source, key="ensemble_model", default=8))
        return fit_tabpfn_surrogate(
            archive_x=archive_x,
            archive_y=archive_y,
            device=resolved_device,
            n_estimators=n_estimators,
        )

    if surrogate_name == "kan":
        raise ValueError("USEMO does not support surrogate_model='kan' because acquisition requires predict_std().")

    raise ValueError(f"Unsupported USEMO surrogate_model: {surrogate_name}")


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
    surrogate_model=None,
    device=None,
):
    acquisition_name = str(acquisition).lower()
    if acquisition_name not in {"lcb", "ei"}:
        raise ValueError(f"USEMO solver supports only acquisition in {{'lcb', 'ei'}}, got {acquisition}.")

    resolved_nsga_steps = _resolve_surrogate_nsga_steps(
        surrogate_nsga_steps,
        n_gen=n_gen,
    )
    effective_beta = float(USEMO_GP_CONFIG["lcb_beta"] if beta is None else beta)

    archive_x_arr = np.asarray(archive_x, dtype=np.float64)
    archive_y_arr = np.asarray(archive_y, dtype=np.float64)
    xl = np.asarray(problem.xl, dtype=np.float32)
    xu = np.asarray(problem.xu, dtype=np.float32)
    gp_suite = _build_usemo_surrogate(
        problem=problem,
        archive_x=archive_x_arr,
        archive_y=archive_y_arr,
        seed=int(seed),
        config_source=surrogate_nsga_steps,
        surrogate_model=surrogate_model,
        device=device,
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
            termination=get_termination("n_gen", int(resolved_nsga_steps)),
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
