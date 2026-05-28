from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from solver.nsga2_solver import GPSurrogateProblem, _ModelListSurrogate


@dataclass(frozen=True)
class MOEADResult:
    x: np.ndarray
    y: np.ndarray


def _build_reference_directions(pop_size: int, n_obj: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    ref_dirs = rng.random((int(pop_size), int(n_obj)), dtype=np.float64)
    ref_sums = np.sum(ref_dirs, axis=1, keepdims=True)
    ref_sums = np.where(ref_sums <= 0.0, 1.0, ref_sums)
    return ref_dirs / ref_sums


def run_surrogate_moead(
    problem,
    archive_x,
    pop_size,
    gps=None,
    surrogate=None,
    surrogate_nsga_steps=100,
    seed=0,
    n_gen=None,
):
    if n_gen is not None:
        surrogate_nsga_steps = n_gen
    if surrogate is None:
        if gps is None:
            raise ValueError("run_surrogate_moead requires either `surrogate` or `gps`.")
        surrogate = _ModelListSurrogate(gps)

    surrogate_problem = GPSurrogateProblem(
        surrogate=surrogate,
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        xl=problem.xl,
        xu=problem.xu,
    )

    init_x = np.asarray(archive_x, dtype=np.float64)
    if init_x.shape[0] >= int(pop_size):
        init_x = init_x[: int(pop_size)].copy()
    else:
        rng = np.random.default_rng(int(seed))
        idx = rng.integers(0, init_x.shape[0], size=int(pop_size))
        init_x = init_x[idx].copy()

    ref_dirs = _build_reference_directions(int(pop_size), int(problem.n_obj), int(seed))
    algorithm = MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=min(max(2, int(pop_size) // 10), int(pop_size)),
        prob_neighbor_mating=0.9,
        sampling=init_x,
    )

    res = minimize(
        surrogate_problem,
        algorithm,
        termination=get_termination("n_gen", int(surrogate_nsga_steps)),
        seed=int(seed),
        verbose=False,
        save_history=False,
    )

    return np.asarray(res.X, dtype=np.float64), np.asarray(res.F, dtype=np.float64)
