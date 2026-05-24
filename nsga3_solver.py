from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from nsga2_solver import GPSurrogateProblem, _ModelListSurrogate


@dataclass(frozen=True)
class NSGA3Result:
    x: np.ndarray
    y: np.ndarray


def _build_reference_directions(n_obj: int, pop_size: int, seed: int) -> np.ndarray:
    try:
        ref_dirs = get_reference_directions("energy", int(n_obj), int(pop_size), seed=int(seed))
        return np.asarray(ref_dirs, dtype=np.float64)
    except TypeError:
        ref_dirs = get_reference_directions("energy", int(n_obj), int(pop_size))
        return np.asarray(ref_dirs, dtype=np.float64)
    except Exception:
        pass

    n_partitions = 1
    while comb(int(n_partitions) + int(n_obj) - 1, int(n_obj) - 1) < int(pop_size):
        n_partitions += 1
    ref_dirs = get_reference_directions("das-dennis", int(n_obj), n_partitions=int(n_partitions))
    ref_dirs = np.asarray(ref_dirs, dtype=np.float64)
    if ref_dirs.shape[0] > int(pop_size):
        ref_dirs = ref_dirs[: int(pop_size)].copy()
    return ref_dirs


def run_surrogate_nsga3(
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
            raise ValueError("run_surrogate_nsga3 requires either `surrogate` or `gps`.")
        surrogate = _ModelListSurrogate(gps)

    surrogate_problem = GPSurrogateProblem(
        surrogate=surrogate,
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        xl=problem.xl,
        xu=problem.xu,
    )

    init_x = np.asarray(archive_x, dtype=np.float64)
    if init_x.shape[0] >= pop_size:
        init_x = init_x[:pop_size].copy()
    else:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, init_x.shape[0], size=pop_size)
        init_x = init_x[idx].copy()

    ref_dirs = _build_reference_directions(int(problem.n_obj), int(pop_size), int(seed))
    algorithm = NSGA3(
        pop_size=int(pop_size),
        ref_dirs=ref_dirs,
        sampling=init_x,
        eliminate_duplicates=True,
    )

    res = minimize(
        surrogate_problem,
        algorithm,
        termination=get_termination("n_gen", int(surrogate_nsga_steps)),
        seed=seed,
        verbose=False,
        save_history=False,
    )

    return np.asarray(res.X, dtype=np.float64), np.asarray(res.F, dtype=np.float64)
