from __future__ import annotations

import argparse

import numpy as np

from nsga2_solver import run_surrogate_nsga2
from problem.problem import SUPPORTED_PROBLEMS, make_problem
from ref_points_hv import get_reference_point
from reward import hypervolume, pareto_front
from surrogate.surrogate_model import fit_tabpfn_surrogate
from tester import latin_hypercube_sample, make_nsga2_problem_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 80 initial samples, evolve with TabPFN surrogate, then pick the true-evaluated offspring that maximizes HV improvement."
    )
    parser.add_argument("--problem", type=str, default="ZDT1", choices=SUPPORTED_PROBLEMS)
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--init_fe", type=int, default=80)
    parser.add_argument("--offspring_size", type=int, default=80)
    parser.add_argument("--surrogate_nsga_steps", type=int, default=30)
    parser.add_argument("--ensemble_model", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))

    problem = make_problem(args.problem, dim=int(args.dim))
    archive_x = latin_hypercube_sample(
        n_samples=int(args.init_fe),
        dim=int(args.dim),
        lower=problem.lower,
        upper=problem.upper,
        seed=int(args.seed),
    )
    archive_y = np.asarray(problem.evaluate(archive_x), dtype=np.float32)
    n_obj = int(archive_y.shape[1])
    ref_point = np.asarray(get_reference_point(args.problem, n_obj=n_obj), dtype=np.float32)
    nsga2_problem = make_nsga2_problem_adapter(problem, n_obj)
    archive_hv = float(hypervolume(archive_y, ref_point))

    print(f"reference_point = {ref_point.tolist()}")
    print(f"init archive size = {int(archive_x.shape[0])} | front = {int(pareto_front(archive_y).shape[0])} | HV = {archive_hv:.6f}")

    surrogate = fit_tabpfn_surrogate(
        archive_x=archive_x,
        archive_y=archive_y,
        device=str(args.device),
        n_estimators=int(args.ensemble_model),
    )

    offspring_x, offspring_pred = run_surrogate_nsga2(
        surrogate=surrogate,
        gps=None,
        problem=nsga2_problem,
        archive_x=archive_x,
        pop_size=int(args.offspring_size),
        surrogate_nsga_steps=int(args.surrogate_nsga_steps),
        seed=int(args.seed),
    )
    offspring_x = np.asarray(offspring_x, dtype=np.float32)
    offspring_pred = np.asarray(offspring_pred, dtype=np.float32)
    true_offspring_y = np.asarray(problem.evaluate(offspring_x), dtype=np.float32)

    best_idx = -1
    best_hv = archive_hv
    best_delta = -float("inf")
    best_x = None
    best_y = None

    for idx in range(int(offspring_x.shape[0])):
        candidate_x = offspring_x[idx : idx + 1]
        candidate_y = true_offspring_y[idx : idx + 1]
        next_hv = float(hypervolume(np.vstack([archive_y, candidate_y]), ref_point))
        delta_hv = float(next_hv - archive_hv)
        if delta_hv > best_delta:
            best_idx = int(idx)
            best_hv = float(next_hv)
            best_delta = float(delta_hv)
            best_x = np.asarray(candidate_x, dtype=np.float32).reshape(-1)
            best_y = np.asarray(candidate_y, dtype=np.float32).reshape(-1)

    if best_idx < 0 or best_x is None or best_y is None:
        raise RuntimeError("Failed to select a best offspring from the evolved candidate set.")

    print(
        f"best_offspring_idx = {best_idx} | "
        f"archive_hv_after_add = {best_hv:.6f} | "
        f"hv_improvement = {best_delta:.6f}"
    )
    print(f"best_offspring_x = {best_x.astype(float).tolist()}")
    print(f"best_offspring_true_y = {best_y.astype(float).tolist()}")
    print(f"best_offspring_surrogate_y = {np.asarray(offspring_pred[best_idx], dtype=np.float32).astype(float).tolist()}")


if __name__ == "__main__":
    main()
