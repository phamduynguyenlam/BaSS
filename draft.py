from __future__ import annotations

import argparse

import numpy as np

from nsga2_solver import run_surrogate_nsga2
from problem.problem import make_problem
from ref_points_hv import get_reference_point
from reward import hypervolume
from surrogate.surrogate_model import fit_tabpfn_surrogate
from tester import latin_hypercube_sample, make_nsga2_problem_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample 80 LHS points on DTLZ6, fit a TabPFN surrogate, run NSGA-II on the surrogate, and print each offspring's true and predicted objective values."
    )
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

    problem_name = "DTLZ6"
    problem = make_problem(problem_name, dim=int(args.dim))
    archive_x = latin_hypercube_sample(
        n_samples=int(args.init_fe),
        dim=int(args.dim),
        lower=problem.lower,
        upper=problem.upper,
        seed=int(args.seed),
    )
    archive_y = np.asarray(problem.evaluate(archive_x), dtype=np.float32)
    n_obj = int(archive_y.shape[1])
    nsga2_problem = make_nsga2_problem_adapter(problem, n_obj)
    ref_point = np.asarray(get_reference_point(problem_name, n_obj=n_obj), dtype=np.float32)
    archive_hv = float(hypervolume(archive_y, ref_point))

    print(f"problem = {problem_name} | dim = {int(args.dim)} | init_fe = {int(args.init_fe)}")
    print(f"archive_x shape = {tuple(archive_x.shape)} | archive_y shape = {tuple(archive_y.shape)}")
    print(f"reference_point = {ref_point.astype(float).tolist()} | archive_hv = {archive_hv:.6f}")

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

    print(
        f"offspring_x shape = {tuple(offspring_x.shape)} | "
        f"true_y shape = {tuple(true_offspring_y.shape)} | "
        f"pred_y shape = {tuple(offspring_pred.shape)}"
    )

    best_idx = -1
    best_next_hv = -float("inf")
    for idx in range(int(offspring_x.shape[0])):
        next_hv = float(hypervolume(np.vstack([archive_y, true_offspring_y[idx : idx + 1]]), ref_point))
        if next_hv > best_next_hv:
            best_idx = int(idx)
            best_next_hv = float(next_hv)

    print(
        f"best_single_add_idx = {best_idx} | "
        f"best_next_hv = {best_next_hv:.6f} | "
        f"best_hv_improvement = {best_next_hv - archive_hv:.6f}"
    )
    if best_idx >= 0:
        print(f"best_single_add_true_y = {true_offspring_y[best_idx].astype(float).tolist()}")
        print(f"best_single_add_pred_y = {offspring_pred[best_idx].astype(float).tolist()}")

    for idx in range(int(offspring_x.shape[0])):
        print(
            f"offspring[{idx:03d}] | "
            f"x = {offspring_x[idx].astype(float).tolist()} | "
            f"true_y = {true_offspring_y[idx].astype(float).tolist()} | "
            f"pred_y = {offspring_pred[idx].astype(float).tolist()}"
        )


if __name__ == "__main__":
    main()
