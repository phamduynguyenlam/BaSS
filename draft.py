from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from nsga2_solver import run_surrogate_nsga2
from nsga3_solver import run_surrogate_nsga3
from problem.problem import SUPPORTED_PROBLEMS, make_problem
from ref_points_hv import get_reference_point
from reward import hypervolume
from surrogate.surrogate_model import fit_tabpfn_surrogate
from tester import (
    build_surrogate,
    latin_hypercube_sample,
    make_nsga2_problem_adapter,
    surrogate_or_models_for_nsga2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample 80 LHS points, fit a surrogate, run NSGA-II on the surrogate, and print each offspring's true and predicted objective values."
    )
    parser.add_argument("--problem", type=str, default="DTLZ6", choices=SUPPORTED_PROBLEMS)
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--init_fe", type=int, default=80)
    parser.add_argument("--offspring_size", type=int, default=80)
    parser.add_argument("--surrogate_nsga_steps", type=int, default=30)
    parser.add_argument("--nsga3", action="store_true")
    parser.add_argument("--surrogate_model", type=str, default="tabpfn", choices=["gp", "gp2", "tabpfn"])
    parser.add_argument("--ensemble_model", type=int, default=8)
    parser.add_argument("--gp_nu", type=float, default=5.0)
    parser.add_argument("--beta", type=float, default=0.1)
    return parser.parse_args()


class LowerConfidenceBoundSurrogate:
    def __init__(self, base_surrogate, beta: float):
        self.base_surrogate = base_surrogate
        self.beta = float(beta)

    def predict_mean(self, x: np.ndarray) -> np.ndarray:
        mean, std = self.base_surrogate.predict_mean_std(np.asarray(x, dtype=np.float32))
        return (np.asarray(mean, dtype=np.float32) - self.beta * np.asarray(std, dtype=np.float32)).astype(np.float32)

    def __getattr__(self, name: str):
        return getattr(self.base_surrogate, name)


def make_logger(args: argparse.Namespace):
    log_dir = Path("draft_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / (
        f"draft_{str(args.surrogate_model).lower()}_{str(args.problem).lower()}_"
        f"d{int(args.dim)}_seed{int(args.seed)}_{timestamp}.txt"
    )
    log_fp = log_path.open("w", encoding="utf-8")

    def _log(message: str) -> None:
        text = str(message)
        print(text)
        log_fp.write(text + "\n")
        log_fp.flush()

    return _log, log_fp, log_path


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))
    log, log_fp, log_path = make_logger(args)

    try:
        problem_name = str(args.problem)
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

        log(f"log_path = {str(log_path.resolve())}")
        log(f"problem = {problem_name} | dim = {int(args.dim)} | init_fe = {int(args.init_fe)}")
        log(f"archive_x shape = {tuple(archive_x.shape)} | archive_y shape = {tuple(archive_y.shape)}")
        log(f"reference_point = {ref_point.astype(float).tolist()} | archive_hv = {archive_hv:.6f}")
        log(f"surrogate_model = {str(args.surrogate_model).lower()} | beta = {float(args.beta):.4f}")
        log(f"surrogate_nsga_steps = {int(args.surrogate_nsga_steps)}")
        log(f"candidate_solver = {'nsga3' if bool(args.nsga3) else 'nsga2'}")

        if str(args.surrogate_model).lower() == "tabpfn":
            base_surrogate = fit_tabpfn_surrogate(
                archive_x=archive_x,
                archive_y=archive_y,
                device=str(args.device),
                n_estimators=int(args.ensemble_model),
            )
            surrogate = LowerConfidenceBoundSurrogate(base_surrogate, beta=float(args.beta))
        else:
            surrogate = build_surrogate(args, archive_x, archive_y)
        nsga2_surrogate, nsga2_models = surrogate_or_models_for_nsga2(surrogate)

        solver = run_surrogate_nsga3 if bool(args.nsga3) else run_surrogate_nsga2
        offspring_x, offspring_pred = solver(
            surrogate=nsga2_surrogate,
            gps=nsga2_models,
            problem=nsga2_problem,
            archive_x=archive_x,
            pop_size=int(args.offspring_size),
            surrogate_nsga_steps=int(args.surrogate_nsga_steps),
            seed=int(args.seed),
        )
        offspring_x = np.asarray(offspring_x, dtype=np.float32)
        offspring_pred = np.asarray(offspring_pred, dtype=np.float32)
        true_offspring_y = np.asarray(problem.evaluate(offspring_x), dtype=np.float32)
        if str(args.surrogate_model).lower() == "tabpfn":
            pred_mean, pred_std = base_surrogate.predict_mean_std(offspring_x)
            pred_mean = np.asarray(pred_mean, dtype=np.float32)
            pred_std = np.asarray(pred_std, dtype=np.float32)
            pred_lcb = np.asarray(offspring_pred, dtype=np.float32)
        else:
            pred_mean = np.asarray(offspring_pred, dtype=np.float32)
            pred_std = np.zeros_like(pred_mean, dtype=np.float32)
            pred_lcb = np.asarray(offspring_pred, dtype=np.float32)

        log(
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

        log(
            f"best_single_add_idx = {best_idx} | "
            f"best_next_hv = {best_next_hv:.6f} | "
            f"best_hv_improvement = {best_next_hv - archive_hv:.6f}"
        )
        if best_idx >= 0:
            log(f"best_single_add_true_y = {true_offspring_y[best_idx].astype(float).tolist()}")
            log(f"best_single_add_pred_mean = {pred_mean[best_idx].astype(float).tolist()}")
            log(f"best_single_add_pred_lcb = {pred_lcb[best_idx].astype(float).tolist()}")
            if str(args.surrogate_model).lower() == "tabpfn":
                log(f"best_single_add_pred_std = {pred_std[best_idx].astype(float).tolist()}")

        for idx in range(int(offspring_x.shape[0])):
            if str(args.surrogate_model).lower() == "tabpfn":
                log(
                    f"offspring[{idx:03d}] | "
                    f"x = {offspring_x[idx].astype(float).tolist()} | "
                    f"true_y = {true_offspring_y[idx].astype(float).tolist()} | "
                    f"pred_mean = {pred_mean[idx].astype(float).tolist()} | "
                    f"pred_std = {pred_std[idx].astype(float).tolist()} | "
                    f"pred_lcb = {pred_lcb[idx].astype(float).tolist()}"
                )
            else:
                log(
                    f"offspring[{idx:03d}] | "
                    f"x = {offspring_x[idx].astype(float).tolist()} | "
                    f"true_y = {true_offspring_y[idx].astype(float).tolist()} | "
                    f"pred_y = {pred_mean[idx].astype(float).tolist()}"
                )
    finally:
        log_fp.close()


if __name__ == "__main__":
    main()
