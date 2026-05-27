from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from solver.nsga3_solver import run_surrogate_nsga3
from problem.problem import SUPPORTED_PROBLEMS, make_problem
from ref_points_hv import get_reference_point
from reward import hypervolume, pareto_front
from surrogate.surrogate_model import TabPFNMinMaxSurrogate, fit_tabpfn_surrogate
from tester import latin_hypercube_sample, make_nsga2_problem_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Greedy surrogate-assisted test with TabPFN: initialize 80 LHS points, "
            "run NSGA-III on the surrogate, true-evaluate all offspring, and add the "
            "candidate with the largest HV improvement for 40 iterations."
        )
    )
    parser.add_argument("--problem", type=str, default="ZDT1", choices=SUPPORTED_PROBLEMS)
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--init_fe", type=int, default=80)
    parser.add_argument("--max_fe", type=int, default=120)
    parser.add_argument("--offspring_size", type=int, default=80)
    parser.add_argument("--surrogate_nsga_steps", type=int, default=100)
    parser.add_argument("--ensemble_model", type=int, default=8)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()
    if int(args.max_fe) <= int(args.init_fe):
        raise ValueError(f"max_fe must be greater than init_fe, got {args.max_fe} and {args.init_fe}.")
    return args


def select_best_hv_candidate(
    *,
    archive_y: np.ndarray,
    offspring_x: np.ndarray,
    offspring_true_y: np.ndarray,
    ref_point: np.ndarray,
) -> tuple[int, float, float]:
    archive_y_arr = np.asarray(archive_y, dtype=np.float32)
    offspring_true_arr = np.asarray(offspring_true_y, dtype=np.float32)
    current_hv = float(hypervolume(archive_y_arr, ref_point))

    best_idx = -1
    best_hv = current_hv
    best_delta = -float("inf")
    for idx in range(int(offspring_x.shape[0])):
        candidate_y = offspring_true_arr[idx : idx + 1]
        next_hv = float(hypervolume(np.vstack([archive_y_arr, candidate_y]), ref_point))
        delta_hv = float(next_hv - current_hv)
        if delta_hv > best_delta:
            best_idx = int(idx)
            best_hv = next_hv
            best_delta = delta_hv

    if best_idx < 0:
        raise RuntimeError("Failed to select a best offspring from the NSGA-III candidate set.")
    return best_idx, best_hv, best_delta


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
    nsga_problem = make_nsga2_problem_adapter(problem, n_obj)
    n_evo_steps = int(args.max_fe) - int(args.init_fe)

    hv_history = [float(hypervolume(archive_y, ref_point))]
    fe_history = [int(args.init_fe)]
    history: list[dict[str, object]] = []

    print(f"reference_point = {ref_point.tolist()} (from ref_points_hv.py)")
    print(
        f"iter 0 | archive = {int(archive_x.shape[0])} | front = {int(pareto_front(archive_y).shape[0])} | "
        f"HV = {hv_history[-1]:.6f}"
    )

    surrogate: TabPFNMinMaxSurrogate | None = fit_tabpfn_surrogate(
        archive_x=archive_x,
        archive_y=archive_y,
        device=str(args.device),
        n_estimators=int(args.ensemble_model),
    )

    for step in range(n_evo_steps):
        offspring_x, offspring_pred = run_surrogate_nsga3(
            surrogate=surrogate,
            gps=None,
            problem=nsga_problem,
            archive_x=archive_x,
            pop_size=int(args.offspring_size),
            surrogate_nsga_steps=int(args.surrogate_nsga_steps),
            seed=int(args.seed) + int(step),
        )
        offspring_x = np.asarray(offspring_x, dtype=np.float32)
        offspring_pred = np.asarray(offspring_pred, dtype=np.float32)
        offspring_true_y = np.asarray(problem.evaluate(offspring_x), dtype=np.float32)

        best_idx, next_hv, delta_hv = select_best_hv_candidate(
            archive_y=archive_y,
            offspring_x=offspring_x,
            offspring_true_y=offspring_true_y,
            ref_point=ref_point,
        )
        selected_x = offspring_x[best_idx : best_idx + 1]
        selected_pred = offspring_pred[best_idx]
        selected_true = offspring_true_y[best_idx : best_idx + 1]

        archive_x = np.vstack([archive_x, selected_x]).astype(np.float32)
        archive_y = np.vstack([archive_y, selected_true]).astype(np.float32)
        hv_history.append(float(next_hv))
        fe_history.append(int(args.init_fe) + step + 1)

        history.append(
            {
                "step": int(step + 1),
                "fe": int(fe_history[-1]),
                "selected_index": int(best_idx),
                "archive_size": int(archive_x.shape[0]),
                "front_size": int(pareto_front(archive_y).shape[0]),
                "hv": float(next_hv),
                "hv_improvement": float(delta_hv),
                "selected_x": selected_x.reshape(-1).astype(float).tolist(),
                "selected_surrogate_y": np.asarray(selected_pred, dtype=np.float32).astype(float).tolist(),
                "selected_true_y": selected_true.reshape(-1).astype(float).tolist(),
            }
        )
        print(
            f"iter {step + 1} | archive = {int(archive_x.shape[0])} | "
            f"front = {history[-1]['front_size']} | HV = {next_hv:.6f} | "
            f"hv_improvement = {delta_hv:.6f}"
        )

        surrogate = fit_tabpfn_surrogate(
            archive_x=archive_x,
            archive_y=archive_y,
            device=str(args.device),
            n_estimators=int(args.ensemble_model),
            existing_surrogate=surrogate,
        )

    summary = {
        "problem": str(args.problem),
        "dim": int(args.dim),
        "seed": int(args.seed),
        "device": str(args.device),
        "surrogate_model": "tabpfn",
        "candidate_solver": "nsga3",
        "init_fe": int(args.init_fe),
        "max_fe": int(args.max_fe),
        "evolution_fe": int(n_evo_steps),
        "offspring_size": int(args.offspring_size),
        "surrogate_nsga_steps": int(args.surrogate_nsga_steps),
        "ensemble_model": int(args.ensemble_model),
        "reference_point": ref_point.astype(float).tolist(),
        "final_hv": float(hv_history[-1]),
        "final_front_size": int(pareto_front(archive_y).shape[0]),
        "archive_size": int(archive_x.shape[0]),
        "fe_history": [int(v) for v in fe_history],
        "hv_history": [float(v) for v in hv_history],
        "history": history,
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"final_hv = {summary['final_hv']:.6f} | final_front = {summary['final_front_size']} | archive = {summary['archive_size']}")
    print(
        json.dumps(
            {
                "problem": summary["problem"],
                "dim": summary["dim"],
                "seed": summary["seed"],
                "surrogate_model": summary["surrogate_model"],
                "candidate_solver": summary["candidate_solver"],
                "surrogate_nsga_steps": summary["surrogate_nsga_steps"],
                "ensemble_model": summary["ensemble_model"],
                "final_hv": summary["final_hv"],
                "final_front_size": summary["final_front_size"],
                "archive_size": summary["archive_size"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
