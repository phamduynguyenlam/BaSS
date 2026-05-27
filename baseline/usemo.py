from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from problem.problem import SUPPORTED_PROBLEMS, make_problem
from ref_points_hv import get_reference_point
from reward import hypervolume, pareto_front
from solver.usemo_solver import run_surrogate_usemo


@dataclass
class USEMOStepRecord:
    step: int
    fe: int
    hv: float
    front_size: int
    selected_idx: int
    selected_uncertainty: float
    selected_true_y: list[float]
    selected_pred_y: list[float]
    selected_sigma: list[float]


def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fp = log_path.open("w", encoding="utf-8")

    def _log(message: str) -> None:
        text = str(message)
        print(text)
        fp.write(text + "\n")
        fp.flush()

    return _log, fp


def default_log_path(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = (
        f"usemo_{str(args.problem).lower()}_{str(args.nsga_af).lower()}_"
        f"seed{int(args.seed)}_{timestamp}.txt"
    )
    return Path("baseline_logs") / stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standalone USeMO baseline with uncertainty-based candidate selection."
    )
    parser.add_argument("--problem", type=str, default="DTLZ6", choices=SUPPORTED_PROBLEMS)
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_fe", type=int, default=120)
    parser.add_argument("--init_fe", type=int, default=80)
    parser.add_argument("--surrogate_nsga_steps", type=int, default=100)
    parser.add_argument("--offspring_size", type=int, default=80)
    parser.add_argument("--nsga_af", type=str, default="ei", choices=["lcb", "ei"])
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    if int(args.max_fe) <= int(args.init_fe):
        raise ValueError(f"max_fe must be greater than init_fe, got {args.max_fe} and {args.init_fe}.")
    return args


def latin_hypercube_sample(
    *,
    n_samples: int,
    dim: int,
    lower: float | np.ndarray,
    upper: float | np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    lower_arr = np.asarray(lower, dtype=np.float32).reshape(-1)
    upper_arr = np.asarray(upper, dtype=np.float32).reshape(-1)
    if lower_arr.size == 1:
        lower_arr = np.repeat(lower_arr, int(dim))
    if upper_arr.size == 1:
        upper_arr = np.repeat(upper_arr, int(dim))

    cut = np.linspace(0.0, 1.0, int(n_samples) + 1, dtype=np.float32)
    u = rng.random((int(n_samples), int(dim)), dtype=np.float32)
    points = cut[:-1, None] + u * (cut[1:, None] - cut[:-1, None])

    lhs = np.empty_like(points, dtype=np.float32)
    for j in range(int(dim)):
        lhs[:, j] = points[rng.permutation(int(n_samples)), j]

    return (lower_arr + lhs * (upper_arr - lower_arr)).astype(np.float32)


def make_problem_adapter(problem, n_obj: int):
    class _ProblemAdapter:
        def __init__(self):
            self.n_var = int(problem.dim)
            self.n_obj = int(n_obj)
            self.xl = np.full(int(problem.dim), float(problem.lower), dtype=np.float32)
            self.xu = np.full(int(problem.dim), float(problem.upper), dtype=np.float32)

    return _ProblemAdapter()


def aggregate_uncertainty(sigma: np.ndarray) -> np.ndarray:
    sigma_arr = np.asarray(sigma, dtype=np.float32)
    return np.linalg.norm(sigma_arr, axis=1).astype(np.float32)


def main() -> None:
    args = parse_args()
    problem = make_problem(args.problem, dim=int(args.dim))
    log_path = Path(args.log_path) if args.log_path else default_log_path(args)
    logger, log_fp = make_logger(log_path)

    try:
        logger(f"test_log_path = {log_path.resolve()}")
        archive_x = latin_hypercube_sample(
            n_samples=int(args.init_fe),
            dim=int(problem.dim),
            lower=problem.lower,
            upper=problem.upper,
            seed=int(args.seed),
        )
        archive_y = np.asarray(problem.evaluate(archive_x), dtype=np.float32)
        n_obj = int(archive_y.shape[1])
        ref_point = get_reference_point(args.problem, n_obj=n_obj)
        nsga_problem = make_problem_adapter(problem, n_obj)

        logger(f"reference_point = {ref_point.tolist()} (from ref_points_hv.py)")
        logger("candidate_solver = usemo")
        logger(f"nsga_af = {str(args.nsga_af).lower()} | beta = {float(args.beta):.4f}")
        logger(f"surrogate_nsga_steps = {int(args.surrogate_nsga_steps)}")

        hv_history: list[float] = [hypervolume(archive_y, ref_point)]
        records: list[USEMOStepRecord] = []
        logger(f"iter 0 | front = {int(pareto_front(archive_y).shape[0])} | HV = {hv_history[-1]:.6f}")

        n_evo_steps = int(args.max_fe) - int(args.init_fe)
        for step in range(n_evo_steps):
            pareto_x, pareto_pred, pareto_sigma = run_surrogate_usemo(
                problem=nsga_problem,
                archive_x=archive_x,
                archive_y=archive_y,
                pop_size=int(args.offspring_size),
                surrogate_nsga_steps=int(args.surrogate_nsga_steps),
                seed=int(args.seed) + int(step),
                acquisition=str(args.nsga_af).lower(),
                beta=float(args.beta),
            )
            uncertainty = aggregate_uncertainty(pareto_sigma)
            selected_idx = int(np.argmax(uncertainty))
            selected_x = pareto_x[selected_idx : selected_idx + 1]
            selected_true_y = np.asarray(problem.evaluate(selected_x), dtype=np.float32).reshape(1, -1)

            archive_x = np.vstack([archive_x, selected_x]).astype(np.float32)
            archive_y = np.vstack([archive_y, selected_true_y]).astype(np.float32)

            hv = hypervolume(archive_y, ref_point)
            hv_history.append(float(hv))
            front_size = int(pareto_front(archive_y).shape[0])
            fe = int(args.init_fe) + step + 1
            record = USEMOStepRecord(
                step=step + 1,
                fe=fe,
                hv=float(hv),
                front_size=front_size,
                selected_idx=selected_idx,
                selected_uncertainty=float(uncertainty[selected_idx]),
                selected_true_y=selected_true_y.reshape(-1).astype(np.float32).tolist(),
                selected_pred_y=pareto_pred[selected_idx].reshape(-1).astype(np.float32).tolist(),
                selected_sigma=pareto_sigma[selected_idx].reshape(-1).astype(np.float32).tolist(),
            )
            records.append(record)
            logger(
                f"iter {record.step} | front = {record.front_size} | HV = {record.hv:.6f}"
            )

        summary = {
            "method": "usemo",
            "problem": str(args.problem),
            "dim": int(args.dim),
            "seed": int(args.seed),
            "init_fe": int(args.init_fe),
            "max_fe": int(args.max_fe),
            "offspring_size": int(args.offspring_size),
            "surrogate_nsga_steps": int(args.surrogate_nsga_steps),
            "acquisition": str(args.nsga_af).lower(),
            "beta": float(args.beta),
            "reference_point": ref_point.astype(np.float32).tolist(),
            "initial_hv": float(hv_history[0]),
            "final_hv": float(hv_history[-1]),
            "hv_history": [float(v) for v in hv_history],
            "history": [asdict(record) for record in records],
        }
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            logger(f"output_json = {output_path.resolve()}")
    finally:
        log_fp.close()


if __name__ == "__main__":
    main()
