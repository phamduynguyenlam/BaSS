from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problem.problem import make_problem
from ref_points_hv import get_reference_point
from reward import hypervolume, pareto_front
from surrogate.surrogate_model import fit_tabpfn_surrogate, predict_multi_context
from trainer import build_training_env_specs


DEFAULT_PROBLEMS = [
    "ZDT1",
    "ZDT2",
    "ZDT3",
    "DTLZ2",
    "DTLZ3",
    "DTLZ4",
    "DTLZ5",
    "DTLZ6",
    "DTLZ7",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit TabPFN on an 80-point archive, evolve 80 surrogate offspring, evaluate them with the true problem, and plot Pareto fronts."
    )
    parser.add_argument("--problem", type=str, default="ZDT1", choices=DEFAULT_PROBLEMS)
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--training_set", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--num_requests", type=int, default=24)
    parser.add_argument("--archive_size", type=int, default=80)
    parser.add_argument("--offspring_size", type=int, default=80)
    parser.add_argument("--surrogate_nsga_steps", type=int, default=20)
    parser.add_argument("--num_thread", type=int, default=12)
    parser.add_argument("--mutation_sigma", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_plot", type=str, default=None)
    return parser.parse_args()


def latin_hypercube_sample(
    *,
    lower: float | np.ndarray,
    upper: float | np.ndarray,
    n_samples: int,
    dim: int,
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


def load_true_pareto_front(problem_name: str, dim: int, n_obj: int, n_points: int = 400) -> np.ndarray | None:
    try:
        from pymoo.problems import get_problem
    except Exception:
        return None

    key = str(problem_name).lower()
    try:
        pymoo_problem = get_problem(key, n_var=int(dim), n_obj=int(n_obj))
    except TypeError:
        try:
            pymoo_problem = get_problem(key, n_var=int(dim))
        except Exception:
            return None
    except Exception:
        return None

    try:
        pareto = pymoo_problem.pareto_front(n_pareto_points=int(n_points))
    except TypeError:
        try:
            pareto = pymoo_problem.pareto_front()
        except Exception:
            return None
    except Exception:
        return None

    if pareto is None:
        return None
    pareto = np.asarray(pareto, dtype=np.float32)
    if pareto.ndim != 2 or pareto.shape[1] < int(n_obj):
        return None
    return pareto[:, : int(n_obj)]


def plot_fronts(
    *,
    problem_name: str,
    dim: int,
    archive_y: np.ndarray,
    offspring_true_y: np.ndarray,
    merged_y: np.ndarray,
    true_front: np.ndarray | None,
    save_plot: str | None,
) -> None:
    archive_front = pareto_front(archive_y)
    offspring_front = pareto_front(offspring_true_y)
    merged_front = pareto_front(merged_y)
    n_obj = int(archive_y.shape[1])

    fig = plt.figure(figsize=(11, 5))
    if n_obj == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(archive_front[:, 0], archive_front[:, 1], archive_front[:, 2], s=28, alpha=0.85, label="Archive PF")
        ax.scatter(
            offspring_front[:, 0],
            offspring_front[:, 1],
            offspring_front[:, 2],
            s=28,
            alpha=0.85,
            label="80 Offspring PF",
        )
        ax.scatter(
            merged_front[:, 0],
            merged_front[:, 1],
            merged_front[:, 2],
            s=16,
            alpha=0.40,
            label="Merged PF",
        )
        if true_front is not None and true_front.shape[1] >= 3:
            ax.scatter(true_front[:, 0], true_front[:, 1], true_front[:, 2], s=8, alpha=0.20, label="True PF")
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.set_zlabel("f3")
    else:
        ax = fig.add_subplot(111)
        ax.scatter(archive_front[:, 0], archive_front[:, 1], s=32, alpha=0.85, label="Archive PF")
        ax.scatter(offspring_front[:, 0], offspring_front[:, 1], s=32, alpha=0.85, label="80 Offspring PF")
        ax.scatter(merged_front[:, 0], merged_front[:, 1], s=18, alpha=0.40, label="Merged PF")
        if true_front is not None and true_front.shape[1] >= 2:
            order = np.argsort(true_front[:, 0])
            ax.plot(true_front[order, 0], true_front[order, 1], linewidth=2.0, label="True PF")
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.grid(True, alpha=0.3)

    ax.set_title(f"{problem_name} {dim}D | Archive / Offspring / True PF")
    ax.legend()
    fig.tight_layout()

    if save_plot:
        plot_path = Path(save_plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=180, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def run_surrogate_nsga2_with_logging(
    *,
    surrogate,
    archive_x: np.ndarray,
    n_generations: int,
    n_candidates: int,
    seed: int,
    lower: np.ndarray,
    upper: np.ndarray,
    mutation_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize

    parents = np.asarray(archive_x, dtype=np.float32)
    pop_size = int(n_candidates)
    rng = np.random.default_rng(int(seed))

    init_idx = rng.integers(0, int(parents.shape[0]), size=pop_size, endpoint=False)
    init_x = parents[np.asarray(init_idx, dtype=np.int64)]
    init_x = init_x + rng.normal(loc=0.0, scale=float(mutation_sigma), size=init_x.shape).astype(np.float32)
    init_x = np.clip(init_x, lower, upper).astype(np.float32)

    class _SurrogateProblem(Problem):
        def __init__(self):
            init_y = np.asarray(surrogate.predict(init_x), dtype=np.float32)
            super().__init__(
                n_var=int(init_x.shape[1]),
                n_obj=int(init_y.shape[1]),
                n_ieq_constr=0,
                xl=lower,
                xu=upper,
                elementwise_evaluation=False,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = np.asarray(surrogate.predict(np.asarray(x, dtype=np.float32)), dtype=np.float32)

    gen_started = time.perf_counter()

    def _callback(algorithm) -> None:
        pop_f = np.asarray(algorithm.pop.get("F"), dtype=np.float32)
        surrogate_front = pareto_front(pop_f)
        elapsed = time.perf_counter() - gen_started
        print(
            f"Gen {int(algorithm.n_gen):02d}/{int(n_generations)} | "
            f"population={int(pop_f.shape[0])} | surrogate_front0={int(surrogate_front.shape[0])} | "
            f"elapsed={elapsed:.2f}s"
        )

    algorithm = NSGA2(pop_size=pop_size, eliminate_duplicates=True)
    res = minimize(
        _SurrogateProblem(),
        algorithm,
        termination=("n_gen", int(n_generations)),
        seed=int(seed),
        save_history=False,
        verbose=False,
        X=init_x,
        callback=_callback,
    )

    pop = getattr(res, "pop", None)
    if pop is None:
        offspring_x = np.asarray(res.X, dtype=np.float32)
        offspring_pred = np.asarray(res.F, dtype=np.float32)
    else:
        offspring_x = np.asarray(pop.get("X"), dtype=np.float32)
        offspring_pred = np.asarray(pop.get("F"), dtype=np.float32)

    if offspring_x.shape[0] < pop_size:
        idx = np.arange(int(pop_size), dtype=np.int64) % int(offspring_x.shape[0])
        offspring_x = offspring_x[idx]
        offspring_pred = offspring_pred[idx]
    elif offspring_x.shape[0] > pop_size:
        offspring_x = offspring_x[:pop_size]
        offspring_pred = offspring_pred[:pop_size]

    return offspring_x, offspring_pred


def _broadcast_bounds(problem, dim: int) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(problem.lower, dtype=np.float32).reshape(-1)
    upper = np.asarray(problem.upper, dtype=np.float32).reshape(-1)
    if lower.size == 1:
        lower = np.repeat(lower, int(dim))
    if upper.size == 1:
        upper = np.repeat(upper, int(dim))
    return lower, upper


def run_multi_context_parallel_demo(args: argparse.Namespace) -> None:
    run_started = time.perf_counter()
    env_specs = build_training_env_specs(args.problem, int(args.training_set))
    env_specs = env_specs[: int(args.num_requests)]
    if len(env_specs) == 0:
        raise ValueError("No environment specs available for multi-context demo.")

    archives_x: list[np.ndarray] = []
    archives_y: list[np.ndarray] = []
    surrogates = []
    queries: list[np.ndarray] = []

    fit_started = time.perf_counter()
    for env_idx, spec in enumerate(env_specs):
        problem = make_problem(str(spec["problem_name"]), dim=int(spec["dim"]))
        archive_x = latin_hypercube_sample(
            lower=problem.lower,
            upper=problem.upper,
            n_samples=int(args.archive_size),
            dim=int(spec["dim"]),
            seed=int(args.seed) + 1000 * env_idx,
        )
        archive_y = np.asarray(problem.evaluate(archive_x), dtype=np.float32)
        lower, upper = _broadcast_bounds(problem, int(spec["dim"]))
        query_x = latin_hypercube_sample(
            lower=lower,
            upper=upper,
            n_samples=int(args.offspring_size),
            dim=int(spec["dim"]),
            seed=int(args.seed) + 1000 * env_idx + 1,
        )

        surrogate = fit_tabpfn_surrogate(
            archive_x=archive_x,
            archive_y=archive_y,
            device=str(args.device),
        )
        archives_x.append(archive_x)
        archives_y.append(archive_y)
        surrogates.append(surrogate)
        queries.append(query_x)
    fit_elapsed = time.perf_counter() - fit_started

    infer_started = time.perf_counter()
    (pred_means, pred_stds), profile = predict_multi_context(
        surrogates,
        queries,
        return_std=True,
        return_profile=True,
        num_threads=int(args.num_thread),
    )
    infer_elapsed = time.perf_counter() - infer_started

    total_points = sum(int(query.shape[0]) for query in queries)
    max_dim = max(int(query.shape[1]) for query in queries)
    total_objective_contexts = sum(int(archive_y.shape[1]) for archive_y in archives_y)

    print(
        f"Parallel TabPFN demo | heldout={args.problem} | training_set={args.training_set} | "
        f"requests={len(env_specs)} | device={args.device} | num_thread={int(args.num_thread)} | "
        f"points={total_points} | max_dim={max_dim} | "
        f"objective_contexts={total_objective_contexts}"
    )
    print(
        f"Fit time: {fit_elapsed:.3f}s | "
        f"Infer time: {infer_elapsed:.3f}s | "
        f"query_transform_sec={float(profile.get('query_transform_sec', 0.0)):.3f} | "
        f"batch_prepare_sec={float(profile.get('batch_prepare_sec', 0.0)):.3f} | "
        f"gpu_forward_sec={float(profile.get('gpu_forward_sec', 0.0)):.3f} | "
        f"postprocess_sec={float(profile.get('postprocess_sec', 0.0)):.3f} | "
        f"fallback_used={int(round(float(profile.get('fallback_used', 0.0))))}"
    )
    fallback_reason = str(profile.get("fallback_reason", "") or "")
    if fallback_reason:
        print(f"Fallback reason: {fallback_reason}")

    for env_idx, (spec, archive_y, mean_y, std_y) in enumerate(zip(env_specs, archives_y, pred_means, pred_stds), start=1):
        print(
            f"[{env_idx:02d}] {str(spec['problem_name']).upper()}-{int(spec['dim'])}D | "
            f"archive_obj={int(archive_y.shape[1])} | "
            f"query_shape={tuple(queries[env_idx - 1].shape)} | "
            f"mean_shape={tuple(np.asarray(mean_y).shape)} | "
            f"std_shape={tuple(np.asarray(std_y).shape)}"
        )

    print(f"Runtime total: {time.perf_counter() - run_started:.3f}s")


def main() -> None:
    args = parse_args()
    if int(args.num_requests) > 1:
        run_multi_context_parallel_demo(args)
        return

    run_started = time.perf_counter()

    problem = make_problem(args.problem, dim=int(args.dim))
    archive_x = latin_hypercube_sample(
        lower=problem.lower,
        upper=problem.upper,
        n_samples=int(args.archive_size),
        dim=int(args.dim),
        seed=int(args.seed),
    )
    archive_y = np.asarray(problem.evaluate(archive_x), dtype=np.float32)
    n_obj = int(archive_y.shape[1])
    ref_point = np.asarray(get_reference_point(args.problem, n_obj=n_obj), dtype=np.float32)

    print(
        f"{args.problem} | dim={args.dim} | archive_size={args.archive_size} | "
        f"offspring_size={args.offspring_size} | surrogate_nsga_steps={args.surrogate_nsga_steps}"
    )
    print(f"Initial archive PF size: {pareto_front(archive_y).shape[0]}")
    print(f"Initial archive HV: {hypervolume(archive_y, ref_point):.6f}")

    surrogate = fit_tabpfn_surrogate(
        archive_x=archive_x,
        archive_y=archive_y,
        device=str(args.device),
    )

    lower, upper = _broadcast_bounds(problem, int(args.dim))

    offspring_x, offspring_pred = run_surrogate_nsga2_with_logging(
        surrogate=surrogate,
        archive_x=archive_x,
        n_generations=int(args.surrogate_nsga_steps),
        n_candidates=int(args.offspring_size),
        seed=int(args.seed) + 1000,
        lower=lower,
        upper=upper,
        mutation_sigma=float(args.mutation_sigma),
    )
    offspring_true_y = np.asarray(problem.evaluate(offspring_x), dtype=np.float32)
    merged_y = np.vstack([archive_y, offspring_true_y]).astype(np.float32)

    archive_hv = hypervolume(archive_y, ref_point)
    offspring_hv = hypervolume(offspring_true_y, ref_point)
    merged_hv = hypervolume(merged_y, ref_point)

    print(f"Surrogate offspring pool size: {offspring_x.shape[0]}")
    print(f"Surrogate offspring PF size (pred): {pareto_front(offspring_pred).shape[0]}")
    print(f"True offspring PF size: {pareto_front(offspring_true_y).shape[0]}")
    print(f"Archive HV: {archive_hv:.6f}")
    print(f"80 offspring true HV: {offspring_hv:.6f}")
    print(f"Merged archive+offspring HV: {merged_hv:.6f}")
    print(f"Runtime without plot: {time.perf_counter() - run_started:.2f}s")

    true_front = load_true_pareto_front(args.problem, int(args.dim), n_obj)
    plot_fronts(
        problem_name=str(args.problem),
        dim=int(args.dim),
        archive_y=archive_y,
        offspring_true_y=offspring_true_y,
        merged_y=merged_y,
        true_front=true_front,
        save_plot=args.save_plot,
    )


if __name__ == "__main__":
    main()
