from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from agents.db_saea import DBSAEAAgent
from agents.meta import Disc, DiscAF
from infill import (
    EPDIExploitation,
    EPDIExploration,
    ExpectedHypervolumeImprovement,
    NDA,
    NDPBIConvergence,
    NDPBIDiversity,
)
from solver.nsga2_solver import run_surrogate_nsga2
from solver.nsga3_solver import run_surrogate_nsga3
from solver.moead_solver import run_surrogate_moead
from solver.usemo_solver import run_surrogate_usemo
from problem.problem import SUPPORTED_PROBLEMS, make_problem
from ref_points_hv import get_reference_point, get_true_pareto_hv
from reward import hypervolume, pareto_front, reward_scheme_1, reward_scheme_2, reward_scheme_3
from surrogate.gp import fit_gp_surrogates
from surrogate.surrogate_model import (
    estimate_uncertainty,
    fit_kan_surrogates,
    fit_tabpfn_surrogate,
    KANSurrogateModel,
    surrogate_model_name,
)


def make_test_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = log_path.open("w", encoding="utf-8")

    def _log(message: str) -> None:
        text = str(message)
        print(text)
        log_fp.write(text + "\n")
        log_fp.flush()

    return _log, log_fp


def default_test_log_path(args: argparse.Namespace, *, agent_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    compare_name = resolve_compare_infill_name(args)
    compare_tag = "" if compare_name is None else f"_compare_{compare_name}"
    stem = (
        f"test_{str(agent_name).lower()}_{str(args.problem).lower()}_"
        f"{str(args.surrogate_model).lower()}_seed{int(args.seed)}{compare_tag}_{timestamp}.txt"
    )
    return Path("testing_logs") / stem


def resolve_compare_infill_name(args: argparse.Namespace) -> str | None:
    raw_value = getattr(args, "compare_infill", None)
    if raw_value is None:
        return None
    text = str(raw_value).strip().lower()
    if text == "":
        return None
    return text.replace("-", "_")


def compare_infill_display_name(name: str) -> str:
    display_map = {
        "ehvi": "EHVI",
        "nd_a": "ND-A",
        "nd_pbi_convergence": "ND-PBI-Convergence",
        "nd_pbi_diversity": "ND-PBI-Diversity",
        "epdi_exploitation": "EPDI-Exploitation",
        "epdi_exploration": "EPDI-Exploration",
    }
    key = str(name).strip().lower().replace("-", "_")
    return display_map.get(key, key.upper())


def build_compare_infill_criterion(name: str, *, ref_point: np.ndarray):
    key = str(name).strip().lower().replace("-", "_")
    if key == "ehvi":
        return ExpectedHypervolumeImprovement(ref_point=ref_point, n_samples=64)
    if key == "nd_a":
        return NDA()
    if key == "nd_pbi_convergence":
        return NDPBIConvergence()
    if key == "nd_pbi_diversity":
        return NDPBIDiversity()
    if key == "epdi_exploitation":
        return EPDIExploitation()
    if key == "epdi_exploration":
        return EPDIExploration()
    raise ValueError(f"Unsupported compare_infill: {name}")


def resolve_test_reward_scheme(args: argparse.Namespace) -> int:
    agent_pth = getattr(args, "agent_pth", None)
    if not agent_pth:
        return 1
    match = re.search(r"rs([123])", Path(str(agent_pth)).name.lower())
    if match is None:
        return 1
    return int(match.group(1))


def compute_test_reward(
    *,
    reward_scheme_id: int,
    previous_front: np.ndarray,
    selected_objectives: np.ndarray,
    ref_point: np.ndarray,
    reward_lambda: float,
    true_pareto_hv: float | None = None,
    archive_true_y: np.ndarray | None = None,
    archive_pred_y: np.ndarray | None = None,
) -> float:
    if int(reward_scheme_id) == 1:
        return float(
            reward_scheme_1(
                previous_front=previous_front,
                selected_objectives=selected_objectives,
                ref_point=ref_point,
                reward_lambda=float(reward_lambda),
            )
        )
    if int(reward_scheme_id) == 2:
        return float(
            reward_scheme_2(
                previous_front=previous_front,
                selected_objectives=selected_objectives,
                ref_point=ref_point,
                reward_lambda=float(reward_lambda),
            )
        )
    if int(reward_scheme_id) == 3:
        if true_pareto_hv is None:
            raise ValueError("reward_scheme_3 requires true_pareto_hv in tester.")
        if archive_true_y is None or archive_pred_y is None:
            raise ValueError("reward_scheme_3 requires archive_true_y and archive_pred_y in tester.")
        return float(
            reward_scheme_3(
                previous_front=previous_front,
                selected_objectives=selected_objectives,
                ref_point=ref_point,
                true_pareto_hv=float(true_pareto_hv),
                archive_true_y=np.asarray(archive_true_y, dtype=np.float32),
                archive_pred_y=np.asarray(archive_pred_y, dtype=np.float32),
            )
        )
    raise ValueError(f"Unsupported reward_scheme_id for tester: {reward_scheme_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DISC-guided surrogate-assisted optimization with 80 LHS init + 40 evolution steps."
    )
    parser.add_argument("--problem", type=str, default="ZDT1", choices=SUPPORTED_PROBLEMS)
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_fe", type=int, default=120)
    parser.add_argument("--init_fe", type=int, default=80)
    parser.add_argument("--surrogate_nsga_steps", type=int, default=100)
    parser.add_argument("--tabpfn_nsga_steps", type=int, default=None)
    parser.add_argument("--gp_nsga_steps", type=int, default=None)
    parser.add_argument("--offspring_size", type=int, default=80)
    parser.add_argument("--mutation_sigma", type=float, default=0.12)
    parser.add_argument("--logit_scale", type=float, default=5.0)
    parser.add_argument("--agent_pth", type=str, default=None)
    parser.add_argument("--random_model", action="store_true")
    parser.add_argument("--surrogate_model", type=str, default="gp", choices=["gp", "kan", "tabpfn"])
    parser.add_argument("--reward_lambda", type=float, default=10.0)
    parser.add_argument("--kan_steps", type=int, default=25)
    parser.add_argument("--kan_hidden_width", type=int, default=10)
    parser.add_argument("--kan_grid", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--nsga_af", type=str, default="mean", choices=["mean", "lcb", "ei"])
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--hybrid_nsga", action="store_true")
    parser.add_argument("--hybrid_nsga_gp", action="store_true")
    parser.add_argument("--compare_infill", type=str, default=None)
    parser.add_argument("--compare_algo", type=str, default=None, choices=["db_saea"])
    parser.add_argument("--compare_agent_pth", type=str, default=None)
    parser.add_argument("--solver", type=str, default="nsga2", choices=["nsga2", "nsga3", "moead", "usemo"])
    parser.add_argument("--pseudo_front_only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--plot_path", type=str, default=None)
    args = parser.parse_args()

    if int(args.max_fe) <= int(args.init_fe):
        raise ValueError(f"max_fe must be greater than init_fe, got {args.max_fe} and {args.init_fe}.")
    if args.compare_infill is not None and args.compare_algo is not None:
        raise ValueError("Use only one of --compare_infill or --compare_algo at a time.")
    if bool(args.hybrid_nsga) and bool(args.hybrid_nsga_gp):
        raise ValueError("Use only one of --hybrid_nsga or --hybrid_nsga_gp at a time.")
    if str(args.solver).lower() == "usemo" and str(args.nsga_af).lower() not in {"lcb", "ei"}:
        args.nsga_af = "ei"
    return args


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def resolve_agent_cls(agent_name: str):
    name = str(agent_name).strip().lower()
    if name == "disc":
        return Disc
    if name == "disc_af":
        return DiscAF
    if name == "db_saea":
        return DBSAEAAgent
    raise ValueError(f"Unsupported agent_name: {agent_name}")


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


def build_named_surrogate(args: argparse.Namespace, archive_x: np.ndarray, archive_y: np.ndarray, name: str):
    surrogate_name = str(name).lower()
    if surrogate_name == "gp":
        return fit_gp_surrogates(
            archive_x=archive_x,
            archive_y=archive_y,
            seed=int(args.seed),
            nu=float(getattr(args, "gp_nu", 5.0)),
            variant="gp",
        )

    if surrogate_name == "gp2":
        return fit_gp_surrogates(
            archive_x=archive_x,
            archive_y=archive_y,
            seed=int(args.seed),
            variant="gp2",
        )

    if surrogate_name == "gp3":
        return fit_gp_surrogates(
            archive_x=archive_x,
            archive_y=archive_y,
            variant="gp3",
            xl=np.min(np.asarray(archive_x, dtype=np.float32), axis=0),
            xu=np.max(np.asarray(archive_x, dtype=np.float32), axis=0),
            seed=int(args.seed),
            nu=float(getattr(args, "gp3_nu", 2.5)),
        )

    if surrogate_name == "tabpfn":
        return fit_tabpfn_surrogate(
            archive_x=archive_x,
            archive_y=archive_y,
            device=str(args.device),
        )

    if surrogate_name == "kan":
        kan_models = fit_kan_surrogates(
            archive_x=archive_x,
            archive_y=archive_y,
            device=str(args.device),
            kan_steps=int(args.kan_steps),
            hidden_width=int(args.kan_hidden_width),
            grid=int(args.kan_grid),
            seed=int(args.seed),
        )
        return KANSurrogateModel(models=kan_models, device=str(args.device))

    raise ValueError(f"Unsupported surrogate_model: {surrogate_name}")


def build_surrogate(args: argparse.Namespace, archive_x: np.ndarray, archive_y: np.ndarray):
    return build_named_surrogate(args, archive_x, archive_y, surrogate_model_name(args))


def surrogate_or_models_for_nsga2(surrogate: Any) -> tuple[Any | None, list[Any] | None]:
    models = getattr(surrogate, "models", None)
    if isinstance(models, list) and len(models) > 0:
        return None, models
    return surrogate, None


class _LCBObjectiveWrapper:
    def __init__(self, base_surrogate: Any, beta: float):
        self.base_surrogate = base_surrogate
        self.beta = float(beta)

    def predict_mean(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        if hasattr(self.base_surrogate, "predict_mean"):
            mean = np.asarray(self.base_surrogate.predict_mean(x_arr), dtype=np.float32)
        else:
            mean = np.asarray(self.base_surrogate.predict(x_arr), dtype=np.float32)
        try:
            std = np.asarray(self.base_surrogate.predict_std(x_arr), dtype=np.float32)
        except Exception:
            std = np.zeros_like(mean, dtype=np.float32)
        if std.ndim == 1:
            std = std.reshape(-1, 1)
        if std.shape != mean.shape:
            if std.shape[1] == 1:
                std = np.repeat(std, mean.shape[1], axis=1)
            else:
                std = np.zeros_like(mean, dtype=np.float32)
        return (mean - self.beta * std).astype(np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_mean(x)


def prepare_nsga_surrogate(args: argparse.Namespace, surrogate: Any) -> tuple[Any | None, list[Any] | None]:
    if str(getattr(args, "nsga_af", "mean")).lower() == "lcb":
        return _LCBObjectiveWrapper(surrogate, beta=float(getattr(args, "beta", 1.0))), None
    return surrogate_or_models_for_nsga2(surrogate)


def make_nsga2_problem_adapter(problem, n_obj: int):
    class _ProblemAdapter:
        def __init__(self):
            self.n_var = int(problem.dim)
            self.n_obj = int(n_obj)
            self.xl = np.full(int(problem.dim), float(problem.lower), dtype=np.float32)
            self.xu = np.full(int(problem.dim), float(problem.upper), dtype=np.float32)

    return _ProblemAdapter()


def run_surrogate_optimizer(
    *,
    args: argparse.Namespace,
    nsga_problem,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    nsga2_surrogate: Any | None,
    nsga2_models: list[Any] | None,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    solver_name = str(getattr(args, "solver", "nsga2")).lower()
    if solver_name == "nsga2":
        solver = run_surrogate_nsga2
        return solver(
            gps=nsga2_models,
            surrogate=nsga2_surrogate,
            problem=nsga_problem,
            archive_x=archive_x,
            pop_size=int(args.offspring_size),
            surrogate_nsga_steps=int(args.surrogate_nsga_steps),
            seed=int(args.seed) + int(step),
        )
    if solver_name == "nsga3":
        solver = run_surrogate_nsga3
        return solver(
            gps=nsga2_models,
            surrogate=nsga2_surrogate,
            problem=nsga_problem,
            archive_x=archive_x,
            pop_size=int(args.offspring_size),
            surrogate_nsga_steps=int(args.surrogate_nsga_steps),
            seed=int(args.seed) + int(step),
        )
    if solver_name == "moead":
        solver = run_surrogate_moead
        return solver(
            gps=nsga2_models,
            surrogate=nsga2_surrogate,
            problem=nsga_problem,
            archive_x=archive_x,
            pop_size=int(args.offspring_size),
            surrogate_nsga_steps=int(args.surrogate_nsga_steps),
            seed=int(args.seed) + int(step),
        )
    if solver_name == "usemo":
        offspring_x, offspring_pred, _ = run_surrogate_usemo(
            problem=nsga_problem,
            archive_x=archive_x,
            archive_y=archive_y,
            pop_size=int(args.offspring_size),
            surrogate_nsga_steps=args,
            seed=int(args.seed) + int(step),
            acquisition=str(getattr(args, "nsga_af", "ei")).lower(),
            beta=float(getattr(args, "beta", 1.0)),
            surrogate_model=args,
            device=args,
        )
        return offspring_x, offspring_pred
    raise ValueError(f"Unsupported solver: {solver_name}")


def resolve_surrogate_nsga_steps(args: argparse.Namespace, surrogate_name: str) -> int:
    name = str(surrogate_name).lower()
    if name == "tabpfn" and getattr(args, "tabpfn_nsga_steps", None) is not None:
        return int(args.tabpfn_nsga_steps)
    if name in {"gp", "gp2"} and getattr(args, "gp_nsga_steps", None) is not None:
        return int(args.gp_nsga_steps)
    return int(args.surrogate_nsga_steps)


def _generate_single_offspring_pool(
    *,
    args: argparse.Namespace,
    nsga_problem,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    surrogate: Any,
    step: int,
    surrogate_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_args = argparse.Namespace(**vars(args))
    local_args.surrogate_nsga_steps = int(resolve_surrogate_nsga_steps(args, surrogate_name))
    solver_name = str(getattr(args, "solver", "nsga2")).lower()
    if solver_name == "usemo":
        offspring_x, offspring_pred, offspring_sigma = run_surrogate_usemo(
            problem=nsga_problem,
            archive_x=archive_x,
            archive_y=archive_y,
            pop_size=int(args.offspring_size),
            surrogate_nsga_steps=local_args,
            seed=int(args.seed) + int(step),
            acquisition=str(getattr(args, "nsga_af", "ei")).lower(),
            beta=float(getattr(args, "beta", 1.0)),
            surrogate_model=local_args,
            device=local_args,
        )
        return (
            np.asarray(offspring_x, dtype=np.float32),
            np.asarray(offspring_pred, dtype=np.float32),
            np.asarray(offspring_sigma, dtype=np.float32),
        )

    nsga2_surrogate, nsga2_models = prepare_nsga_surrogate(args, surrogate)
    offspring_x, offspring_pred = run_surrogate_optimizer(
        args=local_args,
        nsga_problem=nsga_problem,
        archive_x=archive_x,
        archive_y=archive_y,
        nsga2_surrogate=nsga2_surrogate,
        nsga2_models=nsga2_models,
        step=step,
    )
    offspring_x = np.asarray(offspring_x, dtype=np.float32)
    offspring_pred = np.asarray(offspring_pred, dtype=np.float32)
    offspring_sigma = build_offspring_sigma(
        archive_x=archive_x,
        archive_y=archive_y,
        offspring_x=offspring_x,
        surrogate=surrogate,
    )
    return offspring_x, offspring_pred, np.asarray(offspring_sigma, dtype=np.float32)


def generate_offspring_pool(
    *,
    args: argparse.Namespace,
    nsga_problem,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    if str(getattr(args, "solver", "nsga2")).lower() == "usemo" and (
        bool(getattr(args, "hybrid_nsga", False)) or bool(getattr(args, "hybrid_nsga_gp", False))
    ):
        raise ValueError("USEMO solver does not support --hybrid_nsga or --hybrid_nsga_gp.")
    hybrid_gp_tabpfn = bool(getattr(args, "hybrid_nsga", False))
    hybrid_gp_gp2 = bool(getattr(args, "hybrid_nsga_gp", False))
    if not hybrid_gp_tabpfn and not hybrid_gp_gp2:
        surrogate = build_surrogate(args, archive_x, archive_y)
        offspring_x, offspring_pred, offspring_sigma = _generate_single_offspring_pool(
            args=args,
            nsga_problem=nsga_problem,
            archive_x=archive_x,
            archive_y=archive_y,
            surrogate=surrogate,
            step=step,
            surrogate_name=surrogate_model_name(args),
        )
        group_indices = [np.arange(int(offspring_x.shape[0]), dtype=np.int64)]
        return offspring_x, offspring_pred, offspring_sigma, group_indices

    gp_surrogate = build_named_surrogate(args, archive_x, archive_y, "gp")
    second_surrogate_name = "gp2"
    second_surrogate = build_named_surrogate(args, archive_x, archive_y, second_surrogate_name)
    gp_x, gp_pred, gp_sigma = _generate_single_offspring_pool(
        args=args,
        nsga_problem=nsga_problem,
        archive_x=archive_x,
        archive_y=archive_y,
        surrogate=gp_surrogate,
        step=int(step) * 2,
        surrogate_name="gp",
    )
    second_x, second_pred, second_sigma = _generate_single_offspring_pool(
        args=args,
        nsga_problem=nsga_problem,
        archive_x=archive_x,
        archive_y=archive_y,
        surrogate=second_surrogate,
        step=int(step) * 2 + 1,
        surrogate_name=second_surrogate_name,
    )
    merged_x = np.vstack([gp_x, second_x]).astype(np.float32)
    merged_pred = np.vstack([gp_pred, second_pred]).astype(np.float32)
    merged_sigma = np.vstack([gp_sigma, second_sigma]).astype(np.float32)
    group_indices = [
        np.arange(0, int(gp_x.shape[0]), dtype=np.int64),
        np.arange(int(gp_x.shape[0]), int(gp_x.shape[0] + second_x.shape[0]), dtype=np.int64),
    ]
    return merged_x, merged_pred, merged_sigma, group_indices


def predict_surrogate_mean(surrogate: Any, x: np.ndarray) -> np.ndarray:
    return np.asarray(surrogate.predict_mean(np.asarray(x, dtype=np.float32)), dtype=np.float32)


def predict_surrogate_std(
    surrogate: Any,
    x: np.ndarray,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    if hasattr(surrogate, "predict_std"):
        try:
            return np.asarray(surrogate.predict_std(x_arr), dtype=np.float32)
        except NotImplementedError:
            pass
    return np.zeros((int(x_arr.shape[0]), 1), dtype=np.float32)


def build_offspring_sigma(
    *,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    offspring_x: np.ndarray,
    surrogate: Any,
) -> np.ndarray:
    archive_y = np.asarray(archive_y, dtype=np.float32)

    sigma = predict_surrogate_std(surrogate, offspring_x)
    if sigma.ndim == 1:
        sigma = sigma.reshape(-1, 1)

    if sigma.shape[1] == archive_y.shape[1]:
        return sigma.astype(np.float32)

    archive_pred = predict_surrogate_mean(surrogate, archive_x)
    local_sigma = estimate_uncertainty(
        archive_x=archive_x,
        archive_y=archive_y,
        archive_pred=archive_pred,
        offspring_x=offspring_x,
    )
    if local_sigma.ndim == 1:
        local_sigma = local_sigma.reshape(-1, 1)
    if local_sigma.shape[1] != archive_y.shape[1]:
        local_sigma = np.repeat(local_sigma.mean(axis=1, keepdims=True), archive_y.shape[1], axis=1)
    return local_sigma.astype(np.float32)


def build_disc(
    args: argparse.Namespace,
    *,
    map_location: str,
    agent_name: str = "disc",
) -> tuple[Any, dict[str, float]]:
    agent_cls = resolve_agent_cls(agent_name)
    model_to_device_started_at = time.perf_counter()
    disc = agent_cls(
        hidden_dim=int(args.hidden_dim),
        n_heads=int(args.n_heads),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
        logit_scale=float(args.logit_scale),
    ).to(map_location)
    model_to_device_sec = time.perf_counter() - model_to_device_started_at
    disc.eval()

    torch_load_sec = 0.0
    load_state_dict_sec = 0.0

    if args.agent_pth and not bool(args.random_model):
        torch_load_started_at = time.perf_counter()
        state = torch.load(args.agent_pth, map_location=map_location)
        torch_load_sec = time.perf_counter() - torch_load_started_at
        state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        load_state_dict_started_at = time.perf_counter()
        disc.load_state_dict(state_dict, strict=True)
        load_state_dict_sec = time.perf_counter() - load_state_dict_started_at

    return disc, {
        "model_to_device_sec": float(model_to_device_sec),
        "torch_load_sec": float(torch_load_sec),
        "load_state_dict_sec": float(load_state_dict_sec),
    }


def select_offspring_index_epsilon_greedy(logits: torch.Tensor, epsilon: float = 0.05) -> tuple[int, np.ndarray]:
    logits_1d = logits.reshape(-1)
    q_values = logits_1d.detach().cpu().numpy().astype(np.float32)
    n_actions = int(q_values.shape[0])
    if n_actions <= 0:
        raise ValueError("No offspring candidates available for epsilon-greedy selection.")

    if float(np.random.random()) < float(epsilon):
        idx = int(np.random.randint(0, n_actions))
    else:
        idx = int(np.argmax(q_values))
    return idx, q_values


def get_pseudo_front_indices(values: np.ndarray, *, atol: float = 1e-6) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float32)
    if values_arr.ndim != 2:
        raise ValueError(f"values must be 2D, got shape={values_arr.shape}.")
    front = pareto_front(values_arr)
    keep: list[int] = []
    for idx in range(int(values_arr.shape[0])):
        row = values_arr[idx]
        matches = np.isclose(front, row[None, :], atol=float(atol), rtol=0.0)
        if bool(np.any(np.all(matches, axis=1))):
            keep.append(int(idx))
    if len(keep) == 0:
        return np.arange(int(values_arr.shape[0]), dtype=np.int64)
    return np.asarray(keep, dtype=np.int64)


def select_offspring_index_pseudo_front_qmax(logits: torch.Tensor, surrogate_y: np.ndarray) -> tuple[int, np.ndarray]:
    logits_1d = logits.reshape(-1)
    q_values = logits_1d.detach().cpu().numpy().astype(np.float32)
    pseudo_idx = get_pseudo_front_indices(surrogate_y)
    if pseudo_idx.size <= 0:
        raise ValueError("Pseudo-front selection requires at least one offspring candidate.")
    local_best = int(np.argmax(q_values[pseudo_idx]))
    return int(pseudo_idx[local_best]), q_values


def select_offspring_index_grouped_pseudo_front_qmax(
    logits: torch.Tensor,
    surrogate_y: np.ndarray,
    group_indices: list[np.ndarray],
) -> tuple[int, np.ndarray]:
    logits_1d = logits.reshape(-1)
    q_values = logits_1d.detach().cpu().numpy().astype(np.float32)
    keep_parts: list[np.ndarray] = []
    for group in group_indices:
        group_idx = np.asarray(group, dtype=np.int64).reshape(-1)
        if group_idx.size <= 0:
            continue
        local_values = np.asarray(surrogate_y[group_idx], dtype=np.float32)
        local_keep = get_pseudo_front_indices(local_values)
        keep_parts.append(group_idx[local_keep])
    if len(keep_parts) == 0:
        raise ValueError("Grouped pseudo-front selection requires at least one offspring candidate.")
    pseudo_idx = np.concatenate(keep_parts, axis=0).astype(np.int64, copy=False)
    local_best = int(np.argmax(q_values[pseudo_idx]))
    return int(pseudo_idx[local_best]), q_values


@dataclass
class StepRecord:
    step: int
    fe: int
    selected_index: int
    selected_x: list[float]
    surrogate_y: list[float]
    true_y: list[float]
    reward: float
    hv: float
    archive_size: int


def run_policy_rollout(
    *,
    args: argparse.Namespace,
    problem,
    nsga2_problem,
    ref_point: np.ndarray,
    true_pareto: np.ndarray | None,
    archive_x_init: np.ndarray,
    archive_y_init: np.ndarray,
    policy_name: str,
    disc: Any | None = None,
    db_saea_agent: Any | None = None,
    infill_criterion: Any | None = None,
    compare_mode: bool = False,
    make_plot: bool = True,
    logger=print,
    reward_scheme_id: int = 1,
    true_pareto_hv: float | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    archive_x = np.asarray(archive_x_init, dtype=np.float32).copy()
    archive_y = np.asarray(archive_y_init, dtype=np.float32).copy()
    n_evo_steps = int(args.max_fe) - int(args.init_fe)
    fe_history = [int(args.init_fe)]
    hv_history = [float(hypervolume(archive_y, ref_point))]
    history: list[StepRecord] = []
    step_rewards: list[float] = []

    prefix = f"[{policy_name}] " if compare_mode else ""
    logger(f"{prefix}iter 0 | front = {int(pareto_front(archive_y).shape[0])} | HV = {hv_history[-1]:.6f}")
    prev_reward = np.array([np.nan], dtype=np.float32)

    for step in range(n_evo_steps):
        offspring_x, offspring_pred, offspring_sigma, offspring_groups = generate_offspring_pool(
            args=args,
            nsga_problem=nsga2_problem,
            archive_x=archive_x,
            archive_y=archive_y,
            step=step,
        )

        if policy_name.lower() == "disc":
            if disc is None:
                raise ValueError("DISC rollout requires a built disc model.")
            progress = float(step) / float(max(n_evo_steps - 1, 1))
            with torch.no_grad():
                out = disc(
                    x_true=torch.from_numpy(archive_x).to(device=args.device, dtype=torch.float32),
                    y_true=torch.from_numpy(archive_y).to(device=args.device, dtype=torch.float32),
                    x_sur=torch.from_numpy(offspring_x).to(device=args.device, dtype=torch.float32),
                    y_sur=torch.from_numpy(offspring_pred).to(device=args.device, dtype=torch.float32),
                    sigma_sur=torch.from_numpy(offspring_sigma).to(device=args.device, dtype=torch.float32),
                    progress=progress,
                    prev_reward=prev_reward,
                    lower_bound=np.full(int(args.dim), float(problem.lower), dtype=np.float32),
                    upper_bound=np.full(int(args.dim), float(problem.upper), dtype=np.float32),
                    decode_type="epsilon_greedy",
                    epsilon=0.05,
                )
                action_idx = int(out["action"].reshape(-1)[0].item())
            chosen_steps = int(Disc.action_to_surrogate_nsga_steps(action_idx))
            local_args = argparse.Namespace(**vars(args))
            local_args.surrogate_nsga_steps = chosen_steps
            offspring_x, offspring_pred, offspring_sigma, offspring_groups = generate_offspring_pool(
                args=local_args,
                nsga_problem=nsga2_problem,
                archive_x=archive_x,
                archive_y=archive_y,
                step=step,
            )
            sigma_scores = np.asarray(offspring_sigma, dtype=np.float32)
            if sigma_scores.ndim == 1:
                sigma_scores = sigma_scores.reshape(-1, 1)
            sigma_scores = sigma_scores.mean(axis=1)
            pseudo_idx = get_pseudo_front_indices(offspring_pred)
            selected_idx = int(pseudo_idx[int(np.argmax(sigma_scores[pseudo_idx]))])
        elif policy_name.lower() == "db_saea":
            if db_saea_agent is None:
                raise ValueError("DB-SAEA rollout requires a built db_saea model.")
            max_regenerate_attempts = 5
            regenerate_attempts = 0
            while True:
                progress = float(step) / float(max(n_evo_steps - 1, 1))
                with torch.no_grad():
                    out = db_saea_agent(
                        x_true=torch.from_numpy(archive_x).to(device=args.device, dtype=torch.float32),
                        y_true=torch.from_numpy(archive_y).to(device=args.device, dtype=torch.float32),
                        x_sur=torch.from_numpy(offspring_x).to(device=args.device, dtype=torch.float32),
                        y_sur=torch.from_numpy(offspring_pred).to(device=args.device, dtype=torch.float32),
                        sigma_sur=torch.from_numpy(offspring_sigma).to(device=args.device, dtype=torch.float32),
                        progress=progress,
                        lower_bound=np.full(int(args.dim), float(problem.lower), dtype=np.float32),
                        upper_bound=np.full(int(args.dim), float(problem.upper), dtype=np.float32),
                        decode_type="epsilon_greedy",
                        epsilon=0.05,
                    )
                strategy_action = int(out["action"].reshape(-1)[0].item())
                if strategy_action == 0 and regenerate_attempts >= max_regenerate_attempts:
                    strategy_action = 1 + int(torch.argmax(out["q_values"].reshape(-1)[1:]).item())
                if strategy_action == 0:
                    regenerate_attempts += 1
                    offspring_x, offspring_pred, offspring_sigma, offspring_groups = generate_offspring_pool(
                        args=args,
                        nsga_problem=nsga2_problem,
                        archive_x=archive_x,
                        archive_y=archive_y,
                        step=step + regenerate_attempts,
                    )
                    continue
                selected_idx, _ = db_saea_agent.select_candidate_from_action(
                    action_idx=int(strategy_action),
                    archive_y=archive_y,
                    candidate_mean=offspring_pred,
                    candidate_std=offspring_sigma,
                    seed=int(args.seed) + step + regenerate_attempts,
                )
                if selected_idx is None:
                    raise RuntimeError(f"DB-SAEA strategy action {strategy_action} did not produce a candidate index.")
                break
        elif infill_criterion is not None:
            selected_idx, _ = infill_criterion.select_index(
                archive_y=archive_y,
                candidate_mean=offspring_pred,
                candidate_std=offspring_sigma,
                seed=int(args.seed) + step,
            )
        else:
            raise ValueError(f"Unsupported policy_name: {policy_name}")

        selected_x = offspring_x[selected_idx : selected_idx + 1]
        selected_pred = offspring_pred[selected_idx]
        selected_true = np.asarray(problem.evaluate(selected_x), dtype=np.float32)
        previous_front = pareto_front(np.asarray(archive_y, dtype=np.float32))
        archive_x = np.vstack([archive_x, selected_x]).astype(np.float32)
        archive_y = np.vstack([archive_y, selected_true]).astype(np.float32)
        surrogate = build_surrogate(args, archive_x, archive_y)
        archive_pred_y = predict_surrogate_mean(surrogate, archive_x)
        step_reward = compute_test_reward(
            reward_scheme_id=int(reward_scheme_id),
            previous_front=previous_front,
            selected_objectives=selected_true,
            ref_point=ref_point,
            reward_lambda=float(args.reward_lambda),
            true_pareto_hv=true_pareto_hv,
            archive_true_y=archive_y,
            archive_pred_y=archive_pred_y,
        )
        hv = hypervolume(archive_y, ref_point)
        fe = int(args.init_fe) + step + 1
        front_size = int(pareto_front(archive_y).shape[0])
        fe_history.append(fe)
        hv_history.append(float(hv))

        record = StepRecord(
            step=step + 1,
            fe=fe,
            selected_index=selected_idx,
            selected_x=selected_x.reshape(-1).astype(float).tolist(),
            surrogate_y=selected_pred.astype(float).tolist(),
            true_y=selected_true.reshape(-1).astype(float).tolist(),
            reward=step_reward,
            hv=float(hv),
            archive_size=int(archive_x.shape[0]),
        )
        history.append(record)
        step_rewards.append(step_reward)
        prev_reward = np.array([float(step_reward)], dtype=np.float32)

        logger(
            f"{prefix}iter {record.step} | front = {front_size} | "
            f"HV = {record.hv:.6f} | reward = {record.reward:.6f}"
        )

    final_front = pareto_front(archive_y)
    plot_path = None
    npy_paths = None
    if make_plot:
        plot_path = plot_results(
            args=args,
            fe_history=fe_history,
            hv_history=hv_history,
            archive_y=archive_y,
            true_pareto=true_pareto,
        )
        npy_paths = save_npy_outputs(
            args=args,
            archive_x=archive_x,
            archive_y=archive_y,
            final_front=final_front,
            fe_history=fe_history,
            hv_history=hv_history,
        )

    summary = {
        "problem": args.problem,
        "dim": int(args.dim),
        "seed": int(args.seed),
        "max_fe": int(args.max_fe),
        "init_fe": int(args.init_fe),
        "evolution_fe": n_evo_steps,
        "surrogate_model": surrogate_model_name(args),
        "candidate_solver": str(getattr(args, "solver", "nsga2")).lower(),
        "pseudo_front_only": bool(getattr(args, "pseudo_front_only", False)),
        "reward_lambda": float(args.reward_lambda),
        "reward_scheme": int(reward_scheme_id),
        "agent_name": args.agent_name,
        "policy_name": policy_name.lower(),
        "agent_pth": args.agent_pth,
        "random_model": bool(args.random_model),
        "reference_point": ref_point.astype(float).tolist(),
        "archive_size": int(archive_x.shape[0]),
        "final_hv": float(hypervolume(archive_y, ref_point)),
        "mean_reward_40_steps": float(np.mean(step_rewards)) if len(step_rewards) > 0 else 0.0,
        "final_front_size": int(final_front.shape[0]),
        "final_front": final_front.astype(float).tolist(),
        "plot_path": plot_path,
        "npy_paths": npy_paths,
        "history": [asdict(item) for item in history],
        "fe_history": fe_history,
        "hv_history": hv_history,
    }
    return summary, archive_y


def load_true_pareto_front(
    problem_name: str,
    dim: int,
    n_obj: int,
    n_points: int = 400,
) -> np.ndarray | None:
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


def plot_results(
    *,
    args: argparse.Namespace,
    fe_history: list[int],
    hv_history: list[float],
    archive_y: np.ndarray,
    true_pareto: np.ndarray | None,
) -> str:
    agent_tag = str(getattr(args, "agent_name", "disc")).lower()
    plot_path = args.plot_path
    if plot_path is None:
        plot_path = str(Path("png") / f"test_{agent_tag}_{args.problem.lower()}_seed{int(args.seed)}.png")
    else:
        plot_path = str(Path(plot_path))

    plot_file = Path(plot_path)
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    archive_front = pareto_front(archive_y)
    n_obj = int(archive_y.shape[1])

    fig = plt.figure(figsize=(12, 5))
    ax_hv = fig.add_subplot(1, 2, 1)
    if n_obj == 3:
        ax_pf = fig.add_subplot(1, 2, 2, projection="3d")
    else:
        ax_pf = fig.add_subplot(1, 2, 2)

    ax_hv.plot(fe_history, hv_history, marker="o", linewidth=1.8, markersize=4)
    ax_hv.set_xlabel("FE")
    ax_hv.set_ylabel("Hypervolume")
    ax_hv.set_title("HV vs FE")
    ax_hv.grid(True, alpha=0.3)

    if n_obj == 2:
        ax_pf.scatter(archive_y[:, 0], archive_y[:, 1], s=18, alpha=0.45, label="Archive")
        ax_pf.scatter(archive_front[:, 0], archive_front[:, 1], s=26, alpha=0.9, label="Archive PF")
        if true_pareto is not None and true_pareto.shape[1] >= 2:
            order = np.argsort(true_pareto[:, 0])
            ax_pf.plot(
                true_pareto[order, 0],
                true_pareto[order, 1],
                linewidth=2.0,
                label="True PF",
            )
        ax_pf.set_xlabel("f1")
        ax_pf.set_ylabel("f2")
        ax_pf.grid(True, alpha=0.3)
    elif n_obj == 3:
        ax_pf.scatter(archive_y[:, 0], archive_y[:, 1], archive_y[:, 2], s=18, alpha=0.30, label="Archive")
        ax_pf.scatter(
            archive_front[:, 0],
            archive_front[:, 1],
            archive_front[:, 2],
            s=28,
            alpha=0.95,
            label="Archive PF",
        )
        if true_pareto is not None and true_pareto.shape[1] >= 3:
            ax_pf.scatter(
                true_pareto[:, 0],
                true_pareto[:, 1],
                true_pareto[:, 2],
                s=8,
                alpha=0.20,
                label="True PF",
            )
        ax_pf.set_xlabel("f1")
        ax_pf.set_ylabel("f2")
        ax_pf.set_zlabel("f3")
    else:
        raise ValueError(f"plot_results currently supports only 2 or 3 objectives, got n_obj={n_obj}.")

    ax_pf.set_title(f"{args.problem} Archive vs True PF")
    ax_pf.legend()

    fig.tight_layout()
    fig.savefig(plot_file, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return str(plot_file.resolve())


def plot_compare_results(
    *,
    args: argparse.Namespace,
    primary_fe_history: list[int],
    primary_hv_history: list[float],
    primary_archive_y: np.ndarray,
    baseline_fe_history: list[int],
    baseline_hv_history: list[float],
    baseline_archive_y: np.ndarray,
    primary_label: str,
    baseline_label: str,
    true_pareto: np.ndarray | None,
) -> str:
    agent_tag = str(getattr(args, "agent_name", "disc")).lower()
    plot_path = args.plot_path
    if plot_path is None:
        plot_path = str(Path("png") / f"test_{agent_tag}_{args.problem.lower()}_seed{int(args.seed)}_compare.png")
    else:
        plot_file = Path(plot_path)
        plot_file = plot_file.with_name(f"{plot_file.stem}_compare{plot_file.suffix}")
        plot_path = str(plot_file)

    plot_file = Path(plot_path)
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    primary_front = pareto_front(primary_archive_y)
    baseline_front = pareto_front(baseline_archive_y)
    n_obj = int(primary_archive_y.shape[1])

    fig = plt.figure(figsize=(13, 5))
    ax_hv = fig.add_subplot(1, 2, 1)
    if n_obj == 3:
        ax_pf = fig.add_subplot(1, 2, 2, projection="3d")
    else:
        ax_pf = fig.add_subplot(1, 2, 2)

    ax_hv.plot(primary_fe_history, primary_hv_history, marker="o", linewidth=1.8, markersize=4, label=primary_label)
    ax_hv.plot(baseline_fe_history, baseline_hv_history, marker="s", linewidth=1.8, markersize=4, label=baseline_label)
    ax_hv.set_xlabel("FE")
    ax_hv.set_ylabel("Hypervolume")
    ax_hv.set_title("HV Comparison")
    ax_hv.grid(True, alpha=0.3)
    ax_hv.legend()

    if n_obj == 2:
        ax_pf.scatter(primary_archive_y[:, 0], primary_archive_y[:, 1], s=14, alpha=0.22, label=f"{primary_label} Archive")
        ax_pf.scatter(baseline_archive_y[:, 0], baseline_archive_y[:, 1], s=14, alpha=0.22, label=f"{baseline_label} Archive")
        ax_pf.scatter(primary_front[:, 0], primary_front[:, 1], s=28, alpha=0.95, marker="o", label=f"{primary_label} PF")
        ax_pf.scatter(baseline_front[:, 0], baseline_front[:, 1], s=28, alpha=0.95, marker="x", label=f"{baseline_label} PF")
        if true_pareto is not None and true_pareto.shape[1] >= 2:
            order = np.argsort(true_pareto[:, 0])
            ax_pf.plot(true_pareto[order, 0], true_pareto[order, 1], linewidth=2.0, label="True PF")
        ax_pf.set_xlabel("f1")
        ax_pf.set_ylabel("f2")
        ax_pf.grid(True, alpha=0.3)
    elif n_obj == 3:
        ax_pf.scatter(primary_archive_y[:, 0], primary_archive_y[:, 1], primary_archive_y[:, 2], s=12, alpha=0.18, label=f"{primary_label} Archive")
        ax_pf.scatter(baseline_archive_y[:, 0], baseline_archive_y[:, 1], baseline_archive_y[:, 2], s=12, alpha=0.18, label=f"{baseline_label} Archive")
        ax_pf.scatter(primary_front[:, 0], primary_front[:, 1], primary_front[:, 2], s=26, alpha=0.95, marker="o", label=f"{primary_label} PF")
        ax_pf.scatter(baseline_front[:, 0], baseline_front[:, 1], baseline_front[:, 2], s=26, alpha=0.95, marker="x", label=f"{baseline_label} PF")
        if true_pareto is not None and true_pareto.shape[1] >= 3:
            ax_pf.scatter(true_pareto[:, 0], true_pareto[:, 1], true_pareto[:, 2], s=8, alpha=0.20, label="True PF")
        ax_pf.set_xlabel("f1")
        ax_pf.set_ylabel("f2")
        ax_pf.set_zlabel("f3")
    else:
        raise ValueError(f"plot_compare_results currently supports only 2 or 3 objectives, got n_obj={n_obj}.")

    ax_pf.set_title(f"{args.problem} Archive Comparison")
    ax_pf.legend()
    fig.tight_layout()
    fig.savefig(plot_file, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return str(plot_file.resolve())


def save_npy_outputs(
    *,
    args: argparse.Namespace,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    final_front: np.ndarray,
    fe_history: list[int],
    hv_history: list[float],
) -> dict[str, str]:
    out_dir = Path("npy")
    out_dir.mkdir(parents=True, exist_ok=True)
    agent_tag = str(getattr(args, "agent_name", "disc")).lower()
    stem = f"test_{agent_tag}_{args.problem.lower()}_seed{int(args.seed)}"

    paths = {
        "archive_x": out_dir / f"{stem}_archive_x.npy",
        "archive_y": out_dir / f"{stem}_archive_y.npy",
        "final_front": out_dir / f"{stem}_final_front.npy",
        "fe_history": out_dir / f"{stem}_fe_history.npy",
        "hv_history": out_dir / f"{stem}_hv_history.npy",
    }

    np.save(paths["archive_x"], np.asarray(archive_x, dtype=np.float32))
    np.save(paths["archive_y"], np.asarray(archive_y, dtype=np.float32))
    np.save(paths["final_front"], np.asarray(final_front, dtype=np.float32))
    np.save(paths["fe_history"], np.asarray(fe_history, dtype=np.int64))
    np.save(paths["hv_history"], np.asarray(hv_history, dtype=np.float64))

    return {key: str(path.resolve()) for key, path in paths.items()}


def main(agent_name: str = "disc") -> None:
    args = parse_args()
    args.agent_name = str(agent_name).lower()
    set_seed(int(args.seed))
    test_log_path = default_test_log_path(args, agent_name=args.agent_name)
    log, log_fp = make_test_logger(test_log_path)

    try:
        log(f"test_log_path = {str(test_log_path.resolve())}")
        problem = make_problem(args.problem, dim=int(args.dim))
        n_evo_steps = int(args.max_fe) - int(args.init_fe)
        reward_scheme_id = resolve_test_reward_scheme(args)

        lhs_started_at = time.perf_counter()
        archive_x = latin_hypercube_sample(
            n_samples=int(args.init_fe),
            dim=int(args.dim),
            lower=problem.lower,
            upper=problem.upper,
            seed=int(args.seed),
        )
        lhs_sample_sec = time.perf_counter() - lhs_started_at
        init_eval_started_at = time.perf_counter()
        archive_y = np.asarray(problem.evaluate(archive_x), dtype=np.float32)
        init_eval_sec = time.perf_counter() - init_eval_started_at
        n_obj = int(archive_y.shape[1])
        ref_point = get_reference_point(args.problem, n_obj=n_obj)
        nsga2_problem = make_nsga2_problem_adapter(problem, n_obj)
        true_pareto_load_started_at = time.perf_counter()
        true_pareto = load_true_pareto_front(args.problem, int(args.dim), n_obj)
        true_pareto_load_sec = time.perf_counter() - true_pareto_load_started_at
        true_pareto_hv_started_at = time.perf_counter()
        true_pareto_hv = None
        if int(reward_scheme_id) == 3:
            true_pareto_hv = get_true_pareto_hv(args.problem, dim=int(args.dim), n_obj=n_obj)
        true_pareto_hv_sec = time.perf_counter() - true_pareto_hv_started_at
        compare_infill_name = resolve_compare_infill_name(args)
        compare_algo_name = None if args.compare_algo is None else str(args.compare_algo).lower()
        compare_infill = None if compare_infill_name is None else build_compare_infill_criterion(compare_infill_name, ref_point=ref_point)
        compare_label = None if compare_infill_name is None else compare_infill_display_name(compare_infill_name)
        if int(reward_scheme_id) == 3 and true_pareto_hv is None:
            raise RuntimeError(
                f"No precomputed true Pareto HV found for {args.problem}-{int(args.dim)}D-{int(n_obj)}obj in ref_points_hv.py."
            )
        agent_load_started_at = time.perf_counter()
        disc, agent_load_breakdown = build_disc(args, map_location=str(args.device), agent_name=args.agent_name)
        agent_load_sec = time.perf_counter() - agent_load_started_at
        if bool(args.debug):
            log(f"reference_point = {ref_point.tolist()} (from ref_points_hv.py)")
            log(f"candidate_solver = {str(getattr(args, 'solver', 'nsga2')).lower()}")
            log(f"nsga_af = {str(args.nsga_af).lower()} | beta = {float(args.beta):.4f}")
            log(f"hybrid_nsga = {int(bool(args.hybrid_nsga))}")
            log(f"hybrid_nsga_gp = {int(bool(getattr(args, 'hybrid_nsga_gp', False)))}")
            log(f"surrogate_nsga_steps = {int(args.surrogate_nsga_steps)}")
            log(f"gp_nsga_steps = {resolve_surrogate_nsga_steps(args, 'gp')}")
            log(f"tabpfn_nsga_steps = {resolve_surrogate_nsga_steps(args, 'tabpfn')}")
            log(f"lhs_sample_sec = {lhs_sample_sec:.3f}")
            log(f"init_eval_sec = {init_eval_sec:.3f}")
            log(f"true_pareto_load_sec = {true_pareto_load_sec:.3f}")
            log(f"true_pareto_hv_sec = {true_pareto_hv_sec:.3f}")
            log(f"pseudo_front_only = {int(bool(args.pseudo_front_only))}")
            log(f"compare_infill = {compare_infill_name if compare_infill_name is not None else '-'}")
            log(f"compare_algo = {compare_algo_name if compare_algo_name is not None else '-'}")
            log(f"agent_pth = {str(args.agent_pth) if args.agent_pth is not None else '-'}")
            log(f"compare_agent_pth = {str(args.compare_agent_pth) if getattr(args, 'compare_agent_pth', None) is not None else '-'}")
            log(f"reward_scheme = rs{int(reward_scheme_id)} | reward_lambda = {float(args.reward_lambda):.4f}")
            log(f"model_to_device_sec = {float(agent_load_breakdown['model_to_device_sec']):.3f}")
            log(f"torch_load_sec = {float(agent_load_breakdown['torch_load_sec']):.3f}")
            log(f"load_state_dict_sec = {float(agent_load_breakdown['load_state_dict_sec']):.3f}")
            log(f"agent_load_sec = {agent_load_sec:.3f}")
        disc_summary, disc_archive_y = run_policy_rollout(
            args=args,
            problem=problem,
            nsga2_problem=nsga2_problem,
            ref_point=ref_point,
            true_pareto=true_pareto,
            archive_x_init=archive_x,
            archive_y_init=archive_y,
            policy_name="disc",
            disc=disc,
            compare_mode=bool(compare_infill_name),
            make_plot=not bool(compare_infill_name),
            logger=log,
            reward_scheme_id=int(reward_scheme_id),
            true_pareto_hv=true_pareto_hv,
        )
        if compare_infill_name is not None:
            infill_summary, infill_archive_y = run_policy_rollout(
                args=args,
                problem=problem,
                nsga2_problem=nsga2_problem,
                ref_point=ref_point,
                true_pareto=true_pareto,
                archive_x_init=archive_x,
                archive_y_init=archive_y,
                policy_name=str(compare_infill_name),
                infill_criterion=compare_infill,
                compare_mode=True,
                make_plot=False,
                logger=log,
                reward_scheme_id=int(reward_scheme_id),
                true_pareto_hv=true_pareto_hv,
            )
            compare_plot_path = plot_compare_results(
                args=args,
                primary_fe_history=list(disc_summary["fe_history"]),
                primary_hv_history=list(disc_summary["hv_history"]),
                primary_archive_y=disc_archive_y,
                baseline_fe_history=list(infill_summary["fe_history"]),
                baseline_hv_history=list(infill_summary["hv_history"]),
                baseline_archive_y=infill_archive_y,
                primary_label="DISC",
                baseline_label=str(compare_label),
                true_pareto=true_pareto,
            )
            summary = {
                "problem": args.problem,
                "dim": int(args.dim),
                "seed": int(args.seed),
                "surrogate_model": surrogate_model_name(args),
                "agent_name": args.agent_name,
                "compare_infill": compare_infill_name,
                "reward_scheme": int(reward_scheme_id),
                "disc": disc_summary,
                "infill": infill_summary,
                "compare_plot_path": compare_plot_path,
                "test_log_path": str(test_log_path.resolve()),
            }
        elif compare_algo_name is not None:
            compare_args = argparse.Namespace(**vars(args))
            compare_args.agent_pth = args.compare_agent_pth
            compare_args.random_model = bool(args.random_model)
            if compare_args.agent_pth is None and not bool(compare_args.random_model):
                raise ValueError("compare_algo requires --compare_agent_pth unless --random_model is used.")
            baseline_agent, _ = build_disc(compare_args, map_location=str(args.device), agent_name=str(compare_algo_name))
            baseline_summary, baseline_archive_y = run_policy_rollout(
                args=args,
                problem=problem,
                nsga2_problem=nsga2_problem,
                ref_point=ref_point,
                true_pareto=true_pareto,
                archive_x_init=archive_x,
                archive_y_init=archive_y,
                policy_name=str(compare_algo_name),
                db_saea_agent=baseline_agent if compare_algo_name == "db_saea" else None,
                compare_mode=True,
                make_plot=False,
                logger=log,
                reward_scheme_id=int(reward_scheme_id),
                true_pareto_hv=true_pareto_hv,
            )
            compare_plot_path = plot_compare_results(
                args=args,
                primary_fe_history=list(disc_summary["fe_history"]),
                primary_hv_history=list(disc_summary["hv_history"]),
                primary_archive_y=disc_archive_y,
                baseline_fe_history=list(baseline_summary["fe_history"]),
                baseline_hv_history=list(baseline_summary["hv_history"]),
                baseline_archive_y=baseline_archive_y,
                primary_label="DISC",
                baseline_label=str(compare_algo_name).upper(),
                true_pareto=true_pareto,
            )
            summary = {
                "problem": args.problem,
                "dim": int(args.dim),
                "seed": int(args.seed),
                "surrogate_model": surrogate_model_name(args),
                "agent_name": args.agent_name,
                "compare_algo": compare_algo_name,
                "reward_scheme": int(reward_scheme_id),
                "disc": disc_summary,
                "baseline": baseline_summary,
                "compare_plot_path": compare_plot_path,
                "test_log_path": str(test_log_path.resolve()),
            }
        else:
            summary = disc_summary
            summary["test_log_path"] = str(test_log_path.resolve())

        if args.output_json:
            out_path = Path(args.output_json)
            out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if compare_infill_name is not None:
            log(
                f"mean reward ({n_evo_steps} steps) | "
                f"DISC = {summary['disc']['mean_reward_40_steps']:.6f} | "
                f"{compare_label} = {summary['infill']['mean_reward_40_steps']:.6f}"
            )
            log(
                json.dumps(
                    {
                        "problem": summary["problem"],
                        "dim": summary["dim"],
                        "seed": summary["seed"],
                        "surrogate_model": summary["surrogate_model"],
                        "agent_name": summary["agent_name"],
                        "compare_infill": compare_infill_name,
                        "disc_final_hv": summary["disc"]["final_hv"],
                        "infill_final_hv": summary["infill"]["final_hv"],
                        "compare_plot_path": summary["compare_plot_path"],
                        "test_log_path": summary["test_log_path"],
                    },
                    indent=2,
                )
            )
        elif compare_algo_name is not None:
            log(
                f"mean reward ({n_evo_steps} steps) | "
                f"DISC = {summary['disc']['mean_reward_40_steps']:.6f} | "
                f"{str(compare_algo_name).upper()} = {summary['baseline']['mean_reward_40_steps']:.6f}"
            )
            log(
                json.dumps(
                    {
                        "problem": summary["problem"],
                        "dim": summary["dim"],
                        "seed": summary["seed"],
                        "surrogate_model": summary["surrogate_model"],
                        "agent_name": summary["agent_name"],
                        "compare_algo": compare_algo_name,
                        "disc_final_hv": summary["disc"]["final_hv"],
                        "baseline_final_hv": summary["baseline"]["final_hv"],
                        "compare_plot_path": summary["compare_plot_path"],
                        "test_log_path": summary["test_log_path"],
                    },
                    indent=2,
                )
            )
        else:
            log(f"mean reward ({n_evo_steps} steps) = {summary['mean_reward_40_steps']:.6f}")
            log(json.dumps({k: v for k, v in summary.items() if k not in {"history", "final_front", "fe_history", "hv_history"}}, indent=2))
    finally:
        log_fp.close()


if __name__ == "__main__":
    main()
