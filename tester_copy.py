from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from inspect import signature
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from agents.bass import Bass as Disc
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
    TabPFNMinMaxSurrogate,
    _apply_balanced_softmax_probs,
    _get_tabpfn_ensemble_members,
    estimate_uncertainty,
    fit_kan_surrogates,
    fit_tabpfn_surrogate,
    KANSurrogateModel,
    surrogate_model_name,
    tabpfn_probs_to_mean_std,
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


def default_test_log_path(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    compare_name = resolve_compare_infill_name(args)
    compare_tag = "" if compare_name is None else f"_compare_{compare_name}"
    stem = (
        f"test_disc_{str(args.problem).lower()}_{str(args.surrogate_model).lower()}_"
        f"seed{int(args.seed)}{compare_tag}_{timestamp}.txt"
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
            raise ValueError("reward_scheme_3 requires true_pareto_hv in tester_copy.")
        return float(
            reward_scheme_3(
                previous_front=previous_front,
                selected_objectives=selected_objectives,
                ref_point=ref_point,
                true_pareto_hv=float(true_pareto_hv),
            )
        )
    raise ValueError(f"Unsupported reward_scheme_id for tester_copy: {reward_scheme_id}")


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
    parser.add_argument("--increase_steps", type=int, default=3)
    parser.add_argument("--offspring_size", type=int, default=80)
    parser.add_argument("--mutation_sigma", type=float, default=0.12)
    parser.add_argument("--logit_scale", type=float, default=5.0)
    parser.add_argument("--agent_pth", type=str, default=None)
    parser.add_argument("--random_model", action="store_true")
    parser.add_argument("--surrogate_model", type=str, default="gp", choices=["gp", "kan", "tabpfn"])
    parser.add_argument("--reward_lambda", type=float, default=10.0)
    parser.add_argument("--ensemble_model", type=int, default=8)
    parser.add_argument("--kan_steps", type=int, default=25)
    parser.add_argument("--kan_hidden_width", type=int, default=10)
    parser.add_argument("--kan_grid", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--softmax", action="store_true")
    parser.add_argument("--nsga_af", type=str, default="mean", choices=["mean", "lcb", "ei"])
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--compare_infill", type=str, default=None)
    parser.add_argument("--solver", type=str, default="nsga2", choices=["nsga2", "nsga3", "moead", "usemo"])
    parser.add_argument("--pseudo_front_only", action="store_true")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--plot_path", type=str, default=None)
    args = parser.parse_args()

    if int(args.max_fe) <= int(args.init_fe):
        raise ValueError(f"max_fe must be greater than init_fe, got {args.max_fe} and {args.init_fe}.")
    if str(args.solver).lower() == "usemo" and str(args.nsga_af).lower() not in {"lcb", "ei"}:
        args.nsga_af = "ei"
    return args


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


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


def build_surrogate(
    args: argparse.Namespace,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    existing_surrogate: Any | None = None,
):
    name = surrogate_model_name(args)
    if name == "gp":
        return fit_gp_surrogates(
            archive_x=archive_x,
            archive_y=archive_y,
            seed=int(args.seed),
            nu=float(getattr(args, "gp_nu", 5.0)),
            variant="gp",
        )

    if name == "gp2":
        return fit_gp_surrogates(
            archive_x=archive_x,
            archive_y=archive_y,
            seed=int(args.seed),
            variant="gp2",
        )

    if name == "gp3":
        return fit_gp_surrogates(
            archive_x=archive_x,
            archive_y=archive_y,
            variant="gp3",
            xl=np.min(np.asarray(archive_x, dtype=np.float32), axis=0),
            xu=np.max(np.asarray(archive_x, dtype=np.float32), axis=0),
            seed=int(args.seed),
            nu=float(getattr(args, "gp3_nu", 2.5)),
        )

    if name == "tabpfn":
        return fit_tabpfn_surrogate(
            archive_x=archive_x,
            archive_y=archive_y,
            device=str(args.device),
            n_estimators=int(args.ensemble_model),
            existing_surrogate=existing_surrogate,
        )

    if name == "kan":
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

    raise ValueError(f"Unsupported surrogate_model: {name}")


class _TabPFNSoftmaxPredictor:
    def __init__(self, surrogate: TabPFNMinMaxSurrogate):
        self.surrogate = surrogate

    def predict_mean(self, x: np.ndarray) -> np.ndarray:
        mean, _ = _predict_tabpfn_minmax_mean_std_softmax(self.surrogate, x)
        return np.asarray(mean, dtype=np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_mean(x)

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        _, std = _predict_tabpfn_minmax_mean_std_softmax(self.surrogate, x)
        return np.asarray(std, dtype=np.float32)


class _LCBObjectiveWrapper:
    def __init__(self, base_surrogate: Any, beta: float):
        self.base_surrogate = base_surrogate
        self.beta = float(beta)

    def predict_mean(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        mean = np.asarray(self.base_surrogate.predict_mean(x_arr), dtype=np.float32)
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


def surrogate_or_models_for_nsga2(surrogate: Any) -> tuple[Any | None, list[Any] | None]:
    if isinstance(surrogate, TabPFNMinMaxSurrogate):
        return _TabPFNSoftmaxPredictor(surrogate), None
    models = getattr(surrogate, "models", None)
    if isinstance(models, list) and len(models) > 0:
        return None, models
    return surrogate, None


def prepare_nsga_surrogate(args: argparse.Namespace, surrogate: Any) -> tuple[Any | None, list[Any] | None]:
    if str(getattr(args, "nsga_af", "mean")).lower() == "lcb":
        base_surrogate = _TabPFNSoftmaxPredictor(surrogate) if isinstance(surrogate, TabPFNMinMaxSurrogate) else surrogate
        return _LCBObjectiveWrapper(base_surrogate, beta=float(getattr(args, "beta", 1.0))), None
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
    surrogate_nsga_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    solver_name = str(getattr(args, "solver", "nsga2")).lower()
    if solver_name == "nsga2":
        return run_surrogate_nsga2(
            gps=nsga2_models,
            surrogate=nsga2_surrogate,
            problem=nsga_problem,
            archive_x=archive_x,
            pop_size=int(args.offspring_size),
            surrogate_nsga_steps=int(surrogate_nsga_steps),
            seed=int(args.seed) + int(step),
        )
    if solver_name == "nsga3":
        return run_surrogate_nsga3(
            gps=nsga2_models,
            surrogate=nsga2_surrogate,
            problem=nsga_problem,
            archive_x=archive_x,
            pop_size=int(args.offspring_size),
            surrogate_nsga_steps=int(surrogate_nsga_steps),
            seed=int(args.seed) + int(step),
        )
    if solver_name == "moead":
        return run_surrogate_moead(
            gps=nsga2_models,
            surrogate=nsga2_surrogate,
            problem=nsga_problem,
            archive_x=archive_x,
            pop_size=int(args.offspring_size),
            surrogate_nsga_steps=int(surrogate_nsga_steps),
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


def predict_surrogate_mean(surrogate: Any, x: np.ndarray) -> np.ndarray:
    if isinstance(surrogate, TabPFNMinMaxSurrogate):
        mean, _ = _predict_tabpfn_minmax_mean_std_softmax(surrogate, x)
        return mean
    return np.asarray(surrogate.predict_mean(np.asarray(x, dtype=np.float32)), dtype=np.float32)


def predict_surrogate_std(
    surrogate: Any,
    x: np.ndarray,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    if isinstance(surrogate, TabPFNMinMaxSurrogate):
        _, std = _predict_tabpfn_minmax_mean_std_softmax(surrogate, x_arr)
        return std
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
) -> Disc:
    disc = Disc(
        hidden_dim=int(args.hidden_dim),
        n_heads=int(args.n_heads),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
        logit_scale=float(args.logit_scale),
    ).to(map_location)
    disc.eval()

    if args.agent_pth and not bool(args.random_model):
        state = torch.load(args.agent_pth, map_location=map_location)
        state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        disc.load_state_dict(state_dict, strict=True)

    return disc


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


def sample_offspring_index_softmax(logits: torch.Tensor) -> tuple[int, np.ndarray]:
    probs = torch.softmax(logits.reshape(-1), dim=-1)
    idx = int(torch.distributions.Categorical(probs=probs).sample().item())
    return idx, probs.detach().cpu().numpy().astype(np.float32).reshape(-1)


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


def _predict_tabpfn_minmax_mean_std_softmax(
    surrogate: TabPFNMinMaxSurrogate,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if surrogate._model is None:
        raise RuntimeError("TabPFNMinMaxSurrogate is not fit yet.")

    x_arr = np.asarray(x, dtype=np.float32)
    x_norm = surrogate._norm_x(x_arr)
    means = []
    stds = []

    for objective_idx, objective in enumerate(surrogate._model.objectives):
        if objective._fit_x is None or objective._fit_y_bins is None:
            raise RuntimeError("TabPFN objective is not fit yet.")

        classifier = objective.model
        x_train = np.asarray(objective._fit_x, dtype=np.float32)
        if x_train.shape[1] != x_norm.shape[1]:
            raise ValueError(
                f"TabPFN custom ensemble expects train/query to share feature width, got {x_train.shape[1]} and {x_norm.shape[1]}."
            )

        ensemble_members = list(_get_tabpfn_ensemble_members(classifier))
        n_estimators = int(len(ensemble_members))
        if n_estimators <= 0:
            raise RuntimeError("TabPFN custom ensemble did not expose any fitted ensemble members.")

        grouped_estimators: dict[int, list[tuple[int, Any]]] = {}
        for est_idx, member in enumerate(ensemble_members):
            model_index = int(getattr(member.config, "_model_index", getattr(objective, "_fit_model_index", 0)))
            grouped_estimators.setdefault(model_index, []).append((est_idx, member))

        probs_full_estimators: list[np.ndarray | None] = [None] * n_estimators
        device = classifier.devices_[0]
        dtype = classifier.forced_inference_dtype_ if classifier.forced_inference_dtype_ is not None else torch.float32

        for model_index, grouped_members in grouped_estimators.items():
            if hasattr(classifier, "models_"):
                core_model = classifier.models_[model_index]
            elif hasattr(classifier, "model_"):
                core_model = classifier.model_
            else:
                raise RuntimeError(f"TabPFN classifier does not expose model_ / models_: {type(classifier).__name__}.")
            core_model = core_model.to(device)

            transformed_queries = [
                np.asarray(member.transform_X_test(x_norm), dtype=np.float32) for _, member in grouped_members
            ]
            local_batch_size = int(len(grouped_members))
            feature_dim = int(
                max(
                    max(int(np.asarray(member.X_train).shape[1]), int(query.shape[1]))
                    for (_, member), query in zip(grouped_members, transformed_queries)
                )
            )
            train_rows = int(max(int(np.asarray(member.X_train).shape[0]) for _, member in grouped_members))
            query_rows = int(max(int(query.shape[0]) for query in transformed_queries))
            kwargs = {}
            if "task_type" in signature(core_model.forward).parameters:
                kwargs["task_type"] = "multiclass"

            if any(member.gpu_preprocessor is not None for _, member in grouped_members):
                from tabpfn.inference import _maybe_run_gpu_preprocessing

                for (est_idx, member), x_query_member in zip(grouped_members, transformed_queries):
                    x_train_member = np.asarray(member.X_train, dtype=np.float32)
                    y_train_member = np.asarray(member.y_train, dtype=np.float32).reshape(-1)
                    x_full = torch.zeros(
                        (x_train_member.shape[0] + x_query_member.shape[0], 1, feature_dim),
                        dtype=dtype,
                        device=device,
                    )
                    y_full = torch.zeros((x_train_member.shape[0], 1), dtype=dtype, device=device)
                    x_full[: x_train_member.shape[0], 0, : x_train_member.shape[1]] = torch.as_tensor(
                        x_train_member,
                        dtype=dtype,
                        device=device,
                    )
                    if x_query_member.shape[0] > 0:
                        x_full[x_train_member.shape[0] :, 0, : x_query_member.shape[1]] = torch.as_tensor(
                            x_query_member,
                            dtype=dtype,
                            device=device,
                        )
                    y_full[:, 0] = torch.as_tensor(y_train_member, dtype=dtype, device=device)
                    x_full = _maybe_run_gpu_preprocessing(
                        x_full,
                        member.gpu_preprocessor,
                        num_train_rows=int(x_train_member.shape[0]),
                    )
                    with torch.autocast(device_type="cuda", enabled=str(device).startswith("cuda")):
                        with torch.inference_mode():
                            output = core_model(
                                x_full,
                                y_full,
                                only_return_standard_out=True,
                                **kwargs,
                            )
                    logits = output[: x_query_member.shape[0], 0, :]
                    class_permutation = getattr(member.config, "class_permutation", None)
                    if class_permutation is None:
                        logits = logits[:, : classifier.n_classes_]
                    else:
                        class_permutation = np.asarray(class_permutation, dtype=np.int64)
                        if len(class_permutation) != classifier.n_classes_:
                            use_perm = np.arange(classifier.n_classes_, dtype=np.int64)
                            use_perm[: len(class_permutation)] = class_permutation
                        else:
                            use_perm = class_permutation
                        logits = logits[:, use_perm]

                    probs_partial = _apply_balanced_softmax_probs(
                        logits,
                        y_train_member,
                        n_classes_full=int(objective.bins.k),
                        active_classes=objective._fit_classes,
                    )
                    if objective._fit_classes is None or probs_partial.shape[1] == objective.bins.k:
                        probs_full = np.asarray(probs_partial, dtype=np.float32)
                    else:
                        probs_full = np.zeros((probs_partial.shape[0], objective.bins.k), dtype=np.float32)
                        for col, cls_id in enumerate(objective._fit_classes.tolist()):
                            if 0 <= int(cls_id) < objective.bins.k and col < probs_partial.shape[1]:
                                probs_full[:, int(cls_id)] = probs_partial[:, col]
                    probs_full_estimators[est_idx] = probs_full.astype(np.float32)
            else:
                x_full = torch.zeros(
                    (train_rows + query_rows, local_batch_size, feature_dim),
                    dtype=dtype,
                    device=device,
                )
                y_full = torch.zeros((train_rows, local_batch_size), dtype=dtype, device=device)
                query_lengths: list[int] = []

                for local_idx, ((_, member), x_query_member) in enumerate(zip(grouped_members, transformed_queries)):
                    x_train_member = np.asarray(member.X_train, dtype=np.float32)
                    y_train_member = np.asarray(member.y_train, dtype=np.float32).reshape(-1)
                    x_full[: x_train_member.shape[0], local_idx, : x_train_member.shape[1]] = torch.as_tensor(
                        x_train_member,
                        dtype=dtype,
                        device=device,
                    )
                    if x_query_member.shape[0] > 0:
                        x_full[
                            x_train_member.shape[0] : x_train_member.shape[0] + x_query_member.shape[0],
                            local_idx,
                            : x_query_member.shape[1],
                        ] = torch.as_tensor(
                            x_query_member,
                            dtype=dtype,
                            device=device,
                        )
                    y_full[: y_train_member.shape[0], local_idx] = torch.as_tensor(
                        y_train_member,
                        dtype=dtype,
                        device=device,
                    )
                    query_lengths.append(int(x_query_member.shape[0]))

                with torch.autocast(device_type="cuda", enabled=str(device).startswith("cuda")):
                    with torch.inference_mode():
                        output = core_model(
                            x_full,
                            y_full,
                            only_return_standard_out=True,
                            **kwargs,
                        )

                for local_idx, ((est_idx, member), x_query_member) in enumerate(zip(grouped_members, transformed_queries)):
                    logits = output[: query_lengths[local_idx], local_idx, :]
                    class_permutation = getattr(member.config, "class_permutation", None)
                    if class_permutation is None:
                        logits = logits[:, : classifier.n_classes_]
                    else:
                        class_permutation = np.asarray(class_permutation, dtype=np.int64)
                        if len(class_permutation) != classifier.n_classes_:
                            use_perm = np.arange(classifier.n_classes_, dtype=np.int64)
                            use_perm[: len(class_permutation)] = class_permutation
                        else:
                            use_perm = class_permutation
                        logits = logits[:, use_perm]

                    probs_partial = _apply_balanced_softmax_probs(
                        logits,
                        np.asarray(member.y_train, dtype=np.float32),
                        n_classes_full=int(objective.bins.k),
                        active_classes=objective._fit_classes,
                    )
                    if objective._fit_classes is None or probs_partial.shape[1] == objective.bins.k:
                        probs_full = np.asarray(probs_partial, dtype=np.float32)
                    else:
                        probs_full = np.zeros((probs_partial.shape[0], objective.bins.k), dtype=np.float32)
                        for col, cls_id in enumerate(objective._fit_classes.tolist()):
                            if 0 <= int(cls_id) < objective.bins.k and col < probs_partial.shape[1]:
                                probs_full[:, int(cls_id)] = probs_partial[:, col]
                    probs_full_estimators[est_idx] = probs_full.astype(np.float32)

        if any(probs is None for probs in probs_full_estimators):
            missing = [idx for idx, probs in enumerate(probs_full_estimators) if probs is None]
            raise RuntimeError(f"Missing custom TabPFN ensemble probabilities for estimator indices {missing}.")

        probs_full = np.mean(
            np.stack([np.asarray(probs, dtype=np.float32) for probs in probs_full_estimators if probs is not None], axis=0),
            axis=0,
        ).astype(np.float32)

        mean_norm, std_norm = tabpfn_probs_to_mean_std(probs_full, objective.bins, normalize=True)
        y_min = float(surrogate._y_min[objective_idx])
        y_rng = float(surrogate._y_rng[objective_idx])
        means.append((y_min + mean_norm * y_rng).astype(np.float32))
        stds.append((std_norm * y_rng).astype(np.float32))

    return np.stack(means, axis=1), np.stack(stds, axis=1)


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
    run_tag: str | None = None,
) -> str:
    plot_path = args.plot_path
    if plot_path is None:
        stem = f"test_disc_{args.problem.lower()}_seed{int(args.seed)}"
        if run_tag:
            stem = f"{stem}_{run_tag}"
        plot_path = str(Path("png") / f"{stem}.png")
    else:
        plot_file = Path(plot_path)
        if run_tag:
            plot_file = plot_file.with_name(f"{plot_file.stem}_{run_tag}{plot_file.suffix}")
        plot_path = str(plot_file)

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
    disc_fe_history: list[int],
    disc_hv_history: list[float],
    disc_archive_y: np.ndarray,
    infill_fe_history: list[int],
    infill_hv_history: list[float],
    infill_archive_y: np.ndarray,
    infill_label: str,
    true_pareto: np.ndarray | None,
) -> str:
    plot_path = args.plot_path
    if plot_path is None:
        plot_path = str(Path("png") / f"test_disc_{args.problem.lower()}_seed{int(args.seed)}_compare.png")
    else:
        plot_file = Path(plot_path)
        plot_file = plot_file.with_name(f"{plot_file.stem}_compare{plot_file.suffix}")
        plot_path = str(plot_file)

    plot_file = Path(plot_path)
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    disc_front = pareto_front(disc_archive_y)
    infill_front = pareto_front(infill_archive_y)
    n_obj = int(disc_archive_y.shape[1])

    fig = plt.figure(figsize=(13, 5))
    ax_hv = fig.add_subplot(1, 2, 1)
    if n_obj == 3:
        ax_pf = fig.add_subplot(1, 2, 2, projection="3d")
    else:
        ax_pf = fig.add_subplot(1, 2, 2)

    ax_hv.plot(disc_fe_history, disc_hv_history, marker="o", linewidth=1.8, markersize=4, label="DISC")
    ax_hv.plot(infill_fe_history, infill_hv_history, marker="s", linewidth=1.8, markersize=4, label=infill_label)
    ax_hv.set_xlabel("FE")
    ax_hv.set_ylabel("Hypervolume")
    ax_hv.set_title("HV Comparison")
    ax_hv.grid(True, alpha=0.3)
    ax_hv.legend()

    if n_obj == 2:
        ax_pf.scatter(disc_archive_y[:, 0], disc_archive_y[:, 1], s=14, alpha=0.22, label="DISC Archive")
        ax_pf.scatter(infill_archive_y[:, 0], infill_archive_y[:, 1], s=14, alpha=0.22, label=f"{infill_label} Archive")
        ax_pf.scatter(disc_front[:, 0], disc_front[:, 1], s=28, alpha=0.95, marker="o", label="DISC PF")
        ax_pf.scatter(infill_front[:, 0], infill_front[:, 1], s=28, alpha=0.95, marker="x", label=f"{infill_label} PF")
        if true_pareto is not None and true_pareto.shape[1] >= 2:
            order = np.argsort(true_pareto[:, 0])
            ax_pf.plot(true_pareto[order, 0], true_pareto[order, 1], linewidth=2.0, label="True PF")
        ax_pf.set_xlabel("f1")
        ax_pf.set_ylabel("f2")
        ax_pf.grid(True, alpha=0.3)
    elif n_obj == 3:
        ax_pf.scatter(disc_archive_y[:, 0], disc_archive_y[:, 1], disc_archive_y[:, 2], s=12, alpha=0.18, label="DISC Archive")
        ax_pf.scatter(infill_archive_y[:, 0], infill_archive_y[:, 1], infill_archive_y[:, 2], s=12, alpha=0.18, label=f"{infill_label} Archive")
        ax_pf.scatter(disc_front[:, 0], disc_front[:, 1], disc_front[:, 2], s=26, alpha=0.95, marker="o", label="DISC PF")
        ax_pf.scatter(infill_front[:, 0], infill_front[:, 1], infill_front[:, 2], s=26, alpha=0.95, marker="x", label=f"{infill_label} PF")
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
    run_tag: str | None = None,
) -> dict[str, str]:
    out_dir = Path("npy")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"test_disc_{args.problem.lower()}_seed{int(args.seed)}"
    if run_tag:
        stem = f"{stem}_{run_tag}"

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
    disc: Disc | None = None,
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
    selection_times_sec: list[float] = []

    prefix = f"[{policy_name}] " if compare_mode else ""
    logger(f"{prefix}iter 0 | front = {int(pareto_front(archive_y).shape[0])} | HV = {hv_history[-1]:.6f}")

    surrogate = build_surrogate(args, archive_x, archive_y)
    surrogate_needs_refit = False
    for step in range(n_evo_steps):
        current_surrogate_nsga_steps = 10 if int(step) < 10 else 100
        if surrogate_needs_refit:
            reuse_surrogate = surrogate if isinstance(surrogate, TabPFNMinMaxSurrogate) else None
            surrogate = build_surrogate(args, archive_x, archive_y, existing_surrogate=reuse_surrogate)
            surrogate_needs_refit = False

        if str(getattr(args, "solver", "nsga2")).lower() == "usemo":
            offspring_x, offspring_pred, offspring_sigma = run_surrogate_usemo(
                problem=nsga2_problem,
                archive_x=archive_x,
                archive_y=archive_y,
                pop_size=int(args.offspring_size),
                surrogate_nsga_steps={**vars(args), "surrogate_nsga_steps": int(current_surrogate_nsga_steps)},
                seed=int(args.seed) + int(step),
                acquisition=str(getattr(args, "nsga_af", "ei")).lower(),
                beta=float(getattr(args, "beta", 1.0)),
                surrogate_model=args,
                device=args,
            )
            offspring_x = np.asarray(offspring_x, dtype=np.float32)
            offspring_pred = np.asarray(offspring_pred, dtype=np.float32)
            offspring_sigma = np.asarray(offspring_sigma, dtype=np.float32)
        else:
            nsga2_surrogate, nsga2_models = prepare_nsga_surrogate(args, surrogate)
            offspring_x, offspring_pred = run_surrogate_optimizer(
                args=args,
                nsga_problem=nsga2_problem,
                archive_x=archive_x,
                archive_y=archive_y,
                nsga2_surrogate=nsga2_surrogate,
                nsga2_models=nsga2_models,
                step=step,
                surrogate_nsga_steps=int(current_surrogate_nsga_steps),
            )
            offspring_x = np.asarray(offspring_x, dtype=np.float32)
            offspring_pred = np.asarray(offspring_pred, dtype=np.float32)
            offspring_sigma = build_offspring_sigma(
                archive_x=archive_x,
                archive_y=archive_y,
                offspring_x=offspring_x,
                surrogate=surrogate,
            )

        if policy_name.lower() == "disc":
            if disc is None:
                raise ValueError("DISC rollout requires a built disc model.")
            progress = float(step) / float(max(n_evo_steps - 1, 1))
            selection_started_at = time.perf_counter()
            with torch.no_grad():
                out = disc(
                    x_true=torch.from_numpy(archive_x).to(device=args.device, dtype=torch.float32),
                    y_true=torch.from_numpy(archive_y).to(device=args.device, dtype=torch.float32),
                    x_sur=torch.from_numpy(offspring_x).to(device=args.device, dtype=torch.float32),
                    y_sur=torch.from_numpy(offspring_pred).to(device=args.device, dtype=torch.float32),
                    sigma_sur=torch.from_numpy(offspring_sigma).to(device=args.device, dtype=torch.float32),
                    progress=progress,
                    lower_bound=np.full(int(args.dim), float(problem.lower), dtype=np.float32),
                    upper_bound=np.full(int(args.dim), float(problem.upper), dtype=np.float32),
                    decode_type="greedy" if bool(args.softmax) else "epsilon_greedy",
                    epsilon=0.05,
                )
                logits = out["logits"].reshape(-1)
            if bool(getattr(args, "pseudo_front_only", False)):
                selected_idx, _ = select_offspring_index_pseudo_front_qmax(logits, offspring_pred)
            elif bool(args.softmax):
                selected_idx, _ = sample_offspring_index_softmax(logits)
            else:
                selected_idx, _ = select_offspring_index_epsilon_greedy(logits, epsilon=0.05)
            selection_sec = time.perf_counter() - selection_started_at
            selection_label = "disc_forward_sec"
        elif infill_criterion is not None:
            selection_started_at = time.perf_counter()
            selected_idx, _ = infill_criterion.select_index(
                archive_y=archive_y,
                candidate_mean=offspring_pred,
                candidate_std=offspring_sigma,
                seed=int(args.seed) + step,
            )
            selection_sec = time.perf_counter() - selection_started_at
            selection_label = f"{policy_name.lower().replace('-', '_')}_select_sec"
        else:
            raise ValueError(f"Unsupported policy_name: {policy_name}")

        selection_times_sec.append(float(selection_sec))
        selected_x = offspring_x[selected_idx : selected_idx + 1]
        selected_pred = offspring_pred[selected_idx]
        selected_true = np.asarray(problem.evaluate(selected_x), dtype=np.float32)
        previous_front = pareto_front(np.asarray(archive_y, dtype=np.float32))
        step_reward = compute_test_reward(
            reward_scheme_id=int(reward_scheme_id),
            previous_front=previous_front,
            selected_objectives=selected_true,
            ref_point=ref_point,
            reward_lambda=float(args.reward_lambda),
            true_pareto_hv=true_pareto_hv,
        )

        archive_x = np.vstack([archive_x, selected_x]).astype(np.float32)
        archive_y = np.vstack([archive_y, selected_true]).astype(np.float32)
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

        logger(
            f"{prefix}iter {record.step} | front = {front_size} | "
            f"HV = {record.hv:.6f} | reward = {record.reward:.6f} | "
            f"{selection_label} = {selection_sec:.3f} | "
            f"surrogate_nsga_steps = {int(current_surrogate_nsga_steps)}"
        )
        surrogate_needs_refit = True

    final_front = pareto_front(archive_y)
    run_tag = None if not compare_mode else policy_name.lower()
    plot_path = None
    if make_plot:
        plot_path = plot_results(
            args=args,
            fe_history=fe_history,
            hv_history=hv_history,
            archive_y=archive_y,
            true_pareto=true_pareto,
            run_tag=run_tag,
        )
    npy_paths = save_npy_outputs(
        args=args,
        archive_x=archive_x,
        archive_y=archive_y,
        final_front=final_front,
        fe_history=fe_history,
        hv_history=hv_history,
        run_tag=run_tag,
    )

    summary = {
        "policy": policy_name,
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
        "ensemble_model": int(args.ensemble_model),
        "agent_pth": args.agent_pth,
        "random_model": bool(args.random_model),
        "reference_point": ref_point.astype(float).tolist(),
        "archive_size": int(archive_x.shape[0]),
        "final_hv": float(hypervolume(archive_y, ref_point)),
        "mean_reward_40_steps": float(np.mean(step_rewards)) if len(step_rewards) > 0 else 0.0,
        "mean_select_sec": float(np.mean(selection_times_sec)) if len(selection_times_sec) > 0 else 0.0,
        "final_front_size": int(final_front.shape[0]),
        "final_front": final_front.astype(float).tolist(),
        "plot_path": plot_path,
        "npy_paths": npy_paths,
        "history": [asdict(item) for item in history],
    }
    if policy_name.lower() == "disc":
        summary["mean_disc_forward_sec"] = summary["mean_select_sec"]
    if infill_criterion is not None:
        summary["mean_infill_select_sec"] = summary["mean_select_sec"]
    return summary, archive_y


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    test_log_path = default_test_log_path(args)
    log, log_fp = make_test_logger(test_log_path)

    try:
        log(f"test_log_path = {str(test_log_path.resolve())}")
        problem = make_problem(args.problem, dim=int(args.dim))
        reward_scheme_id = resolve_test_reward_scheme(args)
        lhs_started_at = time.perf_counter()
        archive_x_init = latin_hypercube_sample(
            n_samples=int(args.init_fe),
            dim=int(args.dim),
            lower=problem.lower,
            upper=problem.upper,
            seed=int(args.seed),
        )
        lhs_sample_sec = time.perf_counter() - lhs_started_at
        archive_y_init = np.asarray(problem.evaluate(archive_x_init), dtype=np.float32)
        n_obj = int(archive_y_init.shape[1])
        ref_point = get_reference_point(args.problem, n_obj=n_obj)
        nsga2_problem = make_nsga2_problem_adapter(problem, n_obj)
        true_pareto = load_true_pareto_front(args.problem, int(args.dim), n_obj)
        true_pareto_hv = None
        if int(reward_scheme_id) == 3:
            true_pareto_hv = get_true_pareto_hv(args.problem, dim=int(args.dim), n_obj=n_obj)
        log(f"reference_point = {ref_point.tolist()} (from ref_points_hv.py)")
        log(f"candidate_solver = {str(getattr(args, 'solver', 'nsga2')).lower()}")
        log(f"nsga_af = {str(args.nsga_af).lower()} | beta = {float(args.beta):.4f}")
        log(f"lhs_sample_sec = {lhs_sample_sec:.3f}")
        log(f"pseudo_front_only = {int(bool(args.pseudo_front_only))}")
        compare_infill_name = resolve_compare_infill_name(args)
        compare_infill = None if compare_infill_name is None else build_compare_infill_criterion(compare_infill_name, ref_point=ref_point)
        compare_label = None if compare_infill_name is None else compare_infill_display_name(compare_infill_name)
        log(f"compare_infill = {compare_infill_name if compare_infill_name is not None else '-'}")
        log(f"reward_scheme = rs{int(reward_scheme_id)} | reward_lambda = {float(args.reward_lambda):.4f}")
        if int(reward_scheme_id) == 3 and true_pareto_hv is None:
            raise RuntimeError(
                f"No precomputed true Pareto HV found for {args.problem}-{int(args.dim)}D-{int(n_obj)}obj in ref_points_hv.py."
            )
        agent_load_started_at = time.perf_counter()
        disc = build_disc(args, map_location=str(args.device))
        agent_load_sec = time.perf_counter() - agent_load_started_at
        log(f"agent_load_sec = {agent_load_sec:.3f}")
        disc_summary, disc_archive_y = run_policy_rollout(
            args=args,
            problem=problem,
            nsga2_problem=nsga2_problem,
            ref_point=ref_point,
            true_pareto=true_pareto,
            archive_x_init=archive_x_init,
            archive_y_init=archive_y_init,
            policy_name="DISC",
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
                archive_x_init=archive_x_init,
                archive_y_init=archive_y_init,
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
                disc_fe_history=disc_summary["npy_paths"] and np.load(disc_summary["npy_paths"]["fe_history"]).tolist(),
                disc_hv_history=np.load(disc_summary["npy_paths"]["hv_history"]).tolist(),
                disc_archive_y=disc_archive_y,
                infill_fe_history=np.load(infill_summary["npy_paths"]["fe_history"]).tolist(),
                infill_hv_history=np.load(infill_summary["npy_paths"]["hv_history"]).tolist(),
                infill_archive_y=infill_archive_y,
                infill_label=str(compare_label),
                true_pareto=true_pareto,
            )
            disc_summary["plot_path"] = compare_plot_path
            infill_summary["plot_path"] = compare_plot_path
            summary = {
                "compare_infill": compare_infill_name,
                "reward_scheme": int(reward_scheme_id),
                "compare_plot_path": compare_plot_path,
                "disc": disc_summary,
                "infill": infill_summary,
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
                "compare summary | "
                f"disc_final_hv = {disc_summary['final_hv']:.6f} | "
                f"infill_final_hv = {infill_summary['final_hv']:.6f} | "
                f"disc_mean_reward = {disc_summary['mean_reward_40_steps']:.6f} | "
                f"infill_mean_reward = {infill_summary['mean_reward_40_steps']:.6f}"
            )
            log(
                json.dumps(
                    {
                        "compare_infill": compare_infill_name,
                        "test_log_path": summary["test_log_path"],
                        "disc": {k: v for k, v in disc_summary.items() if k not in {"history", "final_front"}},
                        "infill": {k: v for k, v in infill_summary.items() if k not in {"history", "final_front"}},
                    },
                    indent=2,
                )
            )
        else:
            log(f"mean reward ({disc_summary['evolution_fe']} steps) = {disc_summary['mean_reward_40_steps']:.6f}")
            log(json.dumps({k: v for k, v in disc_summary.items() if k not in {"history", "final_front"}}, indent=2))
    finally:
        log_fp.close()


if __name__ == "__main__":
    main()
