from __future__ import annotations

import numpy as np
from pymoo.indicators.hv import HV


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Pareto dominance for minimization."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return bool(np.all(a <= b) and np.any(a < b))


def pareto_front(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values.reshape(0, 0).astype(np.float32)
    if values.ndim != 2:
        raise ValueError(f"values must be 2D, got shape={values.shape}")

    keep: list[int] = []
    for i in range(values.shape[0]):
        dominated = False
        for j in range(values.shape[0]):
            if i == j:
                continue
            if _dominates(values[j], values[i]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return values[np.asarray(keep, dtype=np.int64)]


def hypervolume(values: np.ndarray, ref_point: np.ndarray) -> float:
    front = pareto_front(values)
    if front.size == 0:
        return 0.0
    return float(HV(ref_point=np.asarray(ref_point, dtype=np.float32))(front))


def _normalize_joint(true_values: np.ndarray, pred_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true_arr = np.asarray(true_values, dtype=np.float32)
    pred_arr = np.asarray(pred_values, dtype=np.float32)
    stacked = np.vstack([true_arr, pred_arr])
    min_v = stacked.min(axis=0, keepdims=True)
    max_v = stacked.max(axis=0, keepdims=True)
    denom = np.clip(max_v - min_v, 1e-12, None)
    true_norm = (true_arr - min_v) / denom
    pred_norm = (pred_arr - min_v) / denom
    return true_norm.astype(np.float32), pred_norm.astype(np.float32)


def archive_fit_mse_reward(
    *,
    archive_true_y: np.ndarray,
    archive_pred_y: np.ndarray,
) -> float:
    archive_true = np.asarray(archive_true_y, dtype=np.float32)
    archive_pred = np.asarray(archive_pred_y, dtype=np.float32)
    if archive_true.ndim != 2 or archive_pred.ndim != 2:
        raise ValueError("archive_true_y and archive_pred_y must be 2D.")
    if archive_true.shape != archive_pred.shape:
        raise ValueError(
            f"archive_true_y and archive_pred_y shape mismatch: {archive_true.shape} vs {archive_pred.shape}."
        )
    if archive_true.shape[0] <= 0:
        return 0.0

    true_norm, pred_norm = _normalize_joint(archive_true, archive_pred)
    per_entry_mse = np.mean((true_norm - pred_norm) ** 2, axis=1)
    return float(np.mean(per_entry_mse))


def _filter_candidates_on_front(
    selected_objectives: np.ndarray,
    combined_front: np.ndarray,
    *,
    atol: float = 1e-6,
) -> np.ndarray:
    selected_arr = np.asarray(selected_objectives, dtype=np.float32)
    if selected_arr.ndim == 1:
        selected_arr = selected_arr.reshape(1, -1)
    combined_arr = np.asarray(combined_front, dtype=np.float32)
    if combined_arr.size == 0:
        return selected_arr[:0]

    keep: list[np.ndarray] = []
    for candidate in selected_arr:
        matches = np.isclose(combined_arr, candidate[None, :], atol=float(atol), rtol=0.0)
        if bool(np.any(np.all(matches, axis=1))):
            keep.append(candidate)

    if len(keep) == 0:
        return np.empty((0, selected_arr.shape[1]), dtype=np.float32)
    return np.asarray(keep, dtype=np.float32)


def hv_improvement_reward(
    *,
    previous_archive: np.ndarray,
    selected_objectives: np.ndarray,
    ref_point: np.ndarray,
    epsilon: float = 1e-8,
    no_improve_reward: float = -1.0,
) -> float:
    previous_archive = np.asarray(previous_archive, dtype=np.float32)
    selected_objectives = np.asarray(selected_objectives, dtype=np.float32)
    combined = np.vstack([previous_archive, selected_objectives])

    prev_hv = hypervolume(previous_archive, ref_point)
    next_hv = hypervolume(combined, ref_point)
    if next_hv <= prev_hv:
        return float(no_improve_reward)
    return float((next_hv - prev_hv) / (prev_hv + float(epsilon)))


def fpareto_improvement_reward(
    *,
    previous_front: np.ndarray,
    selected_objectives: np.ndarray,
    no_improve_reward: float = -1.0,
) -> float:
    """Legacy 'fpareto' reward used by older demos (distance-to-front with an improvement gate)."""
    previous_front = pareto_front(np.asarray(previous_front, dtype=np.float32))
    selected_objectives = np.asarray(selected_objectives, dtype=np.float32)

    improved = False
    for candidate in selected_objectives:
        if not any(_dominates(prev, candidate) for prev in previous_front):
            improved = True
            break
    if not improved:
        return float(no_improve_reward)

    reward = 1.0
    origin = np.zeros(previous_front.shape[1], dtype=np.float32)
    for candidate in selected_objectives:
        distances = np.abs(previous_front - candidate).sum(axis=1)
        nearest_idx = int(np.argmin(distances))
        d_i = float(distances[nearest_idx])
        d_ref_i = float(np.abs(previous_front[nearest_idx] - origin).sum())
        reward += d_i / max(d_ref_i, 1e-12)
    return float(reward)


def reward_scheme_1(
    *,
    previous_front: np.ndarray,
    selected_objectives: np.ndarray,
    ref_point: np.ndarray,
    reward_lambda: float = 10.0,
) -> float:
    """Distance-to-front reward, scaled and offset; positive iff a new point stays on the updated Pareto front."""
    previous_front = pareto_front(np.asarray(previous_front, dtype=np.float32))
    selected_objectives = np.asarray(selected_objectives, dtype=np.float32)
    combined_front = pareto_front(np.vstack([previous_front, selected_objectives]))
    front_members = _filter_candidates_on_front(selected_objectives, combined_front)
    if front_members.shape[0] == 0:
        return -1.0

    if previous_front.size == 0:
        return float(max(1e-6, 1.0 + float(reward_lambda) * float(front_members.shape[0])))

    reward = 0.0
    origin = np.zeros(previous_front.shape[1], dtype=np.float32)
    for candidate in front_members:
        distances = np.abs(previous_front - candidate).sum(axis=1)
        nearest_idx = int(np.argmin(distances))
        d_i = float(distances[nearest_idx])
        d_ref_i = float(np.abs(previous_front[nearest_idx] - origin).sum())
        reward += d_i / max(d_ref_i, 1e-12)
    return float(max(1e-6, 1.0 + float(reward_lambda) * reward))


def reward_scheme_2(
    *,
    previous_front: np.ndarray,
    selected_objectives: np.ndarray,
    ref_point: np.ndarray,
    reward_lambda: float = 10.0,
) -> float:
    """Distance-to-front reward; positive iff a new point stays on the updated Pareto front."""
    previous_front = pareto_front(np.asarray(previous_front, dtype=np.float32))
    selected_objectives = np.asarray(selected_objectives, dtype=np.float32)
    combined_front = pareto_front(np.vstack([previous_front, selected_objectives]))
    front_members = _filter_candidates_on_front(selected_objectives, combined_front)
    if front_members.shape[0] == 0:
        return 0.0

    if previous_front.size == 0:
        return float(max(1e-6, float(reward_lambda) * float(front_members.shape[0])))

    reward = 0.0
    origin = np.zeros(previous_front.shape[1], dtype=np.float32)
    for candidate in front_members:
        distances = np.abs(previous_front - candidate).sum(axis=1)
        nearest_idx = int(np.argmin(distances))
        d_i = float(distances[nearest_idx])
        d_ref_i = float(np.abs(previous_front[nearest_idx] - origin).sum())
        reward += d_i / max(d_ref_i, 1e-12)
    return float(max(1e-6, float(reward_lambda) * reward))


def reward_scheme_3(
    *,
    previous_front: np.ndarray,
    selected_objectives: np.ndarray,
    ref_point: np.ndarray,
    true_pareto_hv: float,
    archive_true_y: np.ndarray,
    archive_pred_y: np.ndarray,
    hv_lambda: float = 25.0,
    fit_lambda: float = -1.0,
    hv_epsilon: float = 1e-8,
) -> float:
    """Reward = hv_lambda * normalized HV gain + fit_lambda * archive fit MSE after surrogate refit."""
    previous_front = np.asarray(previous_front, dtype=np.float32)
    selected_objectives = np.asarray(selected_objectives, dtype=np.float32)
    combined_front = np.vstack([previous_front, selected_objectives])

    prev_hv = hypervolume(previous_front, ref_point)
    next_hv = hypervolume(combined_front, ref_point)
    hv_term = 0.0
    if next_hv > prev_hv:
        remaining_gap = max(float(true_pareto_hv) - float(next_hv), float(hv_epsilon))
        hv_term = float(hv_lambda) * (float(next_hv) - float(prev_hv)) / remaining_gap

    fit_term = float(fit_lambda) * archive_fit_mse_reward(
        archive_true_y=archive_true_y,
        archive_pred_y=archive_pred_y,
    )
    return float(hv_term + fit_term)


def reward_scheme_1_breakdown(
    *,
    previous_front: np.ndarray,
    selected_objectives: np.ndarray,
    ref_point: np.ndarray,
    reward_lambda: float = 10.0,
) -> dict[str, float]:
    del ref_point
    previous_front = pareto_front(np.asarray(previous_front, dtype=np.float32))
    selected_objectives = np.asarray(selected_objectives, dtype=np.float32)
    combined_front = pareto_front(np.vstack([previous_front, selected_objectives]))
    front_members = _filter_candidates_on_front(selected_objectives, combined_front)
    if front_members.shape[0] == 0:
        return {
            "reward_total": -1.0,
            "front_count": 0.0,
            "distance_raw": 0.0,
            "distance_term": -1.0,
        }

    if previous_front.size == 0:
        reward_total = float(max(1e-6, 1.0 + float(reward_lambda) * float(front_members.shape[0])))
        return {
            "reward_total": reward_total,
            "front_count": float(front_members.shape[0]),
            "distance_raw": float(front_members.shape[0]),
            "distance_term": reward_total - 1.0,
        }

    reward = 0.0
    origin = np.zeros(previous_front.shape[1], dtype=np.float32)
    for candidate in front_members:
        distances = np.abs(previous_front - candidate).sum(axis=1)
        nearest_idx = int(np.argmin(distances))
        d_i = float(distances[nearest_idx])
        d_ref_i = float(np.abs(previous_front[nearest_idx] - origin).sum())
        reward += d_i / max(d_ref_i, 1e-12)
    return {
        "reward_total": float(max(1e-6, 1.0 + float(reward_lambda) * reward)),
        "front_count": float(front_members.shape[0]),
        "distance_raw": float(reward),
        "distance_term": float(float(reward_lambda) * reward),
    }


def reward_scheme_2_breakdown(
    *,
    previous_front: np.ndarray,
    selected_objectives: np.ndarray,
    ref_point: np.ndarray,
    reward_lambda: float = 10.0,
) -> dict[str, float]:
    del ref_point
    previous_front = pareto_front(np.asarray(previous_front, dtype=np.float32))
    selected_objectives = np.asarray(selected_objectives, dtype=np.float32)
    combined_front = pareto_front(np.vstack([previous_front, selected_objectives]))
    front_members = _filter_candidates_on_front(selected_objectives, combined_front)
    if front_members.shape[0] == 0:
        return {
            "reward_total": 0.0,
            "front_count": 0.0,
            "distance_raw": 0.0,
            "distance_term": 0.0,
        }

    if previous_front.size == 0:
        reward_total = float(max(1e-6, float(reward_lambda) * float(front_members.shape[0])))
        return {
            "reward_total": reward_total,
            "front_count": float(front_members.shape[0]),
            "distance_raw": float(front_members.shape[0]),
            "distance_term": reward_total,
        }

    reward = 0.0
    origin = np.zeros(previous_front.shape[1], dtype=np.float32)
    for candidate in front_members:
        distances = np.abs(previous_front - candidate).sum(axis=1)
        nearest_idx = int(np.argmin(distances))
        d_i = float(distances[nearest_idx])
        d_ref_i = float(np.abs(previous_front[nearest_idx] - origin).sum())
        reward += d_i / max(d_ref_i, 1e-12)
    return {
        "reward_total": float(max(1e-6, float(reward_lambda) * reward)),
        "front_count": float(front_members.shape[0]),
        "distance_raw": float(reward),
        "distance_term": float(float(reward_lambda) * reward),
    }


def reward_scheme_3_breakdown(
    *,
    previous_front: np.ndarray,
    selected_objectives: np.ndarray,
    ref_point: np.ndarray,
    true_pareto_hv: float,
    archive_true_y: np.ndarray,
    archive_pred_y: np.ndarray,
    hv_lambda: float = 25.0,
    fit_lambda: float = -1.0,
    hv_epsilon: float = 1e-8,
) -> dict[str, float]:
    previous_front = np.asarray(previous_front, dtype=np.float32)
    selected_objectives = np.asarray(selected_objectives, dtype=np.float32)
    combined_front = np.vstack([previous_front, selected_objectives])

    prev_hv = hypervolume(previous_front, ref_point)
    next_hv = hypervolume(combined_front, ref_point)
    hv_gain = max(float(next_hv) - float(prev_hv), 0.0)
    remaining_gap = max(float(true_pareto_hv) - float(next_hv), float(hv_epsilon))
    hv_term = 0.0
    if next_hv > prev_hv:
        hv_term = float(hv_lambda) * hv_gain / remaining_gap

    fit_mse = archive_fit_mse_reward(
        archive_true_y=archive_true_y,
        archive_pred_y=archive_pred_y,
    )
    fit_term = float(fit_lambda) * fit_mse
    return {
        "reward_total": float(hv_term + fit_term),
        "prev_hv": float(prev_hv),
        "next_hv": float(next_hv),
        "hv_gain": float(hv_gain),
        "remaining_gap": float(remaining_gap),
        "hv_term": float(hv_term),
        "fit_mse": float(fit_mse),
        "fit_term": float(fit_term),
    }
