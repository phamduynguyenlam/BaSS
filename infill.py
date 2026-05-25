from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from reward import hypervolume, pareto_front


class InfillCriterion(ABC):
    @abstractmethod
    def score_candidates(
        self,
        *,
        archive_y: np.ndarray,
        candidate_mean: np.ndarray,
        candidate_std: np.ndarray,
        seed: int | None = None,
    ) -> np.ndarray: ...

    def select_index(
        self,
        *,
        archive_y: np.ndarray,
        candidate_mean: np.ndarray,
        candidate_std: np.ndarray,
        seed: int | None = None,
    ) -> tuple[int, np.ndarray]:
        scores = np.asarray(
            self.score_candidates(
                archive_y=np.asarray(archive_y, dtype=np.float32),
                candidate_mean=np.asarray(candidate_mean, dtype=np.float32),
                candidate_std=np.asarray(candidate_std, dtype=np.float32),
                seed=seed,
            ),
            dtype=np.float32,
        ).reshape(-1)
        if scores.size <= 0:
            raise ValueError("InfillCriterion received no candidate scores.")
        return int(np.argmax(scores)), scores


class ExpectedHypervolumeImprovement(InfillCriterion):
    def __init__(
        self,
        *,
        ref_point: np.ndarray,
        n_samples: int = 64,
        min_std: float = 1e-6,
    ):
        self.ref_point = np.asarray(ref_point, dtype=np.float32).reshape(-1)
        self.n_samples = max(1, int(n_samples))
        self.min_std = float(max(min_std, 1e-12))

    def score_candidates(
        self,
        *,
        archive_y: np.ndarray,
        candidate_mean: np.ndarray,
        candidate_std: np.ndarray,
        seed: int | None = None,
    ) -> np.ndarray:
        archive_front = pareto_front(np.asarray(archive_y, dtype=np.float32))
        cand_mean = np.asarray(candidate_mean, dtype=np.float32)
        cand_std = np.asarray(candidate_std, dtype=np.float32)
        if cand_mean.ndim != 2:
            raise ValueError(f"candidate_mean must be 2D, got shape={cand_mean.shape}.")
        if cand_std.ndim == 1:
            cand_std = cand_std.reshape(-1, 1)
        if cand_std.shape != cand_mean.shape:
            raise ValueError(
                f"candidate_std must match candidate_mean shape, got {cand_std.shape} vs {cand_mean.shape}."
            )

        rng = np.random.default_rng(seed)
        base_hv = float(hypervolume(archive_front, self.ref_point))
        scores = np.zeros(int(cand_mean.shape[0]), dtype=np.float32)

        for idx in range(int(cand_mean.shape[0])):
            mean_i = np.asarray(cand_mean[idx], dtype=np.float32)
            std_i = np.maximum(np.asarray(cand_std[idx], dtype=np.float32), self.min_std)
            samples = rng.normal(
                loc=mean_i.reshape(1, -1),
                scale=std_i.reshape(1, -1),
                size=(self.n_samples, int(mean_i.shape[0])),
            ).astype(np.float32)
            hv_improvements = np.zeros(self.n_samples, dtype=np.float32)
            for sample_idx, sample in enumerate(samples):
                hv_after = float(hypervolume(np.vstack([archive_front, sample.reshape(1, -1)]), self.ref_point))
                hv_improvements[sample_idx] = max(0.0, hv_after - base_hv)
            scores[idx] = float(np.mean(hv_improvements))

        return scores.astype(np.float32)


def _normalize_scalar(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    min_v = float(arr.min())
    max_v = float(arr.max())
    span = max(max_v - min_v, 1e-12)
    return ((arr - min_v) / span).astype(np.float32)


def _normalize_objectives(values: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float32)
    mins_arr = np.asarray(mins, dtype=np.float32)
    maxs_arr = np.asarray(maxs, dtype=np.float32)
    span = np.maximum(maxs_arr - mins_arr, 1e-12)
    return (values_arr - mins_arr) / span


def _vector_angle(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    denom = float(np.linalg.norm(x_arr) * np.linalg.norm(y_arr))
    if denom <= 1e-12:
        return 0.0
    cos_val = float(np.clip(float(np.dot(x_arr, y_arr)) / denom, -1.0, 1.0))
    return float(np.arccos(cos_val))


def _simplex_reference_vectors(n_obj: int, n_partitions: int) -> np.ndarray:
    refs: list[np.ndarray] = []
    if int(n_obj) == 2:
        for i in range(int(n_partitions) + 1):
            refs.append(np.array([i, int(n_partitions) - i], dtype=np.float32) / max(int(n_partitions), 1))
    elif int(n_obj) == 3:
        for i in range(int(n_partitions) + 1):
            for j in range(int(n_partitions) + 1 - i):
                k = int(n_partitions) - i - j
                refs.append(np.array([i, j, k], dtype=np.float32) / max(int(n_partitions), 1))
    else:
        refs = [np.eye(int(n_obj), dtype=np.float32)[i] for i in range(int(n_obj))]

    ref_vectors = np.asarray(refs, dtype=np.float32)
    norms = np.linalg.norm(ref_vectors, axis=1, keepdims=True)
    return ref_vectors / np.maximum(norms, 1e-12)


def _normalize_for_pbi(values: np.ndarray, reference_values: np.ndarray) -> np.ndarray:
    all_values = np.vstack([reference_values, values]).astype(np.float32)
    mins = all_values.min(axis=0)
    spans = np.maximum(all_values.max(axis=0) - mins, 1e-12)
    return (np.asarray(values, dtype=np.float32) - mins) / spans


def _pbi_stats(normalized_values: np.ndarray, ref_vectors: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d1_all = normalized_values @ ref_vectors.T
    proj = d1_all[..., None] * ref_vectors[None, :, :]
    diff = normalized_values[:, None, :] - proj
    d2_all = np.linalg.norm(diff, axis=2)
    assoc = np.argmin(d2_all, axis=1)
    row_idx = np.arange(normalized_values.shape[0], dtype=np.int64)
    d1 = d1_all[row_idx, assoc]
    d2 = d2_all[row_idx, assoc]
    pbi = d1 + float(theta) * d2
    return assoc.astype(np.int64), d1.astype(np.float32), pbi.astype(np.float32)


def _random_unit_reference_vector(n_obj: int, rng: np.random.Generator) -> np.ndarray:
    vec = rng.random(int(n_obj), dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        return np.full(int(n_obj), 1.0 / np.sqrt(max(int(n_obj), 1)), dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _pd_value(values: np.ndarray, ref_vector: np.ndarray, theta: float = 5.0) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float32)
    ref_vector_arr = np.asarray(ref_vector, dtype=np.float32)
    ref_norm = np.linalg.norm(ref_vector_arr)
    if ref_norm <= 1e-12:
        ref_vector_arr = np.full(ref_vector_arr.shape[0], 1.0 / np.sqrt(max(ref_vector_arr.shape[0], 1)), dtype=np.float32)
        ref_norm = np.linalg.norm(ref_vector_arr)

    d1 = (values_arr @ ref_vector_arr) / max(ref_norm, 1e-12)
    projection = (d1 / max(ref_norm, 1e-12))[:, None] * ref_vector_arr[None, :]
    d2 = np.linalg.norm(values_arr - projection, axis=1)
    return d1 + float(theta) * d2


def _nd_a_components(candidate_values: np.ndarray, archive_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    candidate_values_arr = np.asarray(candidate_values, dtype=np.float32)
    archive_front = pareto_front(np.asarray(archive_values, dtype=np.float32))
    combined = np.vstack([candidate_values_arr, archive_front]).astype(np.float32)
    mins = combined.min(axis=0)
    maxs = combined.max(axis=0)
    candidate_norm = _normalize_objectives(candidate_values_arr, mins, maxs)
    archive_norm = _normalize_objectives(archive_front, mins, maxs)

    angles = np.zeros(candidate_values_arr.shape[0], dtype=np.float32)
    distances = np.zeros(candidate_values_arr.shape[0], dtype=np.float32)
    for idx, candidate in enumerate(candidate_norm):
        if archive_norm.shape[0] == 0:
            angles[idx] = 0.0
            distances[idx] = 0.0
        else:
            angle_values = np.asarray([_vector_angle(candidate, archive_point) for archive_point in archive_norm], dtype=np.float32)
            distance_values = np.linalg.norm(archive_norm - candidate[None, :], axis=1).astype(np.float32)
            angles[idx] = float(angle_values.min())
            distances[idx] = float(distance_values.min())
    return angles, distances


def _nd_pbi_branch_params(focus: str) -> tuple[float, float]:
    focus_name = str(focus).lower()
    if focus_name == "convergence":
        return 2.0, 0.0
    if focus_name == "diversity":
        return 8.0, 0.5
    raise ValueError(f"Unsupported ND-PBI focus: {focus}")


def _ensure_sigma_shape(candidate_mean: np.ndarray, candidate_std: np.ndarray | None) -> np.ndarray:
    cand_mean = np.asarray(candidate_mean, dtype=np.float32)
    if candidate_std is None:
        return np.full_like(cand_mean, 1e-6, dtype=np.float32)
    cand_std_arr = np.asarray(candidate_std, dtype=np.float32)
    if cand_std_arr.ndim == 1:
        cand_std_arr = cand_std_arr.reshape(-1, 1)
    if cand_std_arr.shape[1] == 1 and cand_mean.shape[1] > 1:
        cand_std_arr = np.repeat(cand_std_arr, cand_mean.shape[1], axis=1)
    if cand_std_arr.shape != cand_mean.shape:
        raise ValueError(f"candidate_std must match candidate_mean shape, got {cand_std_arr.shape} vs {cand_mean.shape}.")
    return cand_std_arr.astype(np.float32)


class NDA(InfillCriterion):
    def __init__(self, *, diversity_lambda: float = 1.0):
        self.diversity_lambda = float(diversity_lambda)

    def score_candidates(self, *, archive_y: np.ndarray, candidate_mean: np.ndarray, candidate_std: np.ndarray, seed: int | None = None) -> np.ndarray:
        del seed
        cand_mean = np.asarray(candidate_mean, dtype=np.float32)
        cand_std = _ensure_sigma_shape(cand_mean, candidate_std)
        penalized_values = cand_mean + cand_std
        angles, _ = _nd_a_components(penalized_values, archive_y)
        uncertainty = _normalize_scalar(cand_std.mean(axis=1))
        return (angles + self.diversity_lambda * uncertainty).astype(np.float32)


class NDPBIConvergence(InfillCriterion):
    def score_candidates(self, *, archive_y: np.ndarray, candidate_mean: np.ndarray, candidate_std: np.ndarray, seed: int | None = None) -> np.ndarray:
        del seed
        cand_mean = np.asarray(candidate_mean, dtype=np.float32)
        cand_std = _ensure_sigma_shape(cand_mean, candidate_std)
        theta, empty_bonus = _nd_pbi_branch_params("convergence")
        penalized = cand_mean + cand_std
        archive_front = pareto_front(np.asarray(archive_y, dtype=np.float32))
        reference_values = np.vstack([archive_front, penalized]).astype(np.float32)
        ref_vectors = _simplex_reference_vectors(penalized.shape[1], n_partitions=max(12, penalized.shape[1] * 4))

        arnd_norm = _normalize_for_pbi(archive_front, reference_values)
        cand_norm = _normalize_for_pbi(penalized, reference_values)
        arnd_assoc, _, arnd_pbi = _pbi_stats(arnd_norm, ref_vectors, theta=theta)
        cand_assoc, cand_d1, cand_pbi = _pbi_stats(cand_norm, ref_vectors, theta=theta)

        nonempty_refs = set(int(idx) for idx in arnd_assoc.tolist())
        scores = np.zeros(penalized.shape[0], dtype=np.float32)
        for idx in range(penalized.shape[0]):
            assoc = int(cand_assoc[idx])
            if assoc in nonempty_refs:
                ref_mask = arnd_assoc == assoc
                pbi_min = float(np.min(arnd_pbi[ref_mask]))
                improvement = pbi_min - float(cand_pbi[idx])
            else:
                improvement = float(empty_bonus) - float(cand_pbi[idx])
            scores[idx] = float(improvement - 0.05 * float(cand_d1[idx]))
        return scores.astype(np.float32)


class NDPBIDiversity(InfillCriterion):
    def score_candidates(self, *, archive_y: np.ndarray, candidate_mean: np.ndarray, candidate_std: np.ndarray, seed: int | None = None) -> np.ndarray:
        del seed
        cand_mean = np.asarray(candidate_mean, dtype=np.float32)
        cand_std = _ensure_sigma_shape(cand_mean, candidate_std)
        theta, empty_bonus = _nd_pbi_branch_params("diversity")
        penalized = cand_mean + cand_std
        archive_front = pareto_front(np.asarray(archive_y, dtype=np.float32))
        reference_values = np.vstack([archive_front, penalized]).astype(np.float32)
        ref_vectors = _simplex_reference_vectors(penalized.shape[1], n_partitions=max(12, penalized.shape[1] * 4))

        arnd_norm = _normalize_for_pbi(archive_front, reference_values)
        cand_norm = _normalize_for_pbi(penalized, reference_values)
        arnd_assoc, _, arnd_pbi = _pbi_stats(arnd_norm, ref_vectors, theta=theta)
        cand_assoc, _, cand_pbi = _pbi_stats(cand_norm, ref_vectors, theta=theta)

        nonempty_refs = set(int(idx) for idx in arnd_assoc.tolist())
        scores = np.zeros(penalized.shape[0], dtype=np.float32)
        for idx in range(penalized.shape[0]):
            assoc = int(cand_assoc[idx])
            if assoc in nonempty_refs:
                ref_mask = arnd_assoc == assoc
                pbi_min = float(np.min(arnd_pbi[ref_mask]))
                improvement = pbi_min - float(cand_pbi[idx])
            else:
                improvement = float(empty_bonus) - float(cand_pbi[idx])
            scores[idx] = float(improvement + float(empty_bonus))
        return scores.astype(np.float32)


class EPDIExploitation(InfillCriterion):
    def __init__(self, *, mc_samples: int = 1000):
        self.mc_samples = max(1, int(mc_samples))

    def score_candidates(self, *, archive_y: np.ndarray, candidate_mean: np.ndarray, candidate_std: np.ndarray, seed: int | None = None) -> np.ndarray:
        cand_mean = np.asarray(candidate_mean, dtype=np.float32)
        cand_std = _ensure_sigma_shape(cand_mean, candidate_std)
        archive_front = pareto_front(np.asarray(archive_y, dtype=np.float32))
        combined = np.vstack([archive_front, cand_mean]).astype(np.float32)
        mins = combined.min(axis=0)
        maxs = combined.max(axis=0)
        archive_norm = _normalize_objectives(archive_front, mins, maxs)
        candidate_norm = _normalize_objectives(cand_mean, mins, maxs)
        sigma_norm = cand_std / np.maximum(maxs - mins, 1e-12)

        rng = np.random.default_rng(seed)
        mean_epdi = np.zeros(candidate_norm.shape[0], dtype=np.float32)
        for idx in range(candidate_norm.shape[0]):
            ref_vector = _random_unit_reference_vector(candidate_norm.shape[1], rng)
            pd_min = float(np.min(_pd_value(archive_norm, ref_vector))) if archive_norm.size else 0.0
            sigma = np.maximum(sigma_norm[idx], 1e-6)
            samples = rng.normal(loc=candidate_norm[idx], scale=sigma, size=(self.mc_samples, candidate_norm.shape[1])).astype(np.float32)
            samples = np.clip(samples, 0.0, 1.5)
            pdi_samples = np.maximum(pd_min - _pd_value(samples, ref_vector), 0.0)
            mean_epdi[idx] = float(np.mean(pdi_samples))
        return mean_epdi.astype(np.float32)


class EPDIExploration(InfillCriterion):
    def __init__(self, *, mc_samples: int = 1000):
        self.mc_samples = max(1, int(mc_samples))

    def score_candidates(self, *, archive_y: np.ndarray, candidate_mean: np.ndarray, candidate_std: np.ndarray, seed: int | None = None) -> np.ndarray:
        cand_mean = np.asarray(candidate_mean, dtype=np.float32)
        cand_std = _ensure_sigma_shape(cand_mean, candidate_std)
        archive_front = pareto_front(np.asarray(archive_y, dtype=np.float32))
        combined = np.vstack([archive_front, cand_mean]).astype(np.float32)
        mins = combined.min(axis=0)
        maxs = combined.max(axis=0)
        archive_norm = _normalize_objectives(archive_front, mins, maxs)
        candidate_norm = _normalize_objectives(cand_mean, mins, maxs)
        sigma_norm = cand_std / np.maximum(maxs - mins, 1e-12)

        rng = np.random.default_rng(seed)
        scores = np.zeros(candidate_norm.shape[0], dtype=np.float32)
        for idx in range(candidate_norm.shape[0]):
            ref_vector = _random_unit_reference_vector(candidate_norm.shape[1], rng)
            pd_min = float(np.min(_pd_value(archive_norm, ref_vector))) if archive_norm.size else 0.0
            sigma = np.maximum(sigma_norm[idx], 1e-6)
            samples = rng.normal(loc=candidate_norm[idx], scale=sigma, size=(self.mc_samples, candidate_norm.shape[1])).astype(np.float32)
            samples = np.clip(samples, 0.0, 1.5)
            pdi_samples = np.maximum(pd_min - _pd_value(samples, ref_vector), 0.0)
            scores[idx] = float(np.mean(pdi_samples) + np.std(pdi_samples))
        return scores.astype(np.float32)
