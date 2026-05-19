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
