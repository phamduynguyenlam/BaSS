from __future__ import annotations

import contextlib
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from surrogate.kan import KAN

os.environ["TABPFN_TOKEN"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiMWYwZGEyZWEtOGY1Zi00MGNiLWJlNzUtN2U1OTI2YTAxZGFlIiwiZXhwIjoxODA5MDE1MjA3fQ.rFxK90AswdPigPC-vBVUmAELAiVtOvy5YNGfTDUam8A"

def _hydrate_tabpfn_env_from_windows_user_env() -> None:
    if sys.platform != "win32":
        return
    wanted = [key for key in ("TABPFN_TOKEN", "TABPFN_NO_BROWSER") if not os.environ.get(key)]
    if not wanted:
        return
    try:
        import winreg  # type: ignore

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
            for name in wanted:
                try:
                    value, _ = winreg.QueryValueEx(key, name)
                except OSError:
                    continue
                if isinstance(value, str) and value.strip():
                    os.environ[name] = value.strip()
    except Exception:
        pass


_hydrate_tabpfn_env_from_windows_user_env()


def surrogate_model_name(args) -> str:
    return getattr(args, "surrogate_model", getattr(args, "uncertainty_model", "gp"))


def build_dataset(x: np.ndarray, y: np.ndarray, device: str) -> dict[str, torch.Tensor]:
    n = int(x.shape[0])
    n_train = max(2, int(0.8 * n))
    perm = np.random.permutation(n)
    train_id = perm[:n_train]
    test_id = perm[n_train:] if n_train < n else perm[: min(2, n)]
    return {
        "train_input": torch.tensor(x[train_id], dtype=torch.float32, device=device),
        "train_label": torch.tensor(y[train_id], dtype=torch.float32, device=device),
        "test_input": torch.tensor(x[test_id], dtype=torch.float32, device=device),
        "test_label": torch.tensor(y[test_id], dtype=torch.float32, device=device),
    }


class Surrogate(ABC):
    @abstractmethod
    def predict_mean(self, x: np.ndarray, device: str | None = None) -> np.ndarray: ...

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def refit(self, x: np.ndarray, y: np.ndarray) -> "Surrogate":
        raise NotImplementedError


@dataclass
class KANSurrogateModel(Surrogate):
    models: list[KAN]
    device: str

    def predict_mean(self, x: np.ndarray, device: str | None = None) -> np.ndarray:
        dev = self.device if device is None else str(device)
        return predict_with_kan(self.models, x, dev)

    def refit(self, x: np.ndarray, y: np.ndarray) -> "KANSurrogateModel":
        raise NotImplementedError("KANSurrogateModel does not implement in-place refit; rebuild it via fit_kan_surrogates().")


@dataclass
class TabPFNSurrogateModel(Surrogate):
    model: Any

    def predict_mean(self, x: np.ndarray, device: str | None = None) -> np.ndarray:
        return np.asarray(self.model.predict(np.asarray(x, dtype=np.float32)), dtype=np.float32)

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self.model, "predict_std"):
            raise NotImplementedError("TabPFN surrogate wrapper requires predict_std().")
        return np.asarray(self.model.predict_std(np.asarray(x, dtype=np.float32)), dtype=np.float32)

    def refit(self, x: np.ndarray, y: np.ndarray) -> "TabPFNSurrogateModel":
        if not hasattr(self.model, "fit"):
            raise NotImplementedError("Wrapped TabPFN surrogate does not implement fit().")
        self.model.fit(np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32))
        return self


def fit_kan_surrogates(
    *,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    device: str,
    kan_steps: int,
    hidden_width: int,
    grid: int,
    seed: int,
) -> list[KAN]:
    archive_x = np.asarray(archive_x, dtype=np.float32)
    archive_y = np.asarray(archive_y, dtype=np.float32)
    models: list[KAN] = []
    for obj_id in range(int(archive_y.shape[1])):
        dataset = build_dataset(archive_x, archive_y[:, [obj_id]], device)
        model = KAN(
            width=[archive_x.shape[1], int(hidden_width), 1],
            grid=int(grid),
            k=3,
            seed=int(seed) + int(obj_id),
            device=str(device),
            auto_save=False,
            save_act=False,
        )
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                model.fit(
                    dataset,
                    opt="Adam",
                    steps=int(kan_steps),
                    lr=1e-2,
                    batch=-1,
                    update_grid=False,
                    lamb=0.0,
                    log=1,
                )
        models.append(model)
    return models


def predict_with_kan(models: Sequence[Any], x: np.ndarray, device: str) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    x_tensor = torch.tensor(x_arr, dtype=torch.float32, device=str(device))
    preds: list[np.ndarray] = []
    for model in models:
        with torch.no_grad():
            pred = model(x_tensor).detach().cpu().numpy().reshape(-1)
        preds.append(np.asarray(pred, dtype=np.float32))
    return np.stack(preds, axis=1).astype(np.float32)


def estimate_uncertainty(
    *,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    archive_pred: np.ndarray,
    offspring_x: np.ndarray,
    n_neighbors: int = 5,
) -> np.ndarray:
    archive_x = np.asarray(archive_x, dtype=np.float32)
    archive_y = np.asarray(archive_y, dtype=np.float32)
    archive_pred = np.asarray(archive_pred, dtype=np.float32)
    offspring_x = np.asarray(offspring_x, dtype=np.float32)
    residual = np.abs(archive_pred - archive_y)
    n_neighbors = min(int(n_neighbors), int(archive_x.shape[0]))
    dist = np.linalg.norm(offspring_x[:, None, :] - archive_x[None, :, :], axis=-1)
    nn_idx = np.argsort(dist, axis=1)[:, :n_neighbors]
    local_residual = residual[nn_idx]
    return local_residual.mean(axis=1).astype(np.float32) + 1e-6


def init_uncertainty_archive(archive_x: np.ndarray, archive_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(archive_x, dtype=np.float32).copy(),
        np.asarray(archive_y, dtype=np.float32).copy(),
    )


def update_uncertainty_archive(
    *,
    uncertainty_x: np.ndarray,
    uncertainty_y: np.ndarray,
    new_x: np.ndarray,
    new_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    uncertainty_x = np.asarray(uncertainty_x, dtype=np.float32)
    uncertainty_y = np.asarray(uncertainty_y, dtype=np.float32)
    new_x = np.asarray(new_x, dtype=np.float32)
    new_y = np.asarray(new_y, dtype=np.float32)

    if new_x.size == 0 or new_y.size == 0:
        return uncertainty_x, uncertainty_y

    if uncertainty_x.size == 0:
        merged_x = new_x
        merged_y = new_y
    else:
        merged_x = np.vstack([uncertainty_x, new_x])
        merged_y = np.vstack([uncertainty_y, new_y])

    unique_indices: list[int] = []
    for i in range(int(merged_x.shape[0])):
        is_duplicate = False
        for j in unique_indices:
            if np.allclose(merged_x[i], merged_x[j]) and np.allclose(merged_y[i], merged_y[j]):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_indices.append(i)

    idx = np.asarray(unique_indices, dtype=np.int64)
    return merged_x[idx], merged_y[idx]


def fit_tabpfn_surrogate(
    *,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    device: str,
    n_estimators: int = 8,
    debug: bool = False,
    existing_surrogate: Any | None = None,
) -> Any:
    archive_x = np.asarray(archive_x, dtype=np.float32)
    archive_y = np.asarray(archive_y, dtype=np.float32)
    if existing_surrogate is not None:
        if not isinstance(existing_surrogate, TabPFNMinMaxSurrogate):
            raise TypeError(
                f"existing_surrogate must be TabPFNMinMaxSurrogate when provided, got {type(existing_surrogate).__name__}."
            )
        if int(existing_surrogate.n_objectives) != int(archive_y.shape[1]):
            raise ValueError(
                f"existing_surrogate n_objectives={existing_surrogate.n_objectives} does not match archive_y.shape[1]={archive_y.shape[1]}."
            )
        if int(existing_surrogate.n_estimators) != int(n_estimators):
            raise ValueError(
                f"existing_surrogate n_estimators={existing_surrogate.n_estimators} does not match requested n_estimators={n_estimators}."
            )
        if str(existing_surrogate.tabpfn_device) != str(device):
            raise ValueError(
                f"existing_surrogate tabpfn_device={existing_surrogate.tabpfn_device} does not match requested device={device}."
            )
        existing_surrogate.debug = bool(debug)
        return existing_surrogate.fit(archive_x, archive_y)
    return TabPFNMinMaxSurrogate(
        n_objectives=int(archive_y.shape[1]),
        tabpfn_device=str(device),
        n_estimators=int(n_estimators),
        debug=bool(debug),
    ).fit(archive_x, archive_y)


# --- TabPFN bar-distribution surrogate (moved from tabpfn_surrogate.py) ---


def _as_1d_float(arr: np.ndarray | Sequence[float], *, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).reshape(-1)
    if out.ndim != 1 or out.size < 2:
        raise ValueError(f"{name} must be a 1D array with at least 2 elements, got shape={out.shape}.")
    return out


def _validate_bin_edges(bin_edges: np.ndarray) -> None:
    diffs = np.diff(bin_edges)
    if not np.all(np.isfinite(bin_edges)):
        raise ValueError("bin_edges must be finite.")
    if not np.all(diffs > 0):
        raise ValueError("bin_edges must be strictly increasing.")


def discretize_targets_to_bins(y: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Map continuous targets into bin indices in [0, K-1] using predefined edges."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    k = int(bin_edges.size - 1)
    idx = np.digitize(y, bin_edges[1:-1], right=False).astype(np.int64, copy=False)
    return np.clip(idx, 0, k - 1)


def uniform_bin_edges_from_targets(y: np.ndarray, n_bins: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        raise ValueError("Cannot create bin edges from empty targets.")
    n_bins = int(n_bins)
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}.")
    lo = float(np.min(y))
    hi = float(np.max(y))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Targets must be finite to build uniform bins.")
    if hi <= lo:
        hi = lo + 1e-3
    return np.linspace(lo, hi, n_bins + 1, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class TabPFNBins:
    """Discretization bins for TabPFN bar-distribution outputs."""

    edges: np.ndarray
    midpoints: np.ndarray

    @classmethod
    def from_edges(cls, edges: np.ndarray | Sequence[float]) -> "TabPFNBins":
        edges_arr = _as_1d_float(edges, name="bin_edges")
        _validate_bin_edges(edges_arr)
        mid = (edges_arr[:-1] + edges_arr[1:]) * 0.5
        return cls(edges=edges_arr, midpoints=mid.astype(np.float32))

    @property
    def k(self) -> int:
        return int(self.midpoints.size)


def tabpfn_probs_to_mean_std(
    probs: np.ndarray,
    bins: TabPFNBins,
    *,
    normalize: bool = True,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert discrete probabilities into mean/std using bin midpoints."""
    p = np.asarray(probs, dtype=np.float32)
    if p.ndim != 2:
        raise ValueError(f"probs must have shape (N, K), got shape={p.shape}.")
    if p.shape[1] != bins.k:
        raise ValueError(f"probs K={p.shape[1]} does not match bins K={bins.k}.")

    if normalize:
        denom = np.maximum(p.sum(axis=1, keepdims=True), float(eps))
        p = p / denom

    mu = bins.midpoints.reshape(1, -1)
    mean = np.sum(p * mu, axis=1)
    var = np.sum(p * (mu - mean.reshape(-1, 1)) ** 2, axis=1)
    std = np.sqrt(np.maximum(var, 0.0))
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_balanced_softmax_probs(
    logits: torch.Tensor,
    y_train_bins: np.ndarray,
    *,
    n_classes_full: int,
    active_classes: np.ndarray | None = None,
) -> np.ndarray:
    probs_tensor = torch.softmax(logits, dim=-1)
    probs = probs_tensor.detach().cpu().numpy().astype(np.float32)

    y_arr = np.asarray(y_train_bins, dtype=np.int64).reshape(-1)
    class_counts = np.bincount(y_arr, minlength=int(n_classes_full)).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    total_samples = float(max(int(y_arr.size), 1))
    weights_full = total_samples / (float(int(n_classes_full)) * class_counts)
    if active_classes is None:
        weights = weights_full[: probs.shape[1]]
    else:
        active = np.asarray(active_classes, dtype=np.int64).reshape(-1)
        weights = weights_full[active]

    reweighted = probs * weights.reshape(1, -1)
    denom = np.maximum(reweighted.sum(axis=-1, keepdims=True), 1e-12)
    return (reweighted / denom).astype(np.float32)


class TabPFNObjectiveSurrogate:
    """Single-objective TabPFN surrogate producing (mean, std) via bin probabilities."""

    def __init__(self, model: Any, bin_edges: np.ndarray | Sequence[float], *, debug: bool = False):
        self.model = model
        self.bins = TabPFNBins.from_edges(bin_edges)
        self.debug = bool(debug)
        self._fit_classes: np.ndarray | None = None
        self._fit_x: np.ndarray | None = None
        self._fit_y_bins: np.ndarray | None = None
        self._fit_class_permutation: np.ndarray | None = None
        self._fit_model_index: int = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TabPFNObjectiveSurrogate":
        t0 = time.perf_counter()
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
        if x_arr.ndim != 2:
            raise ValueError(f"x must have shape (N, d), got shape={x_arr.shape}.")
        if y_arr.shape[0] != x_arr.shape[0]:
            raise ValueError(
                f"x and y must have the same number of rows, got {x_arr.shape[0]} and {y_arr.shape[0]}."
            )

        y_bins = discretize_targets_to_bins(y_arr, self.bins.edges)
        t_discretize = time.perf_counter()
        if not hasattr(self.model, "fit"):
            raise TypeError("Wrapped model does not implement fit().")
        self.model.fit(x_arr, y_bins)
        t_clf_fit = time.perf_counter()

        self._fit_x = np.asarray(x_arr, dtype=np.float32).copy()
        self._fit_y_bins = np.asarray(y_bins, dtype=np.float32).copy()
        classes = getattr(self.model, "classes_", None)
        self._fit_classes = None if classes is None else np.asarray(classes, dtype=np.int64).reshape(-1)
        ensemble_members_count = 0
        try:
            ensemble_members_count = len(_get_tabpfn_ensemble_members(self.model))
        except Exception:
            pass
        ensemble_configs = getattr(self.model, "ensemble_configs_", None)
        if ensemble_configs:
            config0 = ensemble_configs[0]
            class_permutation = getattr(config0, "class_permutation", None)
            self._fit_class_permutation = None if class_permutation is None else np.asarray(class_permutation, dtype=np.int64)
            self._fit_model_index = int(getattr(config0, "_model_index", 0))
        t_cache = time.perf_counter()
        if self.debug:
            print(
                f"[TabPFN obj fit] rows={x_arr.shape[0]} dim={x_arr.shape[1]} "
                f"| bins={self.bins.k} "
                f"| discretize={t_discretize - t0:.4f}s "
                f"| clf_fit={t_clf_fit - t_discretize:.4f}s "
                f"| cache={t_cache - t_clf_fit:.4f}s "
                f"| ensemble_members={ensemble_members_count} "
                f"| total={t_cache - t0:.4f}s"
            )
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        p = np.asarray(self.model.predict_proba(x_arr), dtype=np.float32)
        if p.ndim != 2:
            raise ValueError(f"predict_proba must return (N, K), got shape={p.shape}.")
        if p.shape[1] == self.bins.k:
            return p

        classes = self._fit_classes
        if classes is None:
            raise ValueError(
                f"predict_proba returned K={p.shape[1]} but model has no classes_ to map into K={self.bins.k} bins."
            )

        full = np.zeros((p.shape[0], self.bins.k), dtype=np.float32)
        for col, cls_id in enumerate(classes.tolist()):
            if 0 <= int(cls_id) < self.bins.k:
                full[:, int(cls_id)] = p[:, col]
        return full

    def predict_mean_std(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.predict_proba(x)
        return tabpfn_probs_to_mean_std(p, self.bins, normalize=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        mean, _ = self.predict_mean_std(x)
        return mean

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        _, std = self.predict_mean_std(x)
        return std


class TabPFNSurrogate:
    """Multi-objective TabPFN surrogate (one classifier per objective)."""

    def __init__(self, objective_models: Sequence[Any], bin_edges: np.ndarray | Sequence[float], *, debug: bool = False):
        if not objective_models:
            raise ValueError("objective_models must be a non-empty sequence.")
        self.objectives = [TabPFNObjectiveSurrogate(model=m, bin_edges=bin_edges, debug=bool(debug)) for m in objective_models]

    @property
    def n_objectives(self) -> int:
        return int(len(self.objectives))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TabPFNSurrogate":
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        if x_arr.ndim != 2:
            raise ValueError(f"x must have shape (N, d), got shape={x_arr.shape}.")
        if y_arr.ndim != 2:
            raise ValueError(f"y must have shape (N, m), got shape={y_arr.shape}.")
        if y_arr.shape[1] != self.n_objectives:
            raise ValueError(f"y must have {self.n_objectives} objectives, got {y_arr.shape[1]}.")
        if y_arr.shape[0] != x_arr.shape[0]:
            raise ValueError(f"x and y must have the same number of rows, got {x_arr.shape[0]} and {y_arr.shape[0]}.")

        for obj_idx, obj in enumerate(self.objectives):
            obj.fit(x_arr, y_arr[:, obj_idx])
        return self

    def predict_mean_std(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means: list[np.ndarray] = []
        stds: list[np.ndarray] = []
        for obj in self.objectives:
            m, s = obj.predict_mean_std(x)
            means.append(m)
            stds.append(s)
        return np.stack(means, axis=1).astype(np.float32), np.stack(stds, axis=1).astype(np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        mean, _ = self.predict_mean_std(x)
        return mean

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        _, std = self.predict_mean_std(x)
        return std


def build_tabpfn_surrogate(
    n_objectives: int,
    bin_edges: np.ndarray | Sequence[float],
    *,
    tabpfn_device: str = "cpu",
    use_many_class_extension: bool = False,
    random_state: int | None = 0,
    n_estimators: int = 8,
    debug: bool = False,
) -> TabPFNSurrogate:
    """Factory helper that constructs TabPFN classifier surrogates (optional dependency)."""
    try:
        from tabpfn import TabPFNClassifier  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("tabpfn is not installed. Install it with `pip install tabpfn`.") from exc

    models: list[Any] = []
    for _ in range(int(n_objectives)):
        base = TabPFNClassifier(device=tabpfn_device, n_estimators=int(n_estimators))
        if use_many_class_extension:
            try:
                from tabpfn_extensions.manyclass_classifier import ManyClassClassifier  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise ImportError("tabpfn-extensions[many_class] required for use_many_class_extension=True.") from exc
            base = ManyClassClassifier(estimator=base, random_state=random_state)
        models.append(base)
    return TabPFNSurrogate(objective_models=models, bin_edges=bin_edges, debug=bool(debug))


class TabPFNMinMaxSurrogate(Surrogate):
    """TabPFN surrogate with per-fit min-max normalization and fixed 10-bin targets on [0, 1]."""

    def __init__(
        self,
        n_objectives: int,
        *,
        tabpfn_device: str = "cpu",
        use_many_class_extension: bool = False,
        random_state: int | None = 0,
        n_estimators: int = 8,
        debug: bool = False,
    ):
        self.n_objectives = int(n_objectives)
        if self.n_objectives <= 0:
            raise ValueError(f"n_objectives must be positive, got {n_objectives}.")
        self.tabpfn_device = str(tabpfn_device)
        self.use_many_class_extension = bool(use_many_class_extension)
        self.random_state = random_state
        self.n_estimators = int(n_estimators)
        self.debug = bool(debug)
        if self.n_estimators <= 0:
            raise ValueError(f"n_estimators must be positive, got {n_estimators}.")
        self.n_bins = 10

        self._x_min: np.ndarray | None = None
        self._x_rng: np.ndarray | None = None
        self._y_min: np.ndarray | None = None
        self._y_rng: np.ndarray | None = None
        self._n_train_samples: int | None = None
        self._bins: TabPFNBins = TabPFNBins.from_edges(
            np.linspace(0.0, 1.0, int(self.n_bins) + 1, dtype=np.float32)
        )
        self._model: TabPFNSurrogate | None = None

    @staticmethod
    def _minmax_fit(arr: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
        min_v = np.min(arr, axis=0).astype(np.float32)
        max_v = np.max(arr, axis=0).astype(np.float32)
        rng = np.maximum(max_v - min_v, float(eps)).astype(np.float32)
        return min_v, rng

    def _norm_x(self, x: np.ndarray) -> np.ndarray:
        if self._x_min is None or self._x_rng is None:
            raise RuntimeError("TabPFNMinMaxSurrogate is not fit yet (missing x stats).")
        return ((np.asarray(x, dtype=np.float32) - self._x_min) / self._x_rng).astype(np.float32)

    def _norm_y(self, y: np.ndarray) -> np.ndarray:
        if self._y_min is None or self._y_rng is None:
            raise RuntimeError("TabPFNMinMaxSurrogate is not fit yet (missing y stats).")
        return ((np.asarray(y, dtype=np.float32) - self._y_min) / self._y_rng).astype(np.float32)

    def _unnorm_y_mean_std(self, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._y_min is None or self._y_rng is None:
            raise RuntimeError("TabPFNMinMaxSurrogate is not fit yet (missing y stats).")
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        return (self._y_min + mean * self._y_rng).astype(np.float32), (std * self._y_rng).astype(np.float32)

    @staticmethod
    def _choose_k(n_samples: int) -> int:
        del n_samples
        return 10

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TabPFNMinMaxSurrogate":
        t0 = time.perf_counter()
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        if x_arr.ndim != 2:
            raise ValueError(f"x must have shape (N, d), got shape={x_arr.shape}.")
        if y_arr.ndim != 2:
            raise ValueError(f"y must have shape (N, m), got shape={y_arr.shape}.")
        if y_arr.shape[0] != x_arr.shape[0]:
            raise ValueError(f"x and y must have the same number of rows, got {x_arr.shape[0]} and {y_arr.shape[0]}.")
        if y_arr.shape[1] != self.n_objectives:
            raise ValueError(f"y must have {self.n_objectives} objectives, got {y_arr.shape[1]}.")

        self._n_train_samples = int(x_arr.shape[0])
        self._x_min, self._x_rng = self._minmax_fit(x_arr)
        self._y_min, self._y_rng = self._minmax_fit(y_arr)
        t_norm_stats = time.perf_counter()

        x_norm = self._norm_x(x_arr)
        y_norm = self._norm_y(y_arr)
        t_norm_apply = time.perf_counter()

        t_bins = time.perf_counter()
        if self._model is None:
            self._model = build_tabpfn_surrogate(
                n_objectives=self.n_objectives,
                bin_edges=self._bins.edges,
                tabpfn_device=self.tabpfn_device,
                use_many_class_extension=self.use_many_class_extension,
                random_state=self.random_state,
                n_estimators=self.n_estimators,
                debug=self.debug,
            )
        else:
            for objective in self._model.objectives:
                objective.debug = bool(self.debug)
        self._model.fit(x_norm, y_norm)
        t_model_fit = time.perf_counter()
        if self.debug:
            print(
                f"[TabPFN fit] rows={x_arr.shape[0]} dim={x_arr.shape[1]} obj={y_arr.shape[1]} "
                f"| norm_stats={t_norm_stats - t0:.4f}s "
                f"| norm_apply={t_norm_apply - t_norm_stats:.4f}s "
                f"| bins={t_bins - t_norm_apply:.4f}s "
                f"| model_fit={t_model_fit - t_bins:.4f}s "
                f"| total={t_model_fit - t0:.4f}s"
            )
        return self

    def predict_mean_std(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise RuntimeError("TabPFNMinMaxSurrogate is not fit yet.")
        x_norm = self._norm_x(np.asarray(x, dtype=np.float32))
        mean_norm, std_norm = self._model.predict_mean_std(x_norm)
        return self._unnorm_y_mean_std(mean_norm, std_norm)

    def predict(self, x: np.ndarray) -> np.ndarray:
        mean, _ = self.predict_mean_std(x)
        return mean

    def predict_mean(self, x: np.ndarray, device: str | None = None) -> np.ndarray:
        del device
        return self.predict(x)

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        _, std = self.predict_mean_std(x)
        return std

    def refit(self, x: np.ndarray, y: np.ndarray) -> "TabPFNMinMaxSurrogate":
        return self.fit(x, y)

    @property
    def n_train_samples(self) -> int:
        if self._n_train_samples is None:
            raise RuntimeError("TabPFNMinMaxSurrogate is not fit yet.")
        return int(self._n_train_samples)

    @property
    def n_input_features(self) -> int:
        if self._x_min is None:
            raise RuntimeError("TabPFNMinMaxSurrogate is not fit yet (missing x stats).")
        return int(self._x_min.shape[0])

    @property
    def multi_context_signature(self) -> tuple[int, int, int]:
        return (self.n_input_features, self.n_train_samples)

    @staticmethod
    def predict_multi_context(
        surrogates: Sequence["TabPFNMinMaxSurrogate"],
        queries: Sequence[np.ndarray],
        *,
        return_std: bool = False,
        num_threads: int = 12,
    ) -> list[np.ndarray] | tuple[list[np.ndarray], list[np.ndarray]]:
        return predict_multi_context_tabpfn(surrogates, queries, return_std=return_std, num_threads=num_threads)


class _TabPFNMultiContextUnavailableError(RuntimeError):
    pass


def _get_tabpfn_ensemble_members(classifier: Any) -> list[Any]:
    visited_ids: set[int] = set()
    queue = [getattr(classifier, "executor_", None), classifier]

    while queue:
        current = queue.pop(0)
        if current is None:
            continue
        current_id = id(current)
        if current_id in visited_ids:
            continue
        visited_ids.add(current_id)

        members = getattr(current, "ensemble_members", None)
        if isinstance(members, list) and len(members) > 0:
            return members

        for attr_name in ("executor_", "executor", "engine", "inference_engine", "wrapped_estimator_", "estimator"):
            if hasattr(current, attr_name):
                queue.append(getattr(current, attr_name))

    raise _TabPFNMultiContextUnavailableError(
        f"Could not locate TabPFN ensemble members for classifier type {type(classifier).__name__}."
    )


def _predict_multi_context_tabpfn_fallback(
    surrogates: Sequence[TabPFNMinMaxSurrogate],
    queries: Sequence[np.ndarray],
    *,
    return_std: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], list[np.ndarray]]:
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    for surrogate, query in zip(surrogates, queries):
        mean, std = surrogate.predict_mean_std(query)
        means.append(np.asarray(mean, dtype=np.float32))
        stds.append(np.asarray(std, dtype=np.float32))
    return (means, stds) if return_std else means


def _predict_proba_multi_context_batched_public(
    objectives: Sequence[TabPFNObjectiveSurrogate],
    queries: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], dict[str, float]]:
    from inspect import signature

    def _debug_print_batch(
        *,
        enabled: bool,
        label: str,
        device: Any,
        x_shape: tuple[int, ...],
        y_shape: tuple[int, ...],
        local_batch_size: int,
        query_rows: int,
        train_rows: int,
        feature_dim: int,
    ) -> None:
        if not enabled:
            return
        msg = (
            f"[TabPFN batch debug] {label} | device={device} | "
            f"x_full={x_shape} | y_full={y_shape} | "
            f"local_batch={int(local_batch_size)} | train_rows={int(train_rows)} | "
            f"query_rows={int(query_rows)} | feature_dim={int(feature_dim)}"
        )
        if str(device).startswith("cuda"):
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(device=device)
                msg += (
                    f" | vram_free_gb={float(free_bytes) / (1024**3):.3f} "
                    f"| vram_total_gb={float(total_bytes) / (1024**3):.3f}"
                )
            except Exception as exc:
                msg += f" | vram_info_error={type(exc).__name__}: {exc}"
        print(msg)

    def _logits_to_full_probs(
        *,
        logits: torch.Tensor,
        objective: TabPFNObjectiveSurrogate,
        classifier: Any,
        y_train_bins: np.ndarray,
        class_permutation: np.ndarray | None,
    ) -> np.ndarray:
        logits_local = logits
        if class_permutation is None:
            logits_local = logits_local[:, : classifier.n_classes_]
        else:
            class_permutation = np.asarray(class_permutation, dtype=np.int64)
            if len(class_permutation) != classifier.n_classes_:
                use_perm = np.arange(classifier.n_classes_, dtype=np.int64)
                use_perm[: len(class_permutation)] = class_permutation
            else:
                use_perm = class_permutation
            logits_local = logits_local[:, use_perm]

        fit_classes = objective._fit_classes
        probs_partial_np = _apply_balanced_softmax_probs(
            logits_local,
            np.asarray(y_train_bins, dtype=np.float32),
            n_classes_full=int(objective.bins.k),
            active_classes=fit_classes,
        )
        if fit_classes is None or probs_partial_np.shape[1] == objective.bins.k:
            return np.asarray(probs_partial_np, dtype=np.float32)

        probs_full = np.zeros((probs_partial_np.shape[0], objective.bins.k), dtype=np.float32)
        for col, cls_id in enumerate(fit_classes.tolist()):
            if 0 <= int(cls_id) < objective.bins.k and col < probs_partial_np.shape[1]:
                probs_full[:, int(cls_id)] = probs_partial_np[:, col]
        return probs_full.astype(np.float32)

    if len(objectives) != len(queries):
        raise ValueError(f"objectives and queries must have the same length, got {len(objectives)} and {len(queries)}.")
    if len(objectives) == 0:
        return [], {
            "query_transform_sec": 0.0,
            "batch_prepare_sec": 0.0,
            "gpu_forward_sec": 0.0,
            "postprocess_sec": 0.0,
        }

    group_prepare_started_at = time.perf_counter()
    request_states: list[dict[str, Any]] = []
    grouped_items: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    query_transform_sec = 0.0

    for batch_idx, (objective, query) in enumerate(zip(objectives, queries)):
        if objective._fit_x is None or objective._fit_y_bins is None:
            raise RuntimeError("TabPFNObjectiveSurrogate is not fit yet.")

        classifier = objective.model
        ensemble_members = list(_get_tabpfn_ensemble_members(classifier))
        n_estimators = int(len(ensemble_members))
        if n_estimators <= 0:
            raise RuntimeError("TabPFN multi-context batch did not expose any fitted ensemble members.")

        estimator_probs: list[np.ndarray | None] = [None] * n_estimators
        request_states.append(
            {
                "batch_idx": int(batch_idx),
                "objective": objective,
                "classifier": classifier,
                "n_estimators": n_estimators,
                "estimator_probs": estimator_probs,
            }
        )

        query_arr = np.asarray(query, dtype=np.float32)
        transform_started_at = time.perf_counter()
        transformed_queries = [
            np.asarray(member.transform_X_test(query_arr), dtype=np.float32) for member in ensemble_members
        ]
        query_transform_sec += time.perf_counter() - transform_started_at

        for est_idx, (member, transformed_query) in enumerate(zip(ensemble_members, transformed_queries)):
            x_train_member = np.asarray(member.X_train, dtype=np.float32)
            y_train_member = np.asarray(member.y_train, dtype=np.float32).reshape(-1)
            if x_train_member.ndim != 2 or transformed_query.ndim != 2:
                raise ValueError(
                    f"Expected 2D TabPFN member tensors, got train={x_train_member.shape}, query={transformed_query.shape}."
                )
            if x_train_member.shape[1] != transformed_query.shape[1]:
                raise ValueError(
                    "TabPFN multi-context batch expects train/query to share feature width after preprocessing, "
                    f"got {x_train_member.shape[1]} and {transformed_query.shape[1]}."
                )

            model_index = int(getattr(member.config, "_model_index", getattr(objective, "_fit_model_index", 0)))
            device = classifier.devices_[0]
            dtype = classifier.forced_inference_dtype_ if classifier.forced_inference_dtype_ is not None else torch.float32
            has_gpu_preprocessor = member.gpu_preprocessor is not None
            if has_gpu_preprocessor:
                group_key = (
                    "single_gpu_preprocessed_member",
                    int(model_index),
                    str(device),
                    str(dtype),
                    int(batch_idx),
                    int(est_idx),
                )
            else:
                group_key = (
                    "batched_member_group",
                    int(model_index),
                    str(device),
                    str(dtype),
                    int(x_train_member.shape[0]),
                    int(x_train_member.shape[1]),
                )

            grouped_items.setdefault(group_key, []).append(
                {
                    "batch_idx": int(batch_idx),
                    "estimator_idx": int(est_idx),
                    "objective": objective,
                    "classifier": classifier,
                    "member": member,
                    "model_index": int(model_index),
                    "device": device,
                    "dtype": dtype,
                    "x_train": x_train_member,
                    "y_train": y_train_member,
                    "query": transformed_query,
                    "has_gpu_preprocessor": bool(has_gpu_preprocessor),
                }
            )

    probs_by_batch: list[np.ndarray | None] = [None for _ in objectives]
    batch_prepare_sec = time.perf_counter() - group_prepare_started_at
    gpu_forward_sec = 0.0
    postprocess_sec = 0.0

    for group_items in grouped_items.values():
        ref_item = group_items[0]
        classifier = ref_item["classifier"]
        debug_enabled = bool(getattr(ref_item["objective"], "debug", False))
        if not hasattr(classifier, "model_") and not hasattr(classifier, "models_"):
            raise RuntimeError(f"TabPFN classifier does not expose model_ / models_ for batched public forward: {type(classifier).__name__}.")

        model_index = int(ref_item["model_index"])
        if hasattr(classifier, "models_"):
            core_model = classifier.models_[int(model_index)]
        else:
            core_model = classifier.model_

        device = ref_item["device"]
        core_model = core_model.to(device)
        dtype = ref_item["dtype"]

        kwargs = {}
        if "task_type" in signature(core_model.forward).parameters:
            kwargs["task_type"] = "multiclass"

        use_cuda_timing = str(device).startswith("cuda")
        if any(bool(item["has_gpu_preprocessor"]) for item in group_items):
            from tabpfn.inference import _maybe_run_gpu_preprocessing

            for item in group_items:
                fill_started_at = time.perf_counter()
                x_train = np.asarray(item["x_train"], dtype=np.float32)
                y_train = np.asarray(item["y_train"], dtype=np.float32).reshape(-1)
                query = np.asarray(item["query"], dtype=np.float32)
                x_full = torch.zeros(
                    (int(x_train.shape[0]) + int(query.shape[0]), 1, int(x_train.shape[1])),
                    dtype=dtype,
                    device=device,
                )
                y_full = torch.zeros((int(x_train.shape[0]), 1), dtype=dtype, device=device)
                x_full[: x_train.shape[0], 0, :] = torch.as_tensor(x_train, dtype=dtype, device=device)
                if query.shape[0] > 0:
                    x_full[x_train.shape[0] :, 0, :] = torch.as_tensor(query, dtype=dtype, device=device)
                y_full[:, 0] = torch.as_tensor(y_train, dtype=dtype, device=device)
                x_full = _maybe_run_gpu_preprocessing(
                    x_full,
                    item["member"].gpu_preprocessor,
                    num_train_rows=int(x_train.shape[0]),
                )
                _debug_print_batch(
                    enabled=debug_enabled,
                    label="gpu_preprocessed_member",
                    device=device,
                    x_shape=tuple(int(v) for v in x_full.shape),
                    y_shape=tuple(int(v) for v in y_full.shape),
                    local_batch_size=1,
                    query_rows=int(query.shape[0]),
                    train_rows=int(x_train.shape[0]),
                    feature_dim=int(x_train.shape[1]),
                )
                batch_prepare_sec += time.perf_counter() - fill_started_at

                if use_cuda_timing:
                    torch.cuda.synchronize(device=device)
                gpu_started_at = time.perf_counter()
                with torch.autocast(device_type="cuda", enabled=use_cuda_timing):
                    with torch.inference_mode():
                        output = core_model(
                            x_full,
                            y_full,
                            only_return_standard_out=True,
                            **kwargs,
                        )
                if use_cuda_timing:
                    torch.cuda.synchronize(device=device)
                gpu_forward_sec += time.perf_counter() - gpu_started_at

                if output.ndim != 3:
                    raise ValueError(f"Expected core TabPFN output with 3 dims, got shape={tuple(output.shape)}.")

                postprocess_started_at = time.perf_counter()
                logits = output[: query.shape[0], 0, :]
                probs_full = _logits_to_full_probs(
                    logits=logits,
                    objective=item["objective"],
                    classifier=item["classifier"],
                    y_train_bins=y_train,
                    class_permutation=getattr(item["member"].config, "class_permutation", None),
                )
                request_states[int(item["batch_idx"])]["estimator_probs"][int(item["estimator_idx"])] = probs_full.astype(np.float32)
                postprocess_sec += time.perf_counter() - postprocess_started_at
        else:
            train_rows = int(ref_item["x_train"].shape[0])
            feature_dim = int(ref_item["x_train"].shape[1])
            max_query_rows = max(int(np.asarray(item["query"]).shape[0]) for item in group_items)
            local_batch_size = int(len(group_items))

            x_full = torch.zeros((train_rows + max_query_rows, local_batch_size, feature_dim), dtype=dtype, device=device)
            y_full = torch.zeros((train_rows, local_batch_size), dtype=dtype, device=device)
            query_lengths: list[int] = []

            fill_started_at = time.perf_counter()
            for local_idx, item in enumerate(group_items):
                x_train = np.asarray(item["x_train"], dtype=np.float32)
                y_train = np.asarray(item["y_train"], dtype=np.float32).reshape(-1)
                query = np.asarray(item["query"], dtype=np.float32)
                if int(x_train.shape[0]) != train_rows or int(x_train.shape[1]) != feature_dim:
                    raise ValueError(
                        "TabPFN multi-context batched forward requires aligned train shapes within a batch group, "
                        f"got {(x_train.shape[0], x_train.shape[1])} vs {(train_rows, feature_dim)}."
                    )
                x_full[:train_rows, local_idx, :] = torch.as_tensor(x_train, dtype=dtype, device=device)
                if query.shape[0] > 0:
                    x_full[train_rows : train_rows + query.shape[0], local_idx, :] = torch.as_tensor(
                        query,
                        dtype=dtype,
                        device=device,
                    )
                y_full[:, local_idx] = torch.as_tensor(y_train, dtype=dtype, device=device)
                query_lengths.append(int(query.shape[0]))
            batch_prepare_sec += time.perf_counter() - fill_started_at
            _debug_print_batch(
                enabled=debug_enabled,
                label="batched_member_group",
                device=device,
                x_shape=tuple(int(v) for v in x_full.shape),
                y_shape=tuple(int(v) for v in y_full.shape),
                local_batch_size=int(local_batch_size),
                query_rows=int(max_query_rows),
                train_rows=int(train_rows),
                feature_dim=int(feature_dim),
            )

            if use_cuda_timing:
                torch.cuda.synchronize(device=device)
            gpu_started_at = time.perf_counter()
            with torch.autocast(device_type="cuda", enabled=use_cuda_timing):
                with torch.inference_mode():
                    output = core_model(
                        x_full,
                        y_full,
                        only_return_standard_out=True,
                        **kwargs,
                    )
            if use_cuda_timing:
                torch.cuda.synchronize(device=device)
            gpu_forward_sec += time.perf_counter() - gpu_started_at

            if output.ndim != 3:
                raise ValueError(f"Expected core TabPFN output with 3 dims, got shape={tuple(output.shape)}.")

            postprocess_started_at = time.perf_counter()
            for local_idx, item in enumerate(group_items):
                logits = output[: query_lengths[local_idx], local_idx, :]
                probs_full = _logits_to_full_probs(
                    logits=logits,
                    objective=item["objective"],
                    classifier=item["classifier"],
                    y_train_bins=np.asarray(item["y_train"], dtype=np.float32),
                    class_permutation=getattr(item["member"].config, "class_permutation", None),
                )
                request_states[int(item["batch_idx"])]["estimator_probs"][int(item["estimator_idx"])] = probs_full.astype(np.float32)
            postprocess_sec += time.perf_counter() - postprocess_started_at

    for state in request_states:
        estimator_probs = state["estimator_probs"]
        if any(prob is None for prob in estimator_probs):
            missing = [idx for idx, prob in enumerate(estimator_probs) if prob is None]
            raise RuntimeError(
                f"Missing custom TabPFN ensemble probabilities for request {state['batch_idx']} estimator indices {missing}."
            )
        probs_by_batch[int(state["batch_idx"])] = np.mean(
            np.stack([np.asarray(prob, dtype=np.float32) for prob in estimator_probs if prob is not None], axis=0),
            axis=0,
        ).astype(np.float32)

    if any(probs is None for probs in probs_by_batch):
        missing = [idx for idx, probs in enumerate(probs_by_batch) if probs is None]
        raise RuntimeError(f"Missing batched TabPFN probabilities for objective batch indices {missing}.")

    return [np.asarray(probs, dtype=np.float32) for probs in probs_by_batch if probs is not None], {
        "query_transform_sec": float(query_transform_sec),
        "batch_prepare_sec": float(batch_prepare_sec),
        "gpu_forward_sec": float(gpu_forward_sec),
        "postprocess_sec": float(postprocess_sec),
    }


def predict_multi_context_tabpfn(
    surrogates: Sequence[TabPFNMinMaxSurrogate],
    queries: Sequence[np.ndarray],
    *,
    return_std: bool = False,
    return_profile: bool = False,
    num_threads: int = 12,
) -> list[np.ndarray] | tuple[list[np.ndarray], list[np.ndarray]] | tuple[Any, dict[str, float]]:
    del num_threads
    if len(surrogates) != len(queries):
        raise ValueError(f"surrogates and queries must have the same length, got {len(surrogates)} and {len(queries)}.")
    if len(surrogates) == 0:
        empty_profile = {
            "query_transform_sec": 0.0,
            "batch_prepare_sec": 0.0,
            "gpu_forward_sec": 0.0,
            "postprocess_sec": 0.0,
            "fallback_used": 0.0,
            "fallback_reason": "",
        }
        empty_out = ([], []) if return_std else []
        return (empty_out, empty_profile) if return_profile else empty_out

    if len(surrogates) == 1:
        mean, std = surrogates[0].predict_mean_std(queries[0])
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        out = ([mean], [std]) if return_std else [mean]
        profile = {
            "query_transform_sec": 0.0,
            "batch_prepare_sec": 0.0,
            "gpu_forward_sec": 0.0,
            "postprocess_sec": 0.0,
            "fallback_used": 0.0,
            "fallback_reason": "",
        }
        return (out, profile) if return_profile else out
    try:
        train_sizes = [int(surrogate.n_train_samples) for surrogate in surrogates]
        if len(set(train_sizes)) != 1:
            raise ValueError(f"All TabPFN multi-context surrogates must have the same number of train samples, got {train_sizes}.")

        normalize_started_at = time.perf_counter()
        normalized_queries = [surrogate._norm_x(np.asarray(query, dtype=np.float32)) for surrogate, query in zip(surrogates, queries)]
        query_transform_sec = time.perf_counter() - normalize_started_at
        n_contexts = int(len(surrogates))
        max_objectives = max(int(surrogate.n_objectives) for surrogate in surrogates)
        objective_mask = np.zeros((n_contexts, max_objectives), dtype=bool)
        for context_idx, surrogate in enumerate(surrogates):
            objective_mask[context_idx, : int(surrogate.n_objectives)] = True

        means_by_context: list[list[np.ndarray]] = [[] for _ in range(n_contexts)]
        stds_by_context: list[list[np.ndarray]] = [[] for _ in range(n_contexts)]

        flat_context_ids: list[int] = []
        flat_objective_ids: list[int] = []
        objective_tasks: list[TabPFNObjectiveSurrogate] = []
        objective_queries: list[np.ndarray] = []
        for context_idx in range(n_contexts):
            for objective_idx in range(max_objectives):
                if not objective_mask[context_idx, objective_idx]:
                    continue
                flat_context_ids.append(int(context_idx))
                flat_objective_ids.append(int(objective_idx))
                objective_tasks.append(surrogates[context_idx]._model.objectives[objective_idx])  # type: ignore[union-attr]
                objective_queries.append(normalized_queries[context_idx])

        probs_by_task, timing = _predict_proba_multi_context_batched_public(objective_tasks, objective_queries)
        timing["query_transform_sec"] += float(query_transform_sec)

        outer_postprocess_started_at = time.perf_counter()
        for task_idx, probs in enumerate(probs_by_task):
            context_idx = int(flat_context_ids[task_idx])
            objective_idx = int(flat_objective_ids[task_idx])
            objective = objective_tasks[task_idx]
            mean_norm, std_norm = tabpfn_probs_to_mean_std(probs, objective.bins, normalize=True)
            surrogate = surrogates[context_idx]
            y_min = float(surrogate._y_min[objective_idx])  # type: ignore[index]
            y_rng = float(surrogate._y_rng[objective_idx])  # type: ignore[index]
            means_by_context[context_idx].append((y_min + mean_norm * y_rng).astype(np.float32))
            stds_by_context[context_idx].append((std_norm * y_rng).astype(np.float32))
        timing["postprocess_sec"] += time.perf_counter() - outer_postprocess_started_at

        mean_outputs = [np.stack(parts, axis=1).astype(np.float32) for parts in means_by_context]
        std_outputs = [np.stack(parts, axis=1).astype(np.float32) for parts in stds_by_context]
        timing["fallback_used"] = 0.0
        timing["fallback_reason"] = ""
        out = (mean_outputs, std_outputs) if return_std else mean_outputs
        return (out, timing) if return_profile else out
    except Exception as exc:
        out = _predict_multi_context_tabpfn_fallback(surrogates, queries, return_std=return_std)
        timing = {
            "query_transform_sec": 0.0,
            "batch_prepare_sec": 0.0,
            "gpu_forward_sec": 0.0,
            "postprocess_sec": 0.0,
            "fallback_used": 1.0,
            "fallback_reason": f"{type(exc).__name__}: {exc}",
        }
        return (out, timing) if return_profile else out


def predict_multi_context(
    surrogates: Sequence[TabPFNMinMaxSurrogate],
    queries: Sequence[np.ndarray],
    *,
    return_std: bool = False,
    return_profile: bool = False,
    num_threads: int = 12,
) -> list[np.ndarray] | tuple[list[np.ndarray], list[np.ndarray]] | tuple[Any, dict[str, float]]:
    return predict_multi_context_tabpfn(
        surrogates,
        queries,
        return_std=return_std,
        return_profile=return_profile,
        num_threads=num_threads,
    )


from surrogate.gp import GPSurrogateModel, USEMO_GP_CONFIG, fit_gp_surrogates, predict_with_gp_mean, predict_with_gp_std


# Backwards/ergonomic aliases (requested names)
surrogate_model = Surrogate
SurrogateModel = Surrogate
gp = GPSurrogateModel
kan = KANSurrogateModel
tabpfn = TabPFNSurrogateModel
