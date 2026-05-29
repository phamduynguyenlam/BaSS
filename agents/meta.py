from __future__ import annotations

import torch
import torch.nn as nn

from agents.base import LandscapeEncoder


class DuelingActionDecoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 16, n_actions: int = 5, dropout: float = 0.0):
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_actions = int(n_actions)

        self.advantage_head = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.n_actions),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError(f"DuelingActionDecoder expects [B, D], got {tuple(z.shape)}.")
        advantage = self.advantage_head(z)
        value = self.value_head(z)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class MetaAgent(nn.Module):
    ACTION_STEPS = (5, 10, 20, 50, 100)
    ACTION_NAMES = tuple(f"surrogate_nsga_steps_{step}" for step in ACTION_STEPS)

    def __init__(
        self,
        hidden_dim: int = 64,
        ff_dim: int = 256,
        dropout: float = 0.0,
        logit_scale: float = 5.0,
        epsilon: float = 0.05,
        decoder_hidden_dim: int = 16,
        n_heads: int = 1,
        **_: object,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.ff_dim = int(ff_dim)
        self.dropout = float(dropout)
        self.logit_scale = float(logit_scale)
        self.epsilon = float(epsilon)

        self.W_true = nn.Linear(2, self.hidden_dim)
        self.W_prevsurr = nn.Linear(3, self.hidden_dim)
        self.encoder_true = LandscapeEncoder(self.hidden_dim, self.n_heads, self.ff_dim, self.dropout)
        self.encoder_prevsurr = LandscapeEncoder(self.hidden_dim, self.n_heads, self.ff_dim, self.dropout)
        self.q_decoder = DuelingActionDecoder(
            state_dim=2 * self.hidden_dim + 3,
            hidden_dim=int(decoder_hidden_dim),
            n_actions=len(self.ACTION_STEPS),
            dropout=self.dropout,
        )
        self.start_action_token = nn.Parameter(torch.zeros(1, 1))
        self.start_reward_token = nn.Parameter(torch.zeros(1, 1))

    @staticmethod
    def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0) if x.dim() == 2 else x

    @staticmethod
    def _normalize_by_range(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        denom = (upper - lower).clamp_min(1e-12)
        return ((x - lower) / denom).clamp(0.0, 1.0)

    @staticmethod
    def _min_max_normalize(x: torch.Tensor) -> torch.Tensor:
        x_min = x.amin(dim=1, keepdim=True)
        x_max = x.amax(dim=1, keepdim=True)
        return ((x - x_min) / (x_max - x_min).clamp_min(1e-12)).clamp(0.0, 1.0)

    @staticmethod
    def _prepare_progress(progress, device, dtype, batch_size: int) -> torch.Tensor:
        progress_tensor = torch.as_tensor(progress, device=device, dtype=dtype)
        if progress_tensor.dim() == 0:
            progress_tensor = progress_tensor.repeat(int(batch_size))
        progress_tensor = progress_tensor.reshape(int(batch_size), -1)
        if progress_tensor.size(1) != 1:
            progress_tensor = progress_tensor[:, :1]
        return progress_tensor.clamp(0.0, 1.0)

    @staticmethod
    def _prepare_scalar_token(value, device, dtype, batch_size: int, fallback: torch.Tensor) -> torch.Tensor:
        if value is None:
            return fallback.to(device=device, dtype=dtype).expand(int(batch_size), -1)

        value_tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if value_tensor.dim() == 0:
            value_tensor = value_tensor.repeat(int(batch_size))
        value_tensor = value_tensor.reshape(int(batch_size), -1)
        if value_tensor.size(1) != 1:
            value_tensor = value_tensor[:, :1]

        start_mask = torch.isnan(value_tensor)
        if start_mask.any():
            value_tensor = value_tensor.clone()
            value_tensor[start_mask] = fallback.to(device=device, dtype=dtype).view(1)
        return value_tensor

    def _prepare_inputs(
        self,
        x_true,
        y_true,
        x_sur,
        y_sur,
        sigma_sur,
        lower_bound,
        upper_bound,
        archive_mask=None,
        candidate_mask=None,
    ):
        x_true = self._ensure_batch(x_true).float()
        y_true = self._ensure_batch(y_true).float()
        x_sur = self._ensure_batch(x_sur).float()
        y_sur = self._ensure_batch(y_sur).float()
        sigma_sur = self._ensure_batch(sigma_sur).float()

        device = x_true.device
        dtype = x_true.dtype
        batch_size = x_true.size(0)
        n_dim = x_true.size(-1)
        n_archive = x_true.size(1)

        n_prev = x_sur.size(1)
        if y_sur.size(1) != n_prev or sigma_sur.size(1) != n_prev:
            raise ValueError("y_sur and sigma_sur must align with x_sur.")

        if archive_mask is not None:
            archive_mask = torch.as_tensor(archive_mask, device=device, dtype=torch.bool)
            if archive_mask.dim() == 1:
                archive_mask = archive_mask.view(1, -1).expand(batch_size, -1)
            if archive_mask.size(1) != n_archive:
                raise ValueError(f"archive_mask width mismatch: {archive_mask.size(1)} vs {n_archive}")
        if candidate_mask is not None:
            candidate_mask = torch.as_tensor(candidate_mask, device=device, dtype=torch.bool)
            if candidate_mask.dim() == 1:
                candidate_mask = candidate_mask.view(1, -1).expand(batch_size, -1)
            if candidate_mask.size(1) != n_prev:
                raise ValueError(f"candidate_mask width mismatch: {candidate_mask.size(1)} vs {n_prev}")

        lower = torch.as_tensor(lower_bound, device=device, dtype=dtype)
        upper = torch.as_tensor(upper_bound, device=device, dtype=dtype)

        if lower.dim() == 0:
            lower = lower.repeat(n_dim).view(1, n_dim).expand(batch_size, -1)
        elif lower.dim() == 1:
            lower = lower.view(1, -1).expand(batch_size, -1)
        if upper.dim() == 0:
            upper = upper.repeat(n_dim).view(1, n_dim).expand(batch_size, -1)
        elif upper.dim() == 1:
            upper = upper.view(1, -1).expand(batch_size, -1)

        dim_mask = (upper - lower).abs() > 1e-12
        lower = lower.unsqueeze(1)
        upper = upper.unsqueeze(1)

        x_true_norm = self._normalize_by_range(x_true, lower, upper)
        x_sur_norm = self._normalize_by_range(x_sur, lower, upper)
        y_true_norm = self._min_max_normalize(y_true)
        y_sur_norm = self._min_max_normalize(y_sur)
        sigma_sur_norm = self._min_max_normalize(sigma_sur)

        x_true_expand = x_true_norm.transpose(1, 2).unsqueeze(1).unsqueeze(-1)
        x_true_expand = x_true_expand.expand(-1, y_true_norm.size(-1), -1, -1, -1)
        y_true_expand = y_true_norm.transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        y_true_expand = y_true_expand.expand(-1, -1, x_true_norm.size(-1), -1, -1)
        e_true = torch.cat((x_true_expand, y_true_expand), dim=-1)

        x_sur_expand = x_sur_norm.transpose(1, 2).unsqueeze(1).unsqueeze(-1)
        x_sur_expand = x_sur_expand.expand(-1, y_sur_norm.size(-1), -1, -1, -1)
        y_sur_expand = y_sur_norm.transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        y_sur_expand = y_sur_expand.expand(-1, -1, x_sur_norm.size(-1), -1, -1)
        sigma_expand = sigma_sur_norm.transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        sigma_expand = sigma_expand.expand(-1, -1, x_sur_norm.size(-1), -1, -1)
        e_prevsurr = torch.cat((x_sur_expand, y_sur_expand, sigma_expand), dim=-1)

        if archive_mask is None:
            archive_mask = torch.ones(batch_size, n_archive, device=device, dtype=torch.bool)
        if candidate_mask is None:
            candidate_mask = torch.ones(batch_size, n_prev, device=device, dtype=torch.bool)

        return {
            "E_true": e_true,
            "E_prevsurr": e_prevsurr,
            "dim_mask": dim_mask,
            "archive_mask": archive_mask,
            "candidate_mask": candidate_mask,
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
        }

    @staticmethod
    def _masked_pool_over_individuals(h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return h.mean(dim=2)
        mask_f = mask.to(device=h.device, dtype=h.dtype).view(h.size(0), 1, h.size(2), 1)
        denom = mask_f.sum(dim=2).clamp_min(1.0)
        return (h * mask_f).sum(dim=2) / denom

    def encode(
        self,
        x_true,
        y_true,
        x_sur,
        y_sur,
        sigma_sur,
        progress,
        lower_bound,
        upper_bound,
        prev_action=None,
        prev_reward=None,
        archive_mask=None,
        candidate_mask=None,
    ):
        prepared = self._prepare_inputs(
            x_true=x_true,
            y_true=y_true,
            x_sur=x_sur,
            y_sur=y_sur,
            sigma_sur=sigma_sur,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            archive_mask=archive_mask,
            candidate_mask=candidate_mask,
        )

        h_true = self.encoder_true(
            self.W_true(prepared["E_true"]),
            dim_mask=prepared["dim_mask"],
            individual_mask=prepared["archive_mask"],
        )
        h_prevsurr = self.encoder_prevsurr(
            self.W_prevsurr(prepared["E_prevsurr"]),
            dim_mask=prepared["dim_mask"],
            individual_mask=prepared["candidate_mask"],
        )

        ela_current = self._masked_pool_over_individuals(h_true, prepared["archive_mask"]).mean(dim=1)
        ela_previous = self._masked_pool_over_individuals(h_prevsurr, prepared["candidate_mask"]).mean(dim=1)
        prev_action_state = self._prepare_scalar_token(
            value=prev_action,
            device=prepared["device"],
            dtype=prepared["dtype"],
            batch_size=prepared["batch_size"],
            fallback=self.start_action_token,
        )
        prev_reward_state = self._prepare_scalar_token(
            value=prev_reward,
            device=prepared["device"],
            dtype=prepared["dtype"],
            batch_size=prepared["batch_size"],
            fallback=self.start_reward_token,
        )
        progress_state = self._prepare_progress(
            progress=progress,
            device=prepared["device"],
            dtype=prepared["dtype"],
            batch_size=prepared["batch_size"],
        )

        z = torch.cat([ela_current, ela_previous, prev_action_state, prev_reward_state, progress_state], dim=-1)
        return {
            "H_true": h_true,
            "H_prevsurr": h_prevsurr,
            "ela": ela_current,
            "prev_surrogate_ela": ela_previous,
            "prev_action": prev_action_state,
            "prev_reward": prev_reward_state,
            "progress": progress_state,
            "z": z,
        }

    def _epsilon_greedy_action(self, q_values: torch.Tensor, epsilon: float) -> torch.Tensor:
        batch_size, n_actions = q_values.shape
        greedy = torch.argmax(q_values, dim=1)
        random_actions = torch.randint(0, n_actions, (batch_size,), device=q_values.device)
        use_random = torch.rand(batch_size, device=q_values.device) < float(epsilon)
        return torch.where(use_random, random_actions, greedy)

    def decode_action(self, z: torch.Tensor, decode_type: str = "epsilon_greedy", epsilon: float | None = None):
        q_values = self.q_decoder(z) * self.logit_scale
        if epsilon is None:
            epsilon = self.epsilon

        if decode_type in {"greedy", "q_greedy"}:
            action = torch.argmax(q_values, dim=1)
        elif decode_type == "softmax_sample":
            probs = torch.softmax(q_values, dim=-1)
            action = torch.distributions.Categorical(probs=probs).sample()
        elif decode_type == "epsilon_greedy":
            action = self._epsilon_greedy_action(q_values, float(epsilon))
        else:
            raise ValueError(f"Unknown decode_type: {decode_type}")

        return {
            "logits": q_values,
            "q_values": q_values,
            "action": action,
        }

    @classmethod
    def action_to_surrogate_nsga_steps(cls, action_idx: int) -> int:
        idx = int(action_idx)
        if idx < 0 or idx >= len(cls.ACTION_STEPS):
            raise ValueError(f"Invalid action_idx={action_idx}; expected [0, {len(cls.ACTION_STEPS) - 1}].")
        return int(cls.ACTION_STEPS[idx])

    def forward(
        self,
        x_true,
        y_true,
        x_sur,
        y_sur,
        sigma_sur,
        progress,
        lower_bound,
        upper_bound,
        prev_action=None,
        prev_reward=None,
        archive_mask=None,
        candidate_mask=None,
        decode_type="epsilon_greedy",
        epsilon=None,
    ):
        encoded = self.encode(
            x_true=x_true,
            y_true=y_true,
            x_sur=x_sur,
            y_sur=y_sur,
            sigma_sur=sigma_sur,
            progress=progress,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            prev_action=prev_action,
            prev_reward=prev_reward,
            archive_mask=archive_mask,
            candidate_mask=candidate_mask,
        )
        decoded = self.decode_action(
            z=encoded["z"],
            decode_type=decode_type,
            epsilon=epsilon,
        )
        encoded.update(decoded)
        return encoded


Disc = MetaAgent
DiscAF = MetaAgent
