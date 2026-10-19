r"""
Conditional Optimal-Transport Flow Matching for Synthetic Trajectory Generation (`FlowLDM`).

============================================================================
RESEARCH & ARCHITECTURAL ANCHORS
============================================================================
Enterprise stress-testing (CCAR / DFAST), minority-class fraud augmentation,
and cross-team data sharing require synthesizing realistic behavioral-tabular
representations without exposing raw customer PII.

Synthesizes two foundational generative modeling paradigms:
  1. Optimal Transport Conditional Flow Matching (`OT-CFM`, Lipman et al.,
     ICLR 2023, "Flow Matching for Generative Modeling", arXiv:2210.02747):
     - Defines straight probability paths between a standard normal prior
       $z_0 \sim \mathcal{N}(0, I)$ and target joint representations
       $z_1 = [h_{\text{seq}} \parallel x_{\text{tab}}] \in \mathbb{R}^D$:
       $$\psi_t(z_0 \mid z_1) = (1 - (1 - \sigma_{\min}) t) z_0 + t z_1, \quad t \in [0, 1]$$
     - Trains a time- and class-conditioned velocity network $v_\theta(z_t, t, y)$
       to match the conditional vector field $u_t(z_t \mid z_1) = z_1 - (1 - \sigma_{\min}) z_0$.
  2. Conditional Tabular & Behavioral Generation (`TabDDPM`, Kotelnikov et al.,
     ICML 2023, "TabDDPM: Modelling Tabular Data with Diffusion Models",
     arXiv:2209.15421):
     - Conditions the vector field on discrete class / fraud-archetype embeddings
       and sinusoidal continuous-time $t$ encodings, supporting deterministic
       1st-order Euler and 2nd-order Heun ODE integration in 8–16 steps.
============================================================================
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalFlowTimeEmbedding(nn.Module):
    r"""
    Maps continuous flow time $t \in [0, 1]$ into a $D$-dimensional sinusoidal
    embedding vector.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        half_dim = max(embed_dim // 2, 1)
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half_dim, dtype=torch.float32) / max(half_dim, 1)
        )
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(half_dim * 2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Flow time tensor `[B]` or `[B, 1]` in `[0, 1]`.

        Returns:
            Time embedding `[B, embed_dim]`.
        """
        t_col = t.view(-1, 1).float()
        angles = t_col * self.freqs.unsqueeze(0) * 100.0
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.proj(emb)


class ConditionalTrajectoryFlowMatcher(nn.Module):
    r"""
    Optimal-Transport Conditional Flow Matching (`OT-CFM`) head over joint
    behavioral-tabular LDM representations $z \in \mathbb{R}^{D_{\text{joint}}}$.

    Args:
        latent_dim: Dimension of the joint representation $[h_{\text{seq}} \parallel x_{\text{tab}}]$.
        hidden_dim: Width of the residual velocity MLP layers (default: `128`).
        num_classes: Number of discrete conditioning classes (default: `2` for
            legitimate vs. fraud, plus `1` null token for classifier-free guidance).
        sigma_min: Minimum standard deviation at $t=1$ along the OT path (default: `1e-4`).
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        sigma_min: float = 1e-4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.sigma_min = sigma_min

        self.time_embed = SinusoidalFlowTimeEmbedding(hidden_dim)
        # +1 index reserved for unconditional / null class
        self.class_embed = nn.Embedding(num_classes + 1, hidden_dim)

        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.block1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def predict_velocity(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Evaluates the learned conditional vector field $v_\theta(z_t, t, y)$.

        Args:
            z_t: Interpolated state `[B, latent_dim]` at time `t`.
            t: Flow time `[B]` or `[B, 1]` in `[0, 1]`.
            labels: Optional class labels `[B]` in `{0, ..., num_classes - 1}`.

        Returns:
            Predicted velocity vector `[B, latent_dim]`.
        """
        batch_size = z_t.size(0)
        t_emb = self.time_embed(t)

        if labels is None:
            cls_idx = torch.full(
                (batch_size,),
                self.num_classes,
                dtype=torch.long,
                device=z_t.device,
            )
        else:
            cls_idx = labels.view(-1).long().clamp(0, self.num_classes)
        c_emb = self.class_embed(cls_idx)

        cond = t_emb + c_emb
        h = self.input_proj(z_t) + cond
        h = h + self.block1(h) + cond
        h = h + self.block2(h)
        return self.velocity_head(h)

    def compute_loss(
        self,
        z_target: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""
        Computes the Optimal-Transport Conditional Flow Matching (`OT-CFM`)
        mean-squared velocity error over random $t \sim \mathcal{U}(0, 1)$.

        Args:
            z_target: Ground-truth joint representations $z_1$ `[B, latent_dim]`.
            labels: Optional class conditioning labels `[B]`.

        Returns:
            Dictionary with `"loss"` and `"velocity_mse"`.
        """
        batch_size = z_target.size(0)
        device = z_target.device
        dtype = z_target.dtype

        z_0 = torch.randn_like(z_target)
        t = torch.rand(batch_size, 1, device=device, dtype=dtype)

        # Linear OT interpolant: z_t = (1 - (1 - sigma_min) * t) * z_0 + t * z_1
        coeff_0 = 1.0 - (1.0 - self.sigma_min) * t
        z_t = coeff_0 * z_0 + t * z_target

        # Exact conditional target velocity: u_t = z_1 - (1 - sigma_min) * z_0
        target_velocity = z_target - (1.0 - self.sigma_min) * z_0
        pred_velocity = self.predict_velocity(z_t, t.squeeze(-1), labels=labels)

        loss = F.mse_loss(pred_velocity, target_velocity)
        return {
            "loss": loss,
            "velocity_mse": loss.detach(),
        }

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        target_class: Optional[int] = 1,
        num_steps: int = 16,
        solver: str = "heun",
        guidance_scale: float = 1.5,
        seed: Optional[int] = 42,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        r"""
        Synthesizes joint behavioral-tabular representations by integrating the
        learned ODE $\frac{dz_t}{dt} = v_\theta(z_t, t, y)$ from $t=0$ to $t=1$.

        Supports both 1st-order `"euler"` and 2nd-order `"heun"` Runge-Kutta
        solvers with optional classifier-free guidance (`guidance_scale`).

        Args:
            num_samples: Number of synthetic trajectories to generate.
            target_class: Target class index (e.g., `1` for synthetic fraud, `0` for normal).
            num_steps: Number of ODE integration steps (default: `16`).
            solver: `"euler"` or `"heun"` (default: `"heun"`).
            guidance_scale: Classifier-free guidance weight $\ge 1.0$ (default: `1.5`).
            seed: Optional random seed for deterministic synthesis.
            device: Target torch device.

        Returns:
            Synthesized joint representations $z_1$ `[num_samples, latent_dim]`.
        """
        if device is None:
            device = next(self.parameters()).device

        if seed is not None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)
            z = torch.randn((num_samples, self.latent_dim), generator=gen).to(device)
        else:
            z = torch.randn((num_samples, self.latent_dim), device=device)

        if target_class is not None:
            cond_labels = torch.full(
                (num_samples,), int(target_class), dtype=torch.long, device=device
            )
        else:
            cond_labels = None

        dt = 1.0 / float(max(num_steps, 1))
        time_grid = torch.linspace(0.0, 1.0 - dt, max(num_steps, 1), device=device)

        def _guided_velocity(z_curr: torch.Tensor, t_scalar: float) -> torch.Tensor:
            t_vec = torch.full((num_samples,), t_scalar, device=device, dtype=z_curr.dtype)
            v_cond = self.predict_velocity(z_curr, t_vec, labels=cond_labels)
            if cond_labels is not None and abs(guidance_scale - 1.0) > 1e-5:
                v_uncond = self.predict_velocity(z_curr, t_vec, labels=None)
                return v_uncond + guidance_scale * (v_cond - v_uncond)
            return v_cond

        for t_val in time_grid:
            t_f = float(t_val.item())
            v1 = _guided_velocity(z, t_f)
            if solver.lower() == "heun":
                z_euler = z + dt * v1
                v2 = _guided_velocity(z_euler, min(t_f + dt, 1.0))
                z = z + 0.5 * dt * (v1 + v2)
            else:
                z = z + dt * v1

        return z
