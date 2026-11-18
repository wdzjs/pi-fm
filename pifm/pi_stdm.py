"""
PI-STDM: Physics-Informed Spatio-Temporal Diffusion Model.

Combines:
1. physics-informed attention (Y ⊙ QK^T), following Wu et al. (2025), used in the denoiser
2. STDM (Hua 2026) conditional diffusion framework

Novel contribution: First model to fuse physics-informed GNN with diffusion
for distribution system state estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path


# ========== Building blocks ==========

class PhysicsInformedAttention(nn.Module):
    """Multi-head attention with Ybus Hadamard product. From Wu et al. (2025)."""

    def __init__(self, d_model=64, n_heads=4, use_ybus=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = np.sqrt(self.d_k)
        self.use_ybus = use_ybus

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, ybus):
        N = x.shape[0]
        Q = self.W_q(x).view(N, self.n_heads, self.d_k)
        K = self.W_k(x).view(N, self.n_heads, self.d_k)
        V = self.W_v(x).view(N, self.n_heads, self.d_k)

        attn = torch.einsum('ihd,jhd->ijh', Q, K) / self.scale

        # === Physics-informed (Eq. 17): Y ⊙ (QK^T/√dk) ===
        if self.use_ybus and ybus is not None and ybus.shape[0] == N:
            attn = attn * ybus.unsqueeze(-1)

        attn_w = F.softmax(attn, dim=1)
        out = torch.einsum('ijh,jhd->ihd', attn_w, V)
        return self.W_o(out.reshape(N, self.d_model))


class PIDenoiser(nn.Module):
    """
    Physics-Informed Denoiser for diffusion.

    Architecture:
    1. Embedding: input → d_model
    2. Time embedding computed once
    3. N×PI-Attention blocks, each with per-block time FiLM
    4. Output projection: d_model → node_dim

    Key fix: time embedding is injected into EACH PIBlock, not just the end.
    """

    def __init__(self, node_dim=4, cond_dim=4, d_model=32, n_heads=2, n_blocks=3, use_ybus=True):
        super().__init__()
        self.d_model = d_model
        self.use_ybus = use_ybus

        # Input embedding
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim + cond_dim, d_model),
            nn.ReLU(),
        )

        # Time embedding: scalar t → d_model (shared across blocks)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # PI-Attention blocks (each has its own time FiLM projection)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(PIBlock(d_model, n_heads, use_ybus=use_ybus))

        # Output
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, node_dim),
        )

    def forward(self, x_t, t, cond, edge_index=None, edge_attr=None):
        N = x_t.shape[0]

        # Build Ybus matrix from edge_attr (with symmetrization)
        ybus = None
        if edge_index is not None and edge_attr is not None:
            ybus = torch.zeros(N, N, device=x_t.device)
            row, col = edge_index
            ybus[row, col] = edge_attr.squeeze(-1)
            ybus = (ybus + ybus.t()) / 2  # symmetrize

        # Compute time embedding ONCE (shared across blocks)
        t_emb = self.time_mlp(torch.tensor([t], dtype=torch.float32, device=x_t.device))
        # t_emb: (1, d_model) → each PIBlock projects its own FiLM params

        # Concatenate and project input
        h = torch.cat([x_t, cond], dim=-1)  # (N, node_dim + cond_dim)
        h = self.input_proj(h)  # (N, d_model)

        # PI-Attention blocks — each receives time embedding
        for block in self.blocks:
            h = block(h, ybus, t_emb)

        return self.output(h)


class PIBlock(nn.Module):
    """PI-Attention block with residual connection and per-block time FiLM."""

    def __init__(self, d_model, n_heads, dropout=0.1, use_ybus=True):
        super().__init__()
        self.attn = PhysicsInformedAttention(d_model, n_heads, use_ybus=use_ybus)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Per-block time FiLM projection: d_model → 2*d_model
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model * 2),
        )

    def forward(self, x, ybus, t_emb):
        # Time conditioning FIRST (before attention):
        # Project time embedding to per-node scale+shift
        ts = self.time_proj(t_emb)  # (1, 2*d_model)
        scale, shift = ts.chunk(2, dim=-1)  # (1, d_model) each
        x = x * (1 + scale) + shift  # per-node FiLM

        # Attention with physics
        h = self.norm1(x + self.attn(x, ybus))
        return self.norm2(h + self.ffn(h))


# ========== Diffusion Wrapper ==========

class PISTDM:
    """PI-STDM: Physics-Informed Spatio-Temporal Diffusion Model."""

    def __init__(self, node_dim=4, cond_dim=4, d_model=32, n_heads=2,
                 n_blocks=3, timesteps=50, device='cpu', use_ybus=True):
        self.node_dim = node_dim
        self.timesteps = timesteps
        self.device = device
        self.use_ybus = use_ybus

        self.denoiser = PIDenoiser(
            node_dim=node_dim, cond_dim=cond_dim,
            d_model=d_model, n_heads=n_heads, n_blocks=n_blocks,
            use_ybus=use_ybus,
        ).to(device)

        # Cosine schedule
        betas = self._cosine_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus = torch.sqrt(1.0 - alphas_cumprod).to(device)

    @staticmethod
    def _cosine_schedule(T, s=0.008):
        steps = T + 1
        x = torch.linspace(0, T, steps)
        ac = torch.cos(((x / T) + s) / (1 + s) * np.pi * 0.5) ** 2
        ac = ac / ac[0]
        return torch.clip(1 - (ac[1:] / ac[:-1]), 0.0001, 0.999)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        return (self.sqrt_alphas_cumprod[t] * x_0 +
                self.sqrt_one_minus[t] * noise), noise

    def set_normalizer(self, norm):
        self.norm = norm

    def p_loss(self, x_0, cond, edge_index, edge_attr):
        # Normalize
        if hasattr(self, 'norm'):
            x_0 = self.norm.normalize_y(x_0)
            cond = self.norm.normalize_x(cond)
        t = torch.randint(0, self.timesteps, (1,), device=self.device).item()
        noise = torch.randn_like(x_0)
        x_t, _ = self.q_sample(x_0, t, noise)
        eps_pred = self.denoiser(x_t, t, cond, edge_index, edge_attr)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, cond, edge_index, edge_attr, progress=False):
        if hasattr(self, 'norm'):
            cond = self.norm.normalize_x(cond)
        N = cond.shape[0]
        x_t = torch.randn(N, self.node_dim, device=self.device)
        iterator = range(self.timesteps - 1, -1, -1)
        if progress:
            iterator = tqdm(iterator, desc='PI-STDM Sample')
        for t in iterator:
            eps = self.denoiser(x_t, t, cond, edge_index, edge_attr)
            a, ac, b = self.alphas[t], self.alphas_cumprod[t], self.betas[t]
            n = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
            x_t = (1 / torch.sqrt(a)) * (x_t - (b / torch.sqrt(1 - ac + 1e-8)) * eps) + torch.sqrt(b) * n
        if hasattr(self, 'norm'):
            x_t = self.norm.denormalize_y(x_t)
        return x_t

    def train_step(self, data, opt):
        self.denoiser.train()
        opt.zero_grad()
        loss = self.p_loss(data.y, data.x, data.edge_index, data.edge_attr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
        opt.step()
        return loss.item()

    @torch.no_grad()
    def sample_ddim(self, cond, edge_index, edge_attr, steps=None, progress=False):
        """
        DDIM deterministic sampling (eta=0).

        Ref: Song et al. (2021) "Denoising Diffusion Implicit Models"

        Formula (deterministic, eta=0):
            x̂_0 = (x_t - √(1-ᾱ_t) * ε_pred) / √(ᾱ_t)
            x_{t-1} = √(ᾱ_{t-1}) * x̂_0 + √(1-ᾱ_{t-1}) * ε_pred

        Args:
            steps: subsequence size (e.g., 10). None uses self.timesteps.
        """
        if hasattr(self, 'norm'):
            cond = self.norm.normalize_x(cond)

        N = cond.shape[0]
        x_t = torch.randn(N, self.node_dim, device=self.device)

        # Build time subsequence (evenly spaced, from T-1 down to 0)
        if steps is None or steps >= self.timesteps:
            times = list(range(self.timesteps - 1, -1, -1))
        else:
            indices = torch.linspace(self.timesteps - 1, 0, steps).long()
            times = indices.tolist()

        iterator = times if not progress else tqdm(times, desc=f'PI-DDIM ({len(times)} steps)')

        for i, t in enumerate(iterator):
            eps = self.denoiser(x_t, t, cond, edge_index, edge_attr)

            # Estimate x̂_0
            ac_t = self.alphas_cumprod[t]            # ᾱ_t
            sqrt_ac_t = torch.sqrt(ac_t)
            sqrt_one_minus_ac_t = torch.sqrt(1.0 - ac_t)

            x0_pred = (x_t - sqrt_one_minus_ac_t * eps) / (sqrt_ac_t + 1e-8)

            # Clamp x̂_0 to reasonable range (normalized space)
            x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

            if i < len(times) - 1:
                t_prev = times[i + 1]
                ac_t_prev = self.alphas_cumprod[t_prev]
                sqrt_ac_t_prev = torch.sqrt(ac_t_prev)
                sqrt_one_minus_ac_t_prev = torch.sqrt(1.0 - ac_t_prev)
                x_t = sqrt_ac_t_prev * x0_pred + sqrt_one_minus_ac_t_prev * eps
            else:
                x_t = x0_pred  # final: x_0 estimate

        if hasattr(self, 'norm'):
            x_t = self.norm.denormalize_y(x_t)
        return x_t

    @torch.no_grad()
    def predict(self, data):
        self.denoiser.eval()
        return self.sample(data.x, data.edge_index, data.edge_attr)


def train_pistdm(model, train_d, val_d, epochs=200, lr=1e-3, patience=50):
    opt = torch.optim.Adam(model.denoiser.parameters(), lr=lr, weight_decay=1e-12)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=15, min_lr=1e-6)
    best, best_state, pc = float('inf'), None, 0

    for ep in tqdm(range(epochs), desc='PI-STDM'):
        tl = sum(model.train_step(d.to(model.device), opt) for d in train_d) / len(train_d)
        vl = sum(model.p_loss(d.y.to(model.device), d.x.to(model.device),
                              d.edge_index.to(model.device), d.edge_attr.to(model.device)).item()
                 for d in val_d) / len(val_d)
        sch.step(vl)
        if ep % 10 == 0 or vl < best:
            print(f"  ep {ep}: train_loss={tl:.6f} val_loss={vl:.6f}", flush=True)
        if vl < best:
            best, best_state, pc = vl, {k: v.cpu().clone() for k, v in model.denoiser.state_dict().items()}, 0
        else:
            pc += 1
        if pc >= patience:
            print(f"  Early stop at epoch {ep}", flush=True)
            break
    if best_state:
        model.denoiser.load_state_dict(best_state)
    return model


@torch.no_grad()
def eval_pistdm(model, test_d):
    mae, rmse = 0.0, 0.0
    for d in tqdm(test_d, desc='Eval'):
        d = d.to(model.device)
        pred = model.predict(d)
        mae += F.l1_loss(pred, d.y).item()
        rmse += F.mse_loss(pred, d.y).item()
    n = len(test_d)
    return {'mae': mae / n, 'rmse': np.sqrt(rmse / n)}
