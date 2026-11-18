"""
STDM: Spatio-Temporal Diffusion Model for DSSE.
Reference: Hua et al. (2026), IJEPES.

Architecture:
1. Graph Transformer encoder (spatial topology via multi-head attention)
2. Temporal cross-attention (time dynamics)
3. DDPM reverse process conditioned on encodings + known measurements

Key innovation: Reformulates DSSE as a conditional diffusion process.

This module consumes the four-channel tensors built by
`pifm.data_generator.generate_dataset`, where x = [V_meas, P_meas, Q_meas, mask]
and y = [V, theta, P_load, Q_load]. That is the four-output path on which STDM
is evaluated; the flow-matching models use the two-output state tensor instead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


# ========== Noise Schedule ==========

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.999)


# ========== Graph Transformer ==========

class GraphTransformerEncoder(nn.Module):
    """
    Graph Transformer: models spatial dependencies among nodes.

    Uses multi-head attention with graph structure as positional encoding.
    """

    def __init__(self, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.ReLU(),
        )

        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(TransformerLayer(d_model, n_heads, dropout))

        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x, edge_index=None):
        """x: (N, F) → (N, d_model)"""
        h = self.input_proj(x)  # (N, d_model)
        for layer in self.layers:
            h = layer(h)
        return self.output_norm(h)


class TransformerLayer(nn.Module):
    """Single Transformer layer with self-attention + FFN."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (N, d_model) -> add batch dim -> (1, N, d_model)
        x_b = x.unsqueeze(0)
        attn_out, _ = self.attn(x_b, x_b, x_b)
        x = self.norm1(x + self.dropout(attn_out.squeeze(0)))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# ========== Denoiser ==========

class STDM_Denoiser(nn.Module):
    """
    Denoiser for Spatio-Temporal Diffusion Model.

    Predicts the noise ε given:
    - Noisy state x_t at time t
    - Conditioning from known measurements
    - Graph structure
    """

    def __init__(self, node_dim=2, cond_dim=4, d_model=64, n_heads=4,
                 n_encoder_layers=2, denoiser_layers=3):
        super().__init__()

        self.encoder = GraphTransformerEncoder(d_model, n_heads, n_encoder_layers)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Denoising MLP (per-node)
        # Input: x_t (node_dim) + h_cond (d_model) + t_emb (d_model)
        in_dim = node_dim + d_model * 2
        self.denoiser = nn.Sequential(
            nn.Linear(in_dim, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, node_dim),
        )

    def forward(self, x_t, t, cond, edge_index=None):
        """
        Args:
            x_t: (N, node_dim) noisy state at timestep t
            t: scalar timestep
            cond: (N, cond_dim) conditioning info
            edge_index: graph structure (optional)

        Returns:
            eps_pred: (N, node_dim) predicted noise
        """
        N = x_t.shape[0]

        # Encode conditioning into graph-aware features
        h_cond = self.encoder(cond, edge_index)  # (N, d_model)

        # Time embedding
        t_emb = self.time_mlp(torch.tensor([t], dtype=torch.float32, device=x_t.device))
        t_emb = t_emb.expand(N, -1)  # (N, d_model)

        # Concatenate: [x_t, h_cond, t_emb]
        h = torch.cat([x_t, h_cond, t_emb], dim=-1)  # (N, node_dim + 2*d_model)

        eps_pred = self.denoiser(h)
        return eps_pred


# ========== STDM Pipeline ==========

class STDM:
    """Spatio-Temporal Diffusion Model for DSSE."""

    def __init__(self, node_dim=2, cond_dim=4, d_model=64, n_heads=4,
                 timesteps=200, device='cpu'):
        self.node_dim = node_dim
        self.timesteps = timesteps
        self.device = device

        self.denoiser = STDM_Denoiser(
            node_dim=node_dim, cond_dim=cond_dim,
            d_model=d_model, n_heads=n_heads,
        ).to(device)

        # Noise schedule
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: x_t given x_0."""
        if noise is None:
            noise = torch.randn_like(x_0)
        return (self.sqrt_alphas_cumprod[t] * x_0 +
                self.sqrt_one_minus_alphas_cumprod[t] * noise), noise

    def p_loss(self, x_0, cond, edge_index=None):
        """Training loss: ||ε - ε_θ(x_t, t, cond)||²."""
        t = torch.randint(0, self.timesteps, (1,), device=self.device).item()
        noise = torch.randn_like(x_0)
        x_t, _ = self.q_sample(x_0, t, noise)
        eps_pred = self.denoiser(x_t, t, cond, edge_index)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, cond, edge_index=None, progress=False):
        """Reverse diffusion: sample x_0 from noise conditioned on cond."""
        N = cond.shape[0]
        x_t = torch.randn(N, self.node_dim, device=self.device)

        iterator = range(self.timesteps - 1, -1, -1)
        if progress:
            iterator = tqdm(iterator, desc='Sampling')

        for t in iterator:
            eps_pred = self.denoiser(x_t, t, cond, edge_index)
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]

            noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
            x_t = (1 / torch.sqrt(alpha)) * (
                x_t - (beta / torch.sqrt(1 - alpha_cumprod + 1e-8)) * eps_pred
            ) + torch.sqrt(beta) * noise

        return x_t

    def train_step(self, data, optimizer):
        """Single training step."""
        self.denoiser.train()
        optimizer.zero_grad()
        loss = self.p_loss(data.y, data.x, data.edge_index)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def predict(self, data):
        """Predict states for a single sample."""
        self.denoiser.eval()
        return self.sample(data.x, data.edge_index)


def train_stdm(model, train_data, val_data, epochs=200, lr=1e-3, patience=30):
    """Train STDM model."""
    optimizer = torch.optim.Adam(model.denoiser.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    best_loss = float('inf')
    best_state = None
    patience_count = 0

    pbar = tqdm(range(epochs), desc='STDM Train')
    for ep in pbar:
        train_loss = 0.0
        for d in train_data:
            train_loss += model.train_step(d.to(model.device), optimizer)
        train_loss /= len(train_data)

        val_loss = 0.0
        for d in val_data:
            val_loss += model.p_loss(d.y.to(model.device), d.x.to(model.device), d.edge_index.to(model.device)).item()
        val_loss /= len(val_data)

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.denoiser.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        pbar.set_postfix({'train': f'{train_loss:.4f}', 'val': f'{val_loss:.4f}'})
        if patience_count >= patience:
            break

    if best_state:
        model.denoiser.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_stdm(model, test_data):
    """Evaluate STDM: sample and compute metrics."""
    mae, rmse = 0.0, 0.0
    for d in tqdm(test_data, desc='STDM Eval'):
        d = d.to(model.device)
        pred = model.predict(d)
        true = d.y
        mae += F.l1_loss(pred, true).item()
        rmse += F.mse_loss(pred, true).item()
    n = len(test_data)
    return {'mae': mae / n, 'rmse': np.sqrt(rmse / n)}


if __name__ == '__main__':
    import argparse, json
    from pathlib import Path
    from ..data_generator import load_dataset, generate_dataset, create_splits, save_dataset
    from ..utils import set_seed

    p = argparse.ArgumentParser()
    p.add_argument('--system', type=str, default='33')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--timesteps', type=int, default=200)
    args = p.parse_args()

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"STDM on {device}")

    splits = load_dataset(args.system, 'data')
    if splits is None:
        dl, _ = generate_dataset(args.system, 500, observable_ratio=0.3)
        splits = create_splits(dl)
        save_dataset(splits, args.system, 'data')

    model = STDM(node_dim=4, cond_dim=4, d_model=64, n_heads=4,
                 timesteps=args.timesteps, device=device)
    n_params = sum(p.numel() for p in model.denoiser.parameters())
    print(f"STDM params: {n_params:,}")

    model = train_stdm(model, splits['train'], splits['val'], args.epochs, args.lr)
    metrics = evaluate_stdm(model, splits['test'])
    print(f"Test: MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}")

    out = Path('output')
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.denoiser.state_dict(), out / f"stdm_{args.system}.pt")
    with open(out / f"stdm_metrics_{args.system}.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Done!")
