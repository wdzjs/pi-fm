"""
PI-DDPM-Res: a denoising diffusion model with a residual target.

Architecture:
  1. DistFlow coarse estimate: y_phys = distflow(x)
  2. Residual target: r_true = y_true - y_phys
  3. Forward diffusion on the residual:
         r_t = sqrt(alpha_bar_t) * r_0 + sqrt(1 - alpha_bar_t) * noise
  4. PI denoiser (admittance-modulated attention blocks) predicts the noise
     from r_t and the measurements
  5. Reverse sampling denoises r_T -> r_pred, then y_pred = y_phys + r_pred

This is the stochastic counterpart of the flow-matching model: identical
residual formulation and denoiser, trained with the DDPM objective instead of
the flow-matching objective. Both samplers are implemented, the ancestral one
used in training and a deterministic DDIM variant.

Usage:
    python -m pifm.baselines.pi_ddpm_res --system 33 --epochs 200
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm

from ..data_generator import load_dataset, create_splits
from ..test_systems import get_test_system
from ..distflow_module import DistFlowCoarse
from ..utils import set_seed
from ..pi_stdm import PIDenoiser, PhysicsInformedAttention, PIBlock
from ..dataset_builder import generate_paper_dataset


# ============================================================
# PISTDMRes: DDPM + Residual Target
# ============================================================

class PISTDMRes:
    """
    PI-DDPM-Res: Physics-Informed DDPM with Residual Learning.

    Same denoiser architecture as the full-state DDPM baseline, but it
    operates on the residual r = y - y_phys instead of the full state y.

    DDPM process:
      Forward:  r_t = sqrt(a_bar_t) * r_0 + sqrt(1 - a_bar_t) * noise
      Reverse:  predict noise eps, then subtract to get r_{t-1}
      Output:   y_pred = y_phys + r_pred
    """

    def __init__(self, node_dim=2, cond_dim=4, d_model=32, n_heads=2,
                 n_blocks=3, timesteps=50, device='cpu'):
        self.node_dim = node_dim
        self.cond_dim = cond_dim
        self.timesteps = timesteps
        self.device = device

        self.denoiser = PIDenoiser(
            node_dim=node_dim, cond_dim=cond_dim,
            d_model=d_model, n_heads=n_heads, n_blocks=n_blocks,
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

    def _to_device(self, tensor):
        """Move tensor to device."""
        if tensor.device != self.device:
            return tensor.to(self.device)
        return tensor

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: add noise to clean data."""
        if noise is None:
            noise = torch.randn_like(x_0)
        return (self.sqrt_alphas_cumprod[t] * x_0 +
                self.sqrt_one_minus[t] * noise), noise

    def set_normalizer(self, norm):
        """Set data normalizer."""
        self.norm = norm

    # ----------------------------------------------------------
    # Residual computation (via DistFlow)
    # ----------------------------------------------------------

    def compute_residuals(self, x, y, edge_index, edge_attr, distflow):
        """
        Compute residual targets r_true = y - distflow_coarse(x).

        Args:
            x: (N, cond_dim) measurement vector on device.
            y: (N, node_dim) true state (Vm, Va) on device.
            edge_index: (2, E) graph connectivity.
            edge_attr: (E, 1) edge weights.
            distflow: DistFlowCoarse instance.

        Returns:
            r_true: (N, node_dim) residual tensor on same device as x.
            y_phys: (N, node_dim) coarse estimate on same device as x.
        """
        y_phys = distflow(x, edge_index, edge_attr)
        y_phys = self._to_device(y_phys)
        r_true = y - y_phys
        return r_true, y_phys

    # ----------------------------------------------------------
    # DDPM loss on residuals
    # ----------------------------------------------------------

    def p_loss(self, r_true, cond, edge_index, edge_attr):
        """
        DDPM noise prediction loss on residual.

        Instead of predicting noise on full state y, we predict noise
        on the residual r = y - y_phys. The residual has smaller magnitude
        and is easier for the denoiser to learn.
        """
        # Normalize
        if hasattr(self, 'norm'):
            r_true_norm = self.norm.normalize_r(r_true)
            cond_norm = self.norm.normalize_x(cond)
        else:
            r_true_norm = r_true
            cond_norm = cond

        # Sample timestep
        t = torch.randint(0, self.timesteps, (1,), device=self.device).item()
        noise = torch.randn_like(r_true_norm)
        r_t, _ = self.q_sample(r_true_norm, t, noise)

        # Predict noise from noisy residual
        eps_pred = self.denoiser(r_t, t, cond_norm, edge_index, edge_attr)
        return F.mse_loss(eps_pred, noise)

    # ----------------------------------------------------------
    # DDPM reverse sampling on residual
    # ----------------------------------------------------------

    @torch.no_grad()
    def sample(self, cond, edge_index, edge_attr, distflow, progress=False):
        """
        Generate state estimate via DDPM reverse process on residual.

        Steps:
          1. Compute DistFlow coarse estimate y_phys
          2. Start from random noise r_T
          3. Reverse-diffuse T -> 0 to get r_pred
          4. Denormalize r_pred and add y_phys to get y_pred
        """
        if hasattr(self, 'norm'):
            cond_norm = self.norm.normalize_x(cond)
        else:
            cond_norm = cond

        # DistFlow coarse estimate
        y_phys = distflow(cond, edge_index, edge_attr)
        y_phys = self._to_device(y_phys)

        N = cond.shape[0]
        r_t = torch.randn(N, self.node_dim, device=self.device)

        iterator = range(self.timesteps - 1, -1, -1)
        if progress:
            iterator = tqdm(iterator, desc='PI-DDPM-Res Sample')

        for t_idx in iterator:
            eps = self.denoiser(r_t, t_idx, cond_norm, edge_index, edge_attr)
            a = self.alphas[t_idx]
            ac = self.alphas_cumprod[t_idx]
            b = self.betas[t_idx]
            n = torch.randn_like(r_t) if t_idx > 0 else torch.zeros_like(r_t)
            r_t = (1 / torch.sqrt(a)) * (r_t - (b / torch.sqrt(1 - ac + 1e-8)) * eps) + torch.sqrt(b) * n

        # Denormalize residual
        if hasattr(self, 'norm'):
            r_pred = self.norm.denormalize_r(r_t)
        else:
            r_pred = r_t

        # Add back physics coarse estimate
        y_pred = y_phys + r_pred
        return y_pred

    # ----------------------------------------------------------
    # DDIM deterministic sampling on residual
    # ----------------------------------------------------------

    @torch.no_grad()
    def sample_ddim(self, cond, edge_index, edge_attr, distflow,
                    steps=None, progress=False):
        """
        DDIM deterministic sampling on residual (eta=0).

        DDIM update (eta=0) applied in the residual space.
        """
        if hasattr(self, 'norm'):
            cond_norm = self.norm.normalize_x(cond)
        else:
            cond_norm = cond

        y_phys = distflow(cond, edge_index, edge_attr)
        y_phys = self._to_device(y_phys)

        N = cond.shape[0]
        r_t = torch.randn(N, self.node_dim, device=self.device)

        # Build time subsequence
        if steps is None or steps >= self.timesteps:
            times = list(range(self.timesteps - 1, -1, -1))
        else:
            indices = torch.linspace(self.timesteps - 1, 0, steps).long()
            times = indices.tolist()

        iterator = times if not progress else tqdm(times, desc=f'PI-DDPM-Res DDIM ({len(times)} steps)')

        for i, t_idx in enumerate(iterator):
            eps = self.denoiser(r_t, t_idx, cond_norm, edge_index, edge_attr)

            # Estimate r_0
            ac_t = self.alphas_cumprod[t_idx]
            sqrt_ac_t = torch.sqrt(ac_t)
            sqrt_one_minus_ac_t = torch.sqrt(1.0 - ac_t)

            r0_pred = (r_t - sqrt_one_minus_ac_t * eps) / (sqrt_ac_t + 1e-8)
            r0_pred = torch.clamp(r0_pred, -3.0, 3.0)  # clamp in normalized space

            if i < len(times) - 1:
                t_prev = times[i + 1]
                ac_t_prev = self.alphas_cumprod[t_prev]
                sqrt_ac_t_prev = torch.sqrt(ac_t_prev)
                sqrt_one_minus_ac_t_prev = torch.sqrt(1.0 - ac_t_prev)
                r_t = sqrt_ac_t_prev * r0_pred + sqrt_one_minus_ac_t_prev * eps
            else:
                r_t = r0_pred  # final: r_0 estimate

        if hasattr(self, 'norm'):
            r_pred = self.norm.denormalize_r(r_t)
        else:
            r_pred = r_t

        y_pred = y_phys + r_pred
        return y_pred

    # ----------------------------------------------------------
    # Training step
    # ----------------------------------------------------------

    def train_step(self, data, opt, distflow):
        """Single training step."""
        self.denoiser.train()
        opt.zero_grad()

        x = data.x.to(self.device)
        y = data.y.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_attr = data.edge_attr.to(self.device)

        r_true, _ = self.compute_residuals(x, y, edge_index, edge_attr, distflow)
        loss = self.p_loss(r_true, x, edge_index, edge_attr)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
        opt.step()
        return loss.item()

    @torch.no_grad()
    def predict(self, data, distflow):
        """Predict for a single PyG Data object."""
        self.denoiser.eval()
        return self.sample(
            data.x.to(self.device),
            data.edge_index.to(self.device),
            data.edge_attr.to(self.device),
            distflow,
        )


# ============================================================
# Residual Normalizer
# ============================================================

class ResidualNormalizer:
    """
    Z-score normalizer for conditioning (x) and residuals (r).

    Separate from the standard DataNormalizer because we normalize
    residuals (y - y_phys) rather than the full state y.
    """

    def __init__(self, eps=1e-5):
        self.eps = eps
        self.x_mean = None
        self.x_std = None
        self.r_mean = None
        self.r_std = None

    def fit(self, train_d, distflow):
        """Fit normalizer on training data.

        Args:
            train_d: list of PyG Data objects.
            distflow: DistFlowCoarse instance.
        """
        xs = []
        rs = []

        for data in train_d:
            x = data.x
            y = data.y

            y_phys = distflow(x)
            r = y - y_phys

            xs.append(x)
            rs.append(r)

        x_all = torch.cat(xs, dim=0)
        r_all = torch.cat(rs, dim=0)

        self.x_mean = x_all.mean(dim=0, keepdim=True)
        self.x_std = x_all.std(dim=0, keepdim=True) + self.eps
        self.r_mean = r_all.mean(dim=0, keepdim=True)
        self.r_std = r_all.std(dim=0, keepdim=True) + self.eps

        print(f"ResidualNormalizer fitted:")
        print(f"  x_mean: {self.x_mean.squeeze().tolist()}")
        print(f"  x_std:  {self.x_std.squeeze().tolist()}")
        print(f"  r_mean: {self.r_mean.squeeze().tolist()}")
        print(f"  r_std:  {self.r_std.squeeze().tolist()}")

    def normalize_x(self, x):
        if self.x_mean is None:
            return x
        return (x - self.x_mean.to(x.device)) / self.x_std.to(x.device)

    def normalize_r(self, r):
        if self.r_mean is None:
            return r
        return (r - self.r_mean.to(r.device)) / self.r_std.to(r.device)

    def denormalize_r(self, r):
        if self.r_mean is None:
            return r
        return r * self.r_std.to(r.device) + self.r_mean.to(r.device)


# ============================================================
# Training Loop
# ============================================================

def train_pistdm_res(model, train_d, val_d, distflow, epochs=200, lr=1e-3,
                     patience=50, device='cpu'):
    """Train PISTDMRes.

    Args:
        model: PISTDMRes instance.
        train_d: training data list.
        val_d: validation data list.
        distflow: DistFlowCoarse instance.
        epochs: max training epochs.
        lr: learning rate.
        patience: early stopping patience.

    Returns:
        model: trained PISTDMRes.
    """
    opt = Adam(model.denoiser.parameters(), lr=lr, weight_decay=1e-12)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for ep in tqdm(range(epochs), desc='PI-DDPM-Res'):
        # Training
        model.denoiser.train()
        train_loss = 0.0
        for data in train_d:
            tl = model.train_step(data, opt, distflow)
            train_loss += tl
        train_loss /= len(train_d)

        # Validation
        model.denoiser.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_d:
                x = data.x.to(device)
                y = data.y.to(device)
                edge_index = data.edge_index.to(device)
                edge_attr = data.edge_attr.to(device)

                r_true, _ = model.compute_residuals(x, y, edge_index, edge_attr, distflow)
                vl = model.p_loss(r_true, x, edge_index, edge_attr)
                val_loss += vl.item()
        val_loss /= len(val_d)

        sch.step(val_loss)

        # Logging
        if ep % 10 == 0 or val_loss < best_val_loss:
            print(f"  ep {ep:3d}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}",
                  flush=True)

        # Best model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.denoiser.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {ep}", flush=True)
            break

    # Restore best model
    if best_state is not None:
        model.denoiser.load_state_dict(best_state)
        print(f"  Restored best model (val_loss={best_val_loss:.6f})", flush=True)

    return model


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def eval_pistdm_res(model, data_list, distflow):
    """Evaluate full state estimation performance.

    Computes MAE and RMSE on the denormalized predictions.

    Args:
        model: PISTDMRes instance.
        data_list: list of PyG Data objects.
        distflow: DistFlowCoarse instance.

    Returns:
        dict with mae, rmse, mae_phys_only, rmse_phys_only.
    """
    model.denoiser.eval()
    mae_total = 0.0
    rmse_total = 0.0
    mae_phys_total = 0.0
    rmse_phys_total = 0.0
    n_samples = len(data_list)

    for data in tqdm(data_list, desc='Eval PI-DDPM-Res'):
        data = data.to(model.device)
        x = data.x
        y_true = data.y
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # DistFlow coarse estimate
        y_phys = distflow(x, edge_index, edge_attr)
        y_phys = model._to_device(y_phys)

        # PI-DDPM-Res prediction
        y_pred = model.predict(data, distflow)

        # Errors
        err_phys = torch.abs(y_phys - y_true)
        err_pred = torch.abs(y_pred - y_true)

        mae_phys_total += err_phys.mean().item()
        mae_total += err_pred.mean().item()
        rmse_phys_total += (err_phys ** 2).mean().item()
        rmse_total += ((y_pred - y_true) ** 2).mean().item()

    return {
        'mae': mae_total / n_samples,
        'rmse': np.sqrt(rmse_total / n_samples),
        'mae_phys_only': mae_phys_total / n_samples,
        'rmse_phys_only': np.sqrt(rmse_phys_total / n_samples),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='PI-DDPM-Res training')
    parser.add_argument('--system', type=str, default='33',
                        help='Test system name')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--d_model', type=int, default=32,
                        help='Hidden dimension')
    parser.add_argument('--n_heads', type=int, default=2,
                        help='Number of attention heads')
    parser.add_argument('--n_blocks', type=int, default=3,
                        help='Number of PI-Attention blocks')
    parser.add_argument('--timesteps', type=int, default=50,
                        help='Number of DDPM timesteps')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--meas_rate', type=float, default=0.6,
                        help='Measurement rate (default 0.6 = 60%%)')
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ----------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("PI-DDPM-Res: DDPM with residual target")
    print("  (DistFlow coarse estimate + DDPM on residual + Ybus attention)")
    print(f"{'='*60}")

    splits = load_dataset(args.system, 'data')
    if splits is not None and abs(args.meas_rate - 0.6) < 1e-6:
        print(f"Loaded saved v3 dataset (60% meas rate) for '{args.system}'")
        train_d, val_d, test_d = splits['train'], splits['val'], splits['test']
    else:
        print(f"Generating dataset with meas_rate={args.meas_rate}...")
        dl, _ = generate_paper_dataset(args.system, n_steps=500,
                                        meas_rate=args.meas_rate, seed=args.seed)
        splits = create_splits(dl)
        train_d, val_d, test_d = splits['train'], splits['val'], splits['test']
        print(f"Generated: {len(train_d)} train, {len(val_d)} val, {len(test_d)} test")
    n_buses = train_d[0].n_buses
    print(f"Data: {len(train_d)} train, {len(val_d)} val, {len(test_d)} test, "
          f"{n_buses} buses")

    sample = train_d[0]
    node_dim = sample.y.shape[1]  # 2 (Vm, Va)
    cond_dim = sample.x.shape[1]  # 4 (P_inj, Q_inj, P_flow, Q_flow)
    print(f"  node_dim={node_dim}, cond_dim={cond_dim}")

    # ----------------------------------------------------------
    # 2. Initialize DistFlow
    # ----------------------------------------------------------
    params = get_test_system(args.system)
    distflow = DistFlowCoarse(params['parent'], params['r_pu'], params['x_pu'])
    print(f"DistFlow initialised: {n_buses} buses, {params['n_branches']} branches")

    # ----------------------------------------------------------
    # 3. Fit normalizer
    # ----------------------------------------------------------
    print("\nFitting residual normalizer...")
    norm = ResidualNormalizer()
    norm.fit(train_d, distflow)

    # Report raw residual statistics
    rs = []
    for data in train_d:
        y_phys = distflow(data.x)
        rs.append(data.y - y_phys)
    r_all = torch.cat(rs, dim=0)
    print(f"  Raw residual std: {r_all.std(dim=0).tolist()}")
    print(f"  Raw residual range: [{r_all.min().item():.6f}, {r_all.max().item():.6f}]")

    # ----------------------------------------------------------
    # 4. Initialize model
    # ----------------------------------------------------------
    model = PISTDMRes(
        node_dim=node_dim, cond_dim=cond_dim,
        d_model=args.d_model, n_heads=args.n_heads,
        n_blocks=args.n_blocks, timesteps=args.timesteps,
        device=device,
    )
    model.set_normalizer(norm)

    n_params = sum(p.numel() for p in model.denoiser.parameters())
    print(f"\nPIDenoiser (residual): params={n_params:,}")
    print(f"  Architecture: {args.n_blocks} PI-blocks, d_model={args.d_model}, "
          f"n_heads={args.n_heads}")
    print(f"  Timesteps: {args.timesteps}")

    # ----------------------------------------------------------
    # 5. Training
    # ----------------------------------------------------------
    print(f"\nTraining {args.epochs} epochs...")
    print(f"  Optimizer: Adam(lr={args.lr}, weight_decay=1e-12)")
    print(f"  Scheduler: ReduceLROnPlateau(factor=0.5, patience=15)")

    model = train_pistdm_res(
        model, train_d, val_d, distflow,
        epochs=args.epochs, lr=args.lr,
        patience=50, device=device,
    )

    # ----------------------------------------------------------
    # 6. Evaluation
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("Test Set Evaluation")
    print(f"{'='*60}")

    metrics = eval_pistdm_res(model, test_d, distflow)
    metrics['params'] = n_params
    metrics['d_model'] = args.d_model
    metrics['n_blocks'] = args.n_blocks
    metrics['n_heads'] = args.n_heads
    metrics['timesteps'] = args.timesteps
    metrics['model'] = 'PI-DDPM-Res'

    print(f"\n  DistFlow coarse only:")
    print(f"    MAE  = {metrics['mae_phys_only']:.6f}")
    print(f"    RMSE = {metrics['rmse_phys_only']:.6f}")
    print("  PI-DDPM-Res (DDPM + residual + Ybus attention):")
    print(f"    MAE  = {metrics['mae']:.6f}")
    print(f"    RMSE = {metrics['rmse']:.6f}")
    print(f"  Improvement over DistFlow only:")
    print(f"    MAE  = {metrics['mae_phys_only'] - metrics['mae']:.6f} "
          f"({(1 - metrics['mae']/metrics['mae_phys_only'])*100:.1f}%)")
    print(f"  Model params: {n_params:,}")

    # ----------------------------------------------------------
    # 7. Save outputs
    # ----------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"pi_ddpm_res_{args.system}.pt"
    torch.save(model.denoiser.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    metrics_path = out_dir / f"pi_ddpm_res_metrics_{args.system}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    print(f"\nDone!")
    return metrics


if __name__ == '__main__':
    main()
