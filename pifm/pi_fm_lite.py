"""
PI-RFM-Lite: CPU-Practical Lightweight Flow Matching.

A reduced-parameter variant of PI-FM-Res (PI-FM-Res) to validate the
precision-efficiency trade-off of Flow Matching for DSSE.

Architecture (same as PI-FM-Res, scaled down):
  1. DistFlow coarse estimation: y_phys = distflow_coarse(x)
  2. Residual target: r_1 = y_true - y_phys
  3. Flow Matching with linear path: r_t = (1-t)*r_0 + t*r_1
  4. Velocity field network (PI-blocks with Ybus attention): v_theta(r_t, t, x, ...)
  5. ODE inference (Euler): integrate r_0 -> r_pred, then y_pred = y_phys + r_pred

Key differences from PI-FM-Res:
  - d_model: 32 -> 16
  - n_blocks: 3 -> 2
  - flow_steps: 10 -> 5
  - Estimated params: ~15K (vs 59K for PI-FM-Res)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from pathlib import Path
import json
import argparse
import time
from tqdm import tqdm

from .data_generator import load_dataset, create_splits
from .test_systems import get_test_system
from .distflow_module import DistFlowCoarse
from .utils import set_seed
from .pi_stdm import PhysicsInformedAttention, PIBlock
from .dataset_builder import generate_paper_dataset


# ============================================================
# Velocity Field Network (Lite)
# ============================================================

class PIVelocityFieldLite(nn.Module):
    """
    Physics-Informed Velocity Field for Flow Matching (Lite version).

    Identical architecture to PI-FM-Res PIVelocityField, with smaller d_model and
    fewer blocks for reduced parameter count.

    Architecture:
      - Input embedding: [r_t, cond] -> d_model
      - Time embedding: scalar t -> d_model (shared across blocks)
      - n_blocks PI-Attention blocks (each with per-block time FiLM)
      - Output projection: d_model -> node_dim
    """

    def __init__(self, node_dim=2, cond_dim=4, d_model=16, n_heads=2, n_blocks=2):
        """
        Args:
            node_dim: dimension of residual per node (2 for Vm, Va)
            cond_dim: dimension of conditioning input per node (4)
            d_model: hidden dimension (default 16, Lite)
            n_heads: number of attention heads (default 2, d_k=8)
            n_blocks: number of PI-Attention blocks (default 2, Lite)
        """
        super().__init__()
        self.d_model = d_model

        # Input embedding: [r_t, cond] -> d_model
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim + cond_dim, d_model),
            nn.ReLU(),
        )

        # Time embedding: scalar t -> d_model
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # PI-Attention blocks (each has its own time FiLM projection)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(PIBlock(d_model, n_heads))

        # Output projection: d_model -> node_dim (velocity)
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, node_dim),
        )

    def forward(self, r_t, t, cond, edge_index=None, edge_attr=None):
        """
        Args:
            r_t: (N, node_dim) current residual state
            t: float or (1,) tensor, time in [0, 1]
            cond: (N, cond_dim) conditioning (measurements)
            edge_index: (2, E) graph connectivity
            edge_attr: (E, 1) normalized Ybus edge weights

        Returns:
            v_pred: (N, node_dim) predicted velocity dr/dt
        """
        N = r_t.shape[0]

        # Build Ybus matrix from edge_attr (with symmetrization)
        ybus = None
        if edge_index is not None and edge_attr is not None:
            ybus = torch.zeros(N, N, device=r_t.device)
            row, col = edge_index
            ybus[row, col] = edge_attr.squeeze(-1)
            ybus = (ybus + ybus.t()) / 2  # symmetrize

        # Compute time embedding (shared across blocks)
        if isinstance(t, float) or (isinstance(t, torch.Tensor) and t.numel() == 1):
            t_tensor = torch.tensor([t], dtype=torch.float32, device=r_t.device)
        else:
            t_tensor = t
        t_emb = self.time_mlp(t_tensor)

        # Concatenate and project input
        h = torch.cat([r_t, cond], dim=-1)  # (N, node_dim + cond_dim)
        h = self.input_proj(h)

        # PI-Attention blocks — each receives time embedding for FiLM
        for block in self.blocks:
            h = block(h, ybus, t_emb)

        return self.output(h)


# ============================================================
# Flow Matcher Wrapper (Lite)
# ============================================================

class PIFlowMatcherLite:
    """
    PI-RFM-Lite: Physics-Informed Flow Matching with Residual Learning.

    Same algorithmic structure as PI-FM-Res PIFlowMatcher, with reduced model
    capacity and fewer inference steps.

    Combines:
      - DistFlow coarse estimate (physics prior)
      - Flow Matching for residual prediction (deterministic ODE)
      - Ybus-informed attention (physics guidance in velocity field)

    Flow path: r_t = (1-t) * r_0 + t * r_1  (linear interpolation)
    Velocity:  v(r_t, t) = dr_t/dt = r_1 - r_0

    Training: MSE(v_theta(r_t, t, cond, Ybus), r_1 - r_0)

    Inference (ODE, Euler method):
      r_0 ~ N(0, sigma^2 * I)
      for i in range(flow_steps):
          t = i / flow_steps
          dt = 1 / flow_steps
          r_t = r_t + v_theta(r_t, t, cond, Ybus) * dt
      y_pred = y_phys + r_t
    """

    def __init__(self, node_dim=2, cond_dim=4, d_model=16, n_heads=2,
                 n_blocks=2, flow_steps=5, device='cpu'):
        """
        Args:
            node_dim: residual dimension per node (2 for Vm, Va)
            cond_dim: conditioning dimension per node (4)
            d_model: hidden dimension (default 16, Lite)
            n_heads: attention heads (default 2)
            n_blocks: PI-Attention blocks (default 2, Lite)
            flow_steps: ODE integration steps for inference (default 5, Lite)
            device: torch device
        """
        self.node_dim = node_dim
        self.cond_dim = cond_dim
        self.flow_steps = flow_steps
        self.device = device

        self.velocity_field = PIVelocityFieldLite(
            node_dim=node_dim, cond_dim=cond_dim,
            d_model=d_model, n_heads=n_heads, n_blocks=n_blocks,
        ).to(device)

        # Will be set from data statistics
        self.noise_sigma = 1.0  # default, will be overridden

    def _to_device(self, tensor):
        """Move tensor to device, handling tensors on different devices."""
        if tensor.device != self.device:
            return tensor.to(self.device)
        return tensor

    def set_normalizer(self, norm):
        """Set data normalizer for conditioning and denormalization."""
        self.norm = norm

    def set_noise_sigma(self, sigma):
        """Set initial noise standard deviation for sampling."""
        self.noise_sigma = sigma

    # ----------------------------------------------------------
    # Residual computation (via DistFlow)
    # ----------------------------------------------------------

    def compute_residuals(self, x, y, edge_index, edge_attr, distflow):
        """
        Compute residual targets r_1 = y - distflow_coarse(x).

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
        y_phys = distflow(x, edge_index, edge_attr)  # (N, 2)
        y_phys = self._to_device(y_phys)

        r_true = y - y_phys
        return r_true, y_phys

    # ----------------------------------------------------------
    # Flow Matching loss
    # ----------------------------------------------------------

    def fm_loss(self, r_true, r_true_norm, cond_norm, edge_index, edge_attr):
        """
        Compute Flow Matching loss for a batch of residual targets.

        Args:
            r_true: (N, node_dim) raw residuals (for info only).
            r_true_norm: (N, node_dim) normalized residuals (target r_1).
            cond_norm: (N, cond_dim) normalized conditioning.
            edge_index: (2, E) graph structure.
            edge_attr: (E, 1) edge weights.

        Returns:
            loss: scalar MSE loss.
        """
        N = r_true_norm.shape[0]

        # Sample time t ~ Uniform(0, 1)
        t = torch.rand(1, device=self.device).item()

        # Sample initial noise r_0 ~ N(0, sigma^2 I)
        r_0 = torch.randn(N, self.node_dim, device=self.device) * self.noise_sigma

        # Linear interpolation: r_t = (1-t) * r_0 + t * r_1
        r_1 = r_true_norm  # target is the normalized residual
        r_t = (1.0 - t) * r_0 + t * r_1

        # True velocity: v = r_1 - r_0
        v_true = r_1 - r_0

        # Predicted velocity
        v_pred = self.velocity_field(r_t, t, cond_norm, edge_index, edge_attr)

        return F.mse_loss(v_pred, v_true)

    # ----------------------------------------------------------
    # Training step
    # ----------------------------------------------------------

    def train_step(self, data, opt, distflow):
        """
        Single training step.

        Args:
            data: PyG Data object with x, y, edge_index, edge_attr, n_buses.
            opt: torch optimizer.
            distflow: DistFlowCoarse instance.

        Returns:
            loss: scalar training loss.
        """
        self.velocity_field.train()
        opt.zero_grad()

        x = data.x.to(self.device)
        y = data.y.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_attr = data.edge_attr.to(self.device)

        # 1. Compute residual target r_true = y - y_phys
        r_true, _ = self.compute_residuals(x, y, edge_index, edge_attr, distflow)

        # 2. Normalize
        cond_norm = self.norm.normalize_x(x)
        r_true_norm = self.norm.normalize_r(r_true)

        # 3. Flow Matching loss
        loss = self.fm_loss(r_true, r_true_norm, cond_norm, edge_index, edge_attr)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.velocity_field.parameters(), 1.0)
        opt.step()

        return loss.item()

    # ----------------------------------------------------------
    # ODE Inference (Euler method)
    # ----------------------------------------------------------

    @torch.no_grad()
    def sample(self, cond, edge_index, edge_attr, distflow, steps=None):
        """
        Generate samples via ODE integration (Euler method).

        r_0 ~ N(0, sigma^2 * I)
        for i in range(steps):
            t = i / steps
            dt = 1 / steps
            r_t = r_t + v_theta(r_t, t, cond, Ybus) * dt
        y_pred = y_phys + r_t (denormalized)

        Args:
            cond: (N, cond_dim) conditioning (raw measurements).
            edge_index: (2, E) graph structure.
            edge_attr: (E, 1) edge weights.
            distflow: DistFlowCoarse instance.
            steps: ODE integration steps (default: self.flow_steps).

        Returns:
            y_pred: (N, node_dim) final state estimate.
        """
        self.velocity_field.eval()

        if steps is None:
            steps = self.flow_steps

        N = cond.shape[0]
        device = self.device

        # Normalize conditioning
        cond_norm = self.norm.normalize_x(cond)

        # Compute DistFlow coarse estimate
        y_phys = distflow(cond, edge_index, edge_attr)
        y_phys = self._to_device(y_phys)

        # Initial noise
        r_t = torch.randn(N, self.node_dim, device=device) * self.noise_sigma

        # Euler integration
        dt = 1.0 / steps
        for i in range(steps):
            t = i / steps
            v_pred = self.velocity_field(r_t, t, cond_norm, edge_index, edge_attr)
            r_t = r_t + v_pred * dt

        # Denormalize residual and add to physics estimate
        r_pred_raw = self.norm.denormalize_r(r_t)
        y_pred = y_phys + r_pred_raw

        return y_pred

    @torch.no_grad()
    def predict(self, data, distflow, steps=None):
        """Predict for a single PyG Data object."""
        self.velocity_field.eval()
        return self.sample(
            data.x.to(self.device),
            data.edge_index.to(self.device),
            data.edge_attr.to(self.device),
            distflow,
            steps=steps,
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

def train_pifm_lite(model, train_d, val_d, distflow, epochs=200, lr=1e-3,
                    patience=50, device='cpu'):
    """Train PIFlowMatcherLite.

    Args:
        model: PIFlowMatcherLite instance.
        train_d: training data list.
        val_d: validation data list.
        distflow: DistFlowCoarse instance.
        epochs: max training epochs.
        lr: learning rate.
        patience: early stopping patience.

    Returns:
        model: trained PIFlowMatcherLite.
    """
    opt = Adam(model.velocity_field.parameters(), lr=lr, weight_decay=1e-12)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for ep in tqdm(range(epochs), desc='PI-RFM-Lite'):
        # Training
        model.velocity_field.train()
        train_loss = 0.0
        for data in train_d:
            tl = model.train_step(data, opt, distflow)
            train_loss += tl
        train_loss /= len(train_d)

        # Validation
        model.velocity_field.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_d:
                x = data.x.to(device)
                y = data.y.to(device)
                edge_index = data.edge_index.to(device)
                edge_attr = data.edge_attr.to(device)

                r_true, _ = model.compute_residuals(x, y, edge_index, edge_attr, distflow)
                cond_norm = model.norm.normalize_x(x)
                r_true_norm = model.norm.normalize_r(r_true)

                vl = model.fm_loss(r_true, r_true_norm, cond_norm, edge_index, edge_attr)
                val_loss += vl.item()
        val_loss /= len(val_d)

        sch.step(val_loss)

        # Logging
        if ep % 20 == 0 or val_loss < best_val_loss:
            print(f"  ep {ep:3d}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}",
                  flush=True)

        # Best model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.velocity_field.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {ep}", flush=True)
            break

    # Restore best model
    if best_state is not None:
        model.velocity_field.load_state_dict(best_state)
        print(f"  Restored best model (val_loss={best_val_loss:.6f})", flush=True)

    return model


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def eval_pifm_lite(model, data_list, distflow, steps=None):
    """Evaluate full state estimation performance.

    Computes MAE and RMSE on the denormalized predictions.

    Args:
        model: PIFlowMatcherLite instance.
        data_list: list of PyG Data objects.
        distflow: DistFlowCoarse instance.
        steps: override inference steps (None uses model.flow_steps).

    Returns:
        dict with mae, rmse, mae_phys_only, rmse_phys_only.
    """
    model.velocity_field.eval()
    mae_total = 0.0
    rmse_total = 0.0
    mae_phys_total = 0.0
    rmse_phys_total = 0.0
    n_samples = len(data_list)

    desc = f'Eval ({steps} steps)' if steps else 'Eval'
    for data in tqdm(data_list, desc=desc):
        data = data.to(model.device)
        x = data.x
        y_true = data.y
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # DistFlow coarse estimate
        y_phys = distflow(x, edge_index, edge_attr)
        y_phys = model._to_device(y_phys)

        # PI-FM prediction
        y_pred = model.predict(data, distflow, steps=steps)

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
    parser = argparse.ArgumentParser(description='PI-RFM-Lite Training')
    parser.add_argument('--system', type=str, default='33',
                        help='Test system name')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--d_model', type=int, default=16,
                        help='Hidden dimension (Lite default: 16)')
    parser.add_argument('--n_heads', type=int, default=2,
                        help='Number of attention heads')
    parser.add_argument('--n_blocks', type=int, default=2,
                        help='Number of PI-Attention blocks (Lite default: 2)')
    parser.add_argument('--flow_steps', type=int, default=5,
                        help='ODE integration steps for inference (Lite default: 5)')
    parser.add_argument('--sigma', type=float, default=None,
                        help='Initial noise std (default: auto from data)')
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
    print(f"PyTorch version: {torch.__version__}")

    # ----------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("PI-RFM-Lite: Lightweight Flow Matching + Residual Learning")
    print(f"  (Lite: d_model={args.d_model}, n_blocks={args.n_blocks}, "
          f"flow_steps={args.flow_steps})")
    print(f"{'='*60}")

    # Try loading saved dataset first; generate on demand if meas_rate differs
    splits = load_dataset(f'{args.system}v3', 'data')
    if splits is not None and abs(args.meas_rate - 0.6) < 1e-6:
        print(f"Loaded saved v3 dataset (60%% meas rate) for '{args.system}'")
        train_d, val_d, test_d = splits['train'], splits['val'], splits['test']
    else:
        print(f"Generating dataset with meas_rate={args.meas_rate}...")
        dl, _ = generate_paper_dataset(args.system, n_steps=500,
                                        meas_rate=args.meas_rate, seed=args.seed)
        splits = create_splits(dl)  # 400 train / 50 val / 50 test
        train_d, val_d, test_d = splits['train'], splits['val'], splits['test']
        print(f"Generated: {len(train_d)} train, {len(val_d)} val, {len(test_d)} test")
    n_buses = train_d[0].n_buses
    print(f"Data: {len(train_d)} train, {len(val_d)} val, {len(test_d)} test, "
          f"{n_buses} buses")

    # Data format check
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
    # 3. Fit normalizer and compute sigma
    # ----------------------------------------------------------
    print("\nFitting residual normalizer...")
    norm = ResidualNormalizer()
    norm.fit(train_d, distflow)

    # Compute noise sigma from training residual std
    if args.sigma is not None:
        noise_sigma = args.sigma
    else:
        noise_sigma = 0.8
        print(f"  Using default noise_sigma={noise_sigma} (normalized space)")

    # Also report raw residual statistics
    rs = []
    for data in train_d:
        y_phys = distflow(data.x)
        rs.append(data.y - y_phys)
    r_all = torch.cat(rs, dim=0)
    print(f"  Raw residual std: {r_all.std(dim=0).tolist()}")
    print(f"  Noise sigma (normalized): {noise_sigma}")

    # ----------------------------------------------------------
    # 4. Initialize model
    # ----------------------------------------------------------
    model = PIFlowMatcherLite(
        node_dim=node_dim, cond_dim=cond_dim,
        d_model=args.d_model, n_heads=args.n_heads,
        n_blocks=args.n_blocks, flow_steps=args.flow_steps,
        device=device,
    )
    model.set_normalizer(norm)
    model.set_noise_sigma(noise_sigma)

    n_params = sum(p.numel() for p in model.velocity_field.parameters())
    print(f"\nPIVelocityFieldLite: params={n_params:,}")
    print(f"  Architecture: {args.n_blocks} PI-blocks, d_model={args.d_model}, "
          f"n_heads={args.n_heads}")
    print(f"  Flow steps: {args.flow_steps}")
    print(f"  d_k (per head): {args.d_model // args.n_heads}")

    # ----------------------------------------------------------
    # 5. Training
    # ----------------------------------------------------------
    print(f"\nTraining {args.epochs} epochs...")
    print(f"  Optimizer: Adam(lr={args.lr}, weight_decay=1e-12)")
    print(f"  Scheduler: ReduceLROnPlateau(factor=0.5, patience=15)")
    print(f"  Noise sigma: {noise_sigma}")

    t0 = time.time()
    model = train_pifm_lite(
        model, train_d, val_d, distflow,
        epochs=args.epochs, lr=args.lr,
        patience=50, device=device,
    )
    train_time = time.time() - t0
    print(f"\nTraining completed in {train_time:.1f}s ({train_time/60:.1f} min)")

    # ----------------------------------------------------------
    # 6. Evaluation — default steps
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("Test Set Evaluation (default flow_steps)")
    print(f"{'='*60}")

    metrics = eval_pifm_lite(model, test_d, distflow, steps=args.flow_steps)
    metrics['params'] = n_params
    metrics['d_model'] = args.d_model
    metrics['n_blocks'] = args.n_blocks
    metrics['n_heads'] = args.n_heads
    metrics['flow_steps'] = args.flow_steps
    metrics['noise_sigma'] = noise_sigma
    metrics['train_time_s'] = round(train_time, 1)

    print(f"\n  DistFlow coarse only:")
    print(f"    MAE  = {metrics['mae_phys_only']:.6f}")
    print(f"    RMSE = {metrics['rmse_phys_only']:.6f}")
    print(f"  PI-RFM-Lite (DistFlow + FM residual + Ybus):")
    print(f"    MAE  = {metrics['mae']:.6f}")
    print(f"    RMSE = {metrics['rmse']:.6f}")
    print(f"  Improvement over DistFlow only:")
    print(f"    MAE  = {metrics['mae_phys_only'] - metrics['mae']:.6f} "
          f"({(1 - metrics['mae']/metrics['mae_phys_only'])*100:.1f}%)")
    print(f"  Model params: {n_params:,}")
    print(f"  Train time: {train_time:.1f}s ({train_time/60:.1f} min)")

    # ----------------------------------------------------------
    # 7. Inference step comparison: 5 vs 10 steps
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("Inference Step Comparison: 5 steps vs 10 steps")
    print(f"{'='*60}")

    metrics_5 = metrics  # already computed with default flow_steps

    metrics_10 = eval_pifm_lite(model, test_d, distflow, steps=10)
    metrics['mae_5steps'] = metrics_5['mae']
    metrics['rmse_5steps'] = metrics_5['rmse']
    metrics['mae_10steps'] = metrics_10['mae']
    metrics['rmse_10steps'] = metrics_10['rmse']

    print(f"  5-step Euler:  MAE={metrics_5['mae']:.6f}  RMSE={metrics_5['rmse']:.6f}")
    print(f"  10-step Euler: MAE={metrics_10['mae']:.6f}  RMSE={metrics_10['rmse']:.6f}")
    delta_mae = metrics_10['mae'] - metrics_5['mae']
    print(f"  Delta (10-5):  MAE={delta_mae:+.6f} "
          f"({'improvement' if delta_mae < 0 else 'degradation'})")

    # ----------------------------------------------------------
    # 8. Save outputs
    # ----------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rate_suffix = f"_{int(args.meas_rate*100)}meas" if abs(args.meas_rate - 0.6) > 1e-6 else ""
    model_path = out_dir / f"lite_pifm_lite_{args.system}{rate_suffix}.pt"
    torch.save(model.velocity_field.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    metrics_path = out_dir / f"lite_metrics_{args.system}{rate_suffix}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # ----------------------------------------------------------
    # 9. Cross-configuration comparison
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("PI-FM-Res vs PI-RFM-Lite Cross-Configuration Comparison")
    print(f"{'='*60}")

    # Values reported for the full model at 60% coverage
    ref_mae = 0.002625   # PI-FM-Res at 60%: 59K params, 10 steps
    ref_params = 59010
    ref_steps = 10
    ref_train_time = 17 * 60  # ~17 min in seconds

    print(f"  {'Configuration':<20} {'Params':>8} {'Steps':>6} {'MAE':>10} {'Train':>8}")
    print(f"  {'-'*60}")
    print(f"  {'PI-FM-Res':<20} {ref_params:>8,} {ref_steps:>6} {ref_mae:>10.6f} {'~17min':>8}")
    print(f"  {'PI-RFM-Lite':<20} {n_params:>8,} {args.flow_steps:>6} {metrics['mae']:>10.6f} "
          f"{train_time/60:.0f}min")

    # Trade-off analysis
    print(f"\n  Trade-off Analysis:")
    mae_ratio = metrics['mae'] / ref_mae
    param_ratio = n_params / ref_params
    print(f"    PI-RFM-Lite MAE / PI-FM-Res MAE:          {mae_ratio:.2f}x (lower is better)")
    print(f"    PI-RFM-Lite params / PI-FM-Res params:     {param_ratio:.2f}x (lower is cheaper)")
    print(f"    PI-RFM-Lite steps / PI-FM-Res steps:       {args.flow_steps / ref_steps:.2f}x (lower is faster)")
    print(f"    PI-RFM-Lite train time / PI-FM-Res:        {train_time / ref_train_time:.2f}x")

    # Efficiency score: how much MAE per parameter?
    ref_efficiency = ref_mae * ref_params  # lower is better per-param
    lite_efficiency = metrics['mae'] * n_params
    print(f"    MAE*params (PI-FM-Res):           {ref_efficiency:.2f}")
    print(f"    MAE*params (PI-RFM-Lite):           {lite_efficiency:.2f} "
          f"({'better' if lite_efficiency < ref_efficiency else 'worse'})")

    print(f"\nDone!")
    return metrics


if __name__ == '__main__':
    main()
