"""
PI-FM-Res-PL: Flow Matching + Residual + Explicit Physics Loss.

Extends PI-FM-Res (PI-FM-Res) by adding an explicit physics residual loss term
that penalises power flow equation violations on the predicted state.

Architecture:
  1. DistFlow coarse estimation: y_phys = distflow_coarse(x)
  2. Residual target: r_1 = y_true - y_phys
  3. Flow Matching with linear path: r_t = (1-t)*r_0 + t*r_1
  4. Velocity field network (PI-blocks with Ybus attention): v_theta(r_t, t, cond, edge_index, edge_attr)
  5. Single-step r_1 estimation from r_t: r_1_hat = r_t + (1-t)*v_pred
  6. Physics loss on y_pred = y_phys + denorm(r_1_hat)
  7. Total loss = fm_loss + lambda_t * phys_loss (lambda with linear warmup)

Key improvement over PI-FM-Res: explicit physics regularisation guides the model
toward physically consistent predictions during training.
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

from .data_generator import load_dataset, create_splits
from .test_systems import get_test_system
from .distflow_module import DistFlowCoarse
from .utils import set_seed, build_admittance_matrix
from .pi_fm import PIVelocityField, ResidualNormalizer
from .dataset_builder import generate_paper_dataset


# ============================================================
# Physics Loss
# ============================================================

def physics_loss(y_pred, cond, ybus):
    """
    Compute power injection physics residual loss.

    Simplified power flow equation:
        P_inj[i] = Vm[i] * sum_j( Vm[j] * |Y[i,j]| * cos(theta[i] - theta[j]) )

    For each node i, computes the predicted active power injection from the
    estimated voltage profile and admittance matrix, then compares with the
    measured P_inj from the conditioning vector.

    Args:
        y_pred: (N, 2) — [Vm, theta] predicted state (raw/denormalised space)
        cond: (N, 4) — [P_inj, Q_inj, P_flow, Q_flow] raw measurements
        ybus: (N, N) — admittance magnitude matrix |Y| (diagonal zeroed)

    Returns:
        phys_loss: scalar MSE between predicted and true P_inj
    """
    Vm = y_pred[:, 0]  # (N,)
    theta = y_pred[:, 1]  # (N,)

    Vm_i = Vm.unsqueeze(1)  # (N, 1)
    Vm_j = Vm.unsqueeze(0)  # (1, N)
    theta_diff = theta.unsqueeze(1) - theta.unsqueeze(0)  # (N, N)

    # P_inj_pred[i] = Vm_i * sum_j( Vm_j * |Y_ij| * cos(theta_i - theta_j) )
    P_pred = (Vm_i * Vm_j * ybus * torch.cos(theta_diff)).sum(dim=1)

    # True P_inj from conditioning (first column)
    P_true = cond[:, 0]

    return F.mse_loss(P_pred, P_true)


def build_ybus_from_edges(edge_index, edge_attr, n_buses, device='cpu'):
    """
    Build full (N, N) admittance magnitude matrix from edge data.

    Diagonal entries are zeroed out — the physics loss only considers
    inter-bus coupling, not self-admittance.

    Args:
        edge_index: (2, E) graph edges (bidirectional).
        edge_attr: (E, 1) edge weights |Y_ij|.
        n_buses: number of buses.
        device: target device.

    Returns:
        ybus: (N, N) admittance magnitude matrix.
    """
    ybus = torch.zeros(n_buses, n_buses, device=device)
    row, col = edge_index
    # edge_attr already contains |Y_ij| for each directed edge
    ybus[row, col] = edge_attr.squeeze(-1)
    # Diagonal stays zero (no self-admittance in simplified physics loss)
    return ybus


# ============================================================
# PIFlowMatcher with Physics Loss
# ============================================================

class PIFlowMatcherPhysLoss:
    """
    PI-FM-Res-PL: PI-FM-Res with explicit physics loss regularisation.

    Extends PI-FM-Res's PIFlowMatcher by adding a physics residual loss term
    during training.  The physics loss penalises deviations from the
    power flow equations on the predicted state.

    Physics loss integration:
      - During FM training, estimate r_1_hat = r_t + (1-t) * v_pred
        (exact for linear path when v_pred = v_true).
      - Denormalise to get y_pred = y_phys + denorm(r_1_hat).
      - Compute P_inj_pred from y_pred via simplified power flow.
      - Compare with true P_inj from measurements.

    Lambda schedule:
      lambda_t = lambda_max * min(1.0, epoch / warmup_epochs)
      Linear warmup from 0 to lambda_max.
    """

    def __init__(self, node_dim=2, cond_dim=4, d_model=32, n_heads=2,
                 n_blocks=3, flow_steps=10, lambda_max=0.1, warmup_epochs=20,
                 device='cpu'):
        """
        Args:
            node_dim: residual dimension per node (2 for Vm, Va)
            cond_dim: conditioning dimension per node (4)
            d_model: hidden dimension
            n_heads: attention heads
            n_blocks: PI-Attention blocks
            flow_steps: ODE integration steps for inference
            lambda_max: maximum physics loss weight
            warmup_epochs: epochs to linearly warm up lambda from 0 to lambda_max
            device: torch device
        """
        self.node_dim = node_dim
        self.cond_dim = cond_dim
        self.flow_steps = flow_steps
        self.device = device
        self.lambda_max = lambda_max
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0  # updated each epoch for lambda scheduling

        self.velocity_field = PIVelocityField(
            node_dim=node_dim, cond_dim=cond_dim,
            d_model=d_model, n_heads=n_heads, n_blocks=n_blocks,
        ).to(device)

        # Will be set from data statistics
        self.noise_sigma = 1.0  # default, will be overridden
        self.ybus_full = None  # (N, N) admittance matrix, set by set_ybus()

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

    def set_ybus(self, ybus):
        """Set the full admittance magnitude matrix for physics loss."""
        self.ybus_full = ybus.to(self.device)

    @property
    def lambda_t(self):
        """Current physics loss weight (linear warmup)."""
        return self.lambda_max * min(1.0, self.current_epoch / max(1, self.warmup_epochs))

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
        # DistFlow coarse estimate
        y_phys = distflow(x, edge_index, edge_attr)  # (N, 2)
        y_phys = self._to_device(y_phys)

        r_true = y - y_phys
        return r_true, y_phys

    # ----------------------------------------------------------
    # Flow Matching loss (same as PI-FM-Res)
    # ----------------------------------------------------------

    def fm_loss(self, r_true_norm, cond_norm, edge_index, edge_attr):
        """
        Compute Flow Matching loss for a batch of residual targets.

        Args:
            r_true_norm: (N, node_dim) normalized residuals (target r_1).
            cond_norm: (N, cond_dim) normalized conditioning.
            edge_index: (2, E) graph structure.
            edge_attr: (E, 1) edge weights.

        Returns:
            fm_loss_val: scalar MSE loss.
            t: sampled time (for physics loss reuse).
            r_0: sampled noise (for physics loss reuse).
            v_pred: predicted velocity (for physics loss reuse).
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

        fm_loss_val = F.mse_loss(v_pred, v_true)

        return fm_loss_val, t, r_0, r_t, v_pred

    # ----------------------------------------------------------
    # Training step (with physics loss)
    # ----------------------------------------------------------

    def train_step(self, data, opt, distflow):
        """
        Single training step with physics loss.

        Computes FM loss + lambda_t * physics_loss.

        Args:
            data: PyG Data object with x, y, edge_index, edge_attr, n_buses.
            opt: torch optimizer.
            distflow: DistFlowCoarse instance.

        Returns:
            total: scalar total training loss.
            fm_val: FM loss component.
            phys_val: physics loss component.
        """
        self.velocity_field.train()
        opt.zero_grad()

        x = data.x.to(self.device)          # (N, 4) raw measurements
        y = data.y.to(self.device)          # (N, 2) true state
        edge_index = data.edge_index.to(self.device)
        edge_attr = data.edge_attr.to(self.device)

        # 1. Compute residual target r_true = y - y_phys (raw space)
        r_true, y_phys = self.compute_residuals(x, y, edge_index, edge_attr, distflow)

        # 2. Normalize
        cond_norm = self.norm.normalize_x(x)
        r_true_norm = self.norm.normalize_r(r_true)

        # 3. Flow Matching loss
        fm_val, t, r_0, r_t, v_pred = self.fm_loss(
            r_true_norm, cond_norm, edge_index, edge_attr
        )

        # 4. Physics loss: estimate r_1 from r_t and v_pred
        #    For linear path: r_1 = r_t + (1-t) * v  (exact when v = v_true)
        r_1_hat = r_t + (1.0 - t) * v_pred   # (N, 2) in normalised space

        # Denormalise residual and add to physics estimate
        r_1_raw = self.norm.denormalize_r(r_1_hat)
        y_pred = y_phys + r_1_raw  # (N, 2) predicted state in raw space

        # Build Ybus from edge data for this batch
        n_buses = x.shape[0]
        ybus = build_ybus_from_edges(edge_index, edge_attr, n_buses, self.device)

        phys_val = physics_loss(y_pred, x, ybus)

        # 5. Total loss with lambda scheduling
        lam = self.lambda_t
        total = fm_val + lam * phys_val

        total.backward()
        torch.nn.utils.clip_grad_norm_(self.velocity_field.parameters(), 1.0)
        opt.step()

        return total.item(), fm_val.item(), phys_val.item(), lam

    # ----------------------------------------------------------
    # ODE Inference (Euler method) — same as PI-FM-Res
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
        y_pred = y_phys + r_t (denormalised)

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

        # Denormalise residual and add to physics estimate
        r_pred_raw = self.norm.denormalize_r(r_t)
        y_pred = y_phys + r_pred_raw

        return y_pred

    @torch.no_grad()
    def predict(self, data, distflow):
        """Predict for a single PyG Data object."""
        self.velocity_field.eval()
        return self.sample(
            data.x.to(self.device),
            data.edge_index.to(self.device),
            data.edge_attr.to(self.device),
            distflow,
        )


# ============================================================
# Training Loop with Lambda Scheduling
# ============================================================

def train_pifm_physloss(model, train_d, val_d, distflow, epochs=200, lr=1e-3,
                        patience=50, device='cpu'):
    """Train PIFlowMatcherPhysLoss with lambda scheduling.

    Physics loss weight lambda_t linearly warms up from 0 to lambda_max
    over warmup_epochs.

    Args:
        model: PIFlowMatcherPhysLoss instance.
        train_d: training data list.
        val_d: validation data list.
        distflow: DistFlowCoarse instance.
        epochs: max training epochs.
        lr: learning rate.
        patience: early stopping patience.
        device: torch device.

    Returns:
        model: trained PIFlowMatcherPhysLoss.
        history: list of dicts with per-epoch metrics.
    """
    opt = Adam(model.velocity_field.parameters(), lr=lr, weight_decay=1e-12)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )

    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    patience_counter = 0
    history = []

    for ep in tqdm(range(epochs), desc='PI-FM-Res-PhysLoss'):
        # Update epoch for lambda scheduling
        model.current_epoch = ep

        # --- Training ---
        model.velocity_field.train()
        train_total = 0.0
        train_fm = 0.0
        train_phys = 0.0
        for data in train_d:
            total, fm_val, phys_val, lam = model.train_step(data, opt, distflow)
            train_total += total
            train_fm += fm_val
            train_phys += phys_val
        n_train = len(train_d)
        train_total /= n_train
        train_fm /= n_train
        train_phys /= n_train

        # --- Validation ---
        model.velocity_field.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_d:
                x = data.x.to(device)
                y = data.y.to(device)
                edge_index = data.edge_index.to(device)
                edge_attr = data.edge_attr.to(device)

                r_true, _ = model.compute_residuals(
                    x, y, edge_index, edge_attr, distflow
                )
                cond_norm = model.norm.normalize_x(x)
                r_true_norm = model.norm.normalize_r(r_true)

                fm_val, _, _, _, _ = model.fm_loss(
                    r_true_norm, cond_norm, edge_index, edge_attr
                )
                val_loss += fm_val.item()
        val_loss /= len(val_d)

        sch.step(val_loss)

        # Logging
        if ep % 20 == 0 or ep < 5 or val_loss < best_val_loss:
            print(f"  ep {ep:3d}: total={train_total:.6f}  fm={train_fm:.6f}  "
                  f"phys={train_phys:.6f}  lam={lam:.4f}  val_loss={val_loss:.6f}",
                  flush=True)

        history.append({
            'epoch': ep,
            'train_total': train_total,
            'train_fm': train_fm,
            'train_phys': train_phys,
            'lambda': lam,
            'val_loss': val_loss,
            'lr': opt.param_groups[0]['lr'],
        })

        # Best model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.velocity_field.state_dict().items()}
            best_epoch = ep
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {ep}", flush=True)
            break

    # Restore best model
    if best_state is not None:
        model.velocity_field.load_state_dict(best_state)
        print(f"  Restored best model (epoch {best_epoch}, "
              f"val_loss={best_val_loss:.6f})", flush=True)

    return model, history


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def eval_pifm(model, data_list, distflow):
    """Evaluate full state estimation performance.

    Computes MAE and RMSE on the denormalised predictions.

    Args:
        model: PIFlowMatcherPhysLoss instance.
        data_list: list of PyG Data objects.
        distflow: DistFlowCoarse instance.

    Returns:
        dict with mae, rmse, mae_phys_only, rmse_phys_only.
    """
    model.velocity_field.eval()
    mae_total = 0.0
    rmse_total = 0.0
    mae_phys_total = 0.0
    rmse_phys_total = 0.0
    n_samples = len(data_list)

    for data in tqdm(data_list, desc='Eval'):
        data = data.to(model.device)
        x = data.x
        y_true = data.y
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # DistFlow coarse estimate
        y_phys = distflow(x, edge_index, edge_attr)
        y_phys = model._to_device(y_phys)

        # PI-FM prediction
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
    parser = argparse.ArgumentParser(description='PI-FM-Res-PL Training')
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
    parser.add_argument('--flow_steps', type=int, default=10,
                        help='ODE integration steps for inference')
    parser.add_argument('--sigma', type=float, default=None,
                        help='Initial noise std (default: auto from data)')
    parser.add_argument('--lambda_max', type=float, default=0.1,
                        help='Maximum physics loss weight')
    parser.add_argument('--warmup_epochs', type=int, default=20,
                        help='Epochs for linear lambda warmup')
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
    print("PI-FM-Res-PL: Flow Matching + Residual + Physics Loss")
    print(f"  (PI-FM-Res base + explicit power flow constraint)")
    print(f"  lambda_max={args.lambda_max}, warmup={args.warmup_epochs} epochs")
    print(f"{'='*60}")

    # Try loading saved dataset first
    splits = load_dataset(f'{args.system}v3', 'data')
    if splits is not None and abs(args.meas_rate - 0.6) < 1e-6:
        print(f"Loaded saved v3 dataset (60% meas rate) for '{args.system}'")
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

    # Report raw residual statistics
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
    model = PIFlowMatcherPhysLoss(
        node_dim=node_dim, cond_dim=cond_dim,
        d_model=args.d_model, n_heads=args.n_heads,
        n_blocks=args.n_blocks, flow_steps=args.flow_steps,
        lambda_max=args.lambda_max, warmup_epochs=args.warmup_epochs,
        device=device,
    )
    model.set_normalizer(norm)
    model.set_noise_sigma(noise_sigma)

    # Build Ybus from test system params for physics loss
    _, _, Y_mag = build_admittance_matrix(params['r_pu'], params['x_pu'])
    ybus_full = torch.tensor(Y_mag, dtype=torch.float32)
    # Zero diagonal for simplified physics loss
    ybus_full.fill_diagonal_(0.0)
    model.set_ybus(ybus_full)
    print(f"Ybus built: {ybus_full.shape}, "
          f"mean |Y_ij|={ybus_full[ybus_full > 0].mean().item():.4f}")

    n_params = sum(p.numel() for p in model.velocity_field.parameters())
    print(f"\nPIVelocityField: params={n_params:,}")
    print(f"  Architecture: {args.n_blocks} PI-blocks, d_model={args.d_model}, "
          f"n_heads={args.n_heads}")
    print(f"  Flow steps: {args.flow_steps}")
    print(f"  Lambda max: {args.lambda_max}, warmup: {args.warmup_epochs} epochs")

    # ----------------------------------------------------------
    # 5. Training
    # ----------------------------------------------------------
    print(f"\nTraining {args.epochs} epochs...")
    print(f"  Optimizer: Adam(lr={args.lr}, weight_decay=1e-12)")
    print(f"  Scheduler: ReduceLROnPlateau(factor=0.5, patience=15)")
    print(f"  Noise sigma: {noise_sigma}")
    print(f"  Lambda schedule: 0 -> {args.lambda_max} over {args.warmup_epochs} epochs")

    model, history = train_pifm_physloss(
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

    metrics = eval_pifm(model, test_d, distflow)
    metrics['params'] = n_params
    metrics['d_model'] = args.d_model
    metrics['n_blocks'] = args.n_blocks
    metrics['n_heads'] = args.n_heads
    metrics['flow_steps'] = args.flow_steps
    metrics['noise_sigma'] = noise_sigma
    metrics['lambda_max'] = args.lambda_max
    metrics['warmup_epochs'] = args.warmup_epochs

    print(f"\n  DistFlow coarse only:")
    print(f"    MAE  = {metrics['mae_phys_only']:.6f}")
    print(f"    RMSE = {metrics['rmse_phys_only']:.6f}")
    print(f"  PI-FM-Res-PL:")
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

    # Save model
    rate_suffix = f"_{int(args.meas_rate*100)}meas" if abs(args.meas_rate - 0.6) > 1e-6 else ""
    model_path = out_dir / f"physloss_pifmres_physloss_{args.system}{rate_suffix}.pt"
    torch.save(model.velocity_field.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # Save metrics
    metrics_path = out_dir / f"physloss_metrics_{args.system}{rate_suffix}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Save training history
    history_path = out_dir / f"physloss_history_{args.system}{rate_suffix}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # ----------------------------------------------------------
    # 8. Comparison with PI-FM-Res
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("PI-FM-Res vs PI-FM-Res-PL Comparison (33-bus 60%)")
    print(f"{'='*60}")

    ref_mae = 0.002625  # PI-FM-Res at 60%, reported value
    physloss_mae = metrics['mae']

    print(f"  PI-FM-Res:          MAE = {ref_mae:.6f}")
    print(f"  PI-FM-Res-PL: MAE = {physloss_mae:.6f}")

    if physloss_mae < ref_mae:
        improvement = ref_mae - physloss_mae
        rel_improvement = (improvement / ref_mae) * 100
        print(f"  PI-FM-Res-PL IS better: improvement = {improvement:.6f} ({rel_improvement:.1f}%)")
    else:
        degradation = physloss_mae - ref_mae
        rel_degradation = (degradation / ref_mae) * 100
        print(f"  PI-FM-Res-PL is NOT better: degradation = {degradation:.6f} ({rel_degradation:.1f}%)")

    print(f"\nDone!")
    return metrics


if __name__ == '__main__':
    main()
