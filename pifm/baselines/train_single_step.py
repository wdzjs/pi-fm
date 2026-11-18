"""
Training and evaluation for the single-step baselines (PIGACN, GCN).

Input is the same measurement tensor the flow models consume: P/Q injections at
measured buses, branch P/Q flows on a subset of lines, zero elsewhere, with a
per-node observation mask. Output is (Vm, Va).

Usage:
    python -m pifm.baselines.train_single_step --model pigacn --system 33
    python -m pifm.baselines.train_single_step --model gcn --system 33
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json

from .pigacn import PIGACN, GCN, DataNormalizer
from ..dataset_builder import generate_paper_dataset
from ..data_generator import load_dataset, create_splits, save_dataset
from ..utils import set_seed


def train_epoch(model, data_list, optimizer, normalizer, batch_size, device, use_physics=True):
    """Train with mini-batch sampling."""
    model.train()
    total_loss = 0.0
    n = len(data_list)
    indices = np.random.permutation(n)

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        if len(batch_idx) == 0:
            continue

        optimizer.zero_grad()
        batch_loss = 0.0
        for i in batch_idx:
            data = data_list[i].to(device)
            x_n = normalizer.normalize_x(data.x)
            y_t = normalizer.normalize_y(data.y)
            ei = data.edge_index if use_physics else None
            ea = data.edge_attr if use_physics else None
            pred = model(x_n, ei, ea, data.obs_mask)
            batch_loss += F.mse_loss(pred, y_t)
        batch_loss /= len(batch_idx)

        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += batch_loss.item()

    return total_loss / max(1, (n // batch_size))


@torch.no_grad()
def evaluate(model, data_list, normalizer, device, use_physics=True):
    """Evaluate with denormalized metrics."""
    model.eval()
    mae, rmse, mape = 0.0, 0.0, 0.0
    n = len(data_list)
    for data in data_list:
        data = data.to(device)
        x_n = normalizer.normalize_x(data.x)
        ei = data.edge_index if use_physics else None
        ea = data.edge_attr if use_physics else None
        pred_n = model(x_n, ei, ea, data.obs_mask)
        pred = normalizer.denormalize_y(pred_n)
        true = data.y

        err = torch.abs(pred - true)
        mae += err.mean().item()
        rmse += (err ** 2).mean().item()
        mape += (err / (true.abs() + 1e-6)).mean().item()

    return {'mae': mae / n, 'rmse': np.sqrt(rmse / n), 'mape': mape / n * 100}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', type=str, default='33')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--model', type=str, default='pigacn',
                        choices=['pigacn', 'gcn'])
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_physics', action='store_true',
                        help='Disable Ybus physics-informed attention (ablation)')
    parser.add_argument('--noise_std', type=float, default=0.001,
                        help='Measurement noise std (default 0.001 = 0.1%%)')
    parser.add_argument('--meas_rate', type=float, default=0.6,
                        help='Measurement rate (default 0.6 = 60%%)')

    args = parser.parse_args()
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Generate data
    splits = load_dataset(args.system, 'data')
    if splits is None:
        dl, _ = generate_paper_dataset(args.system, n_steps=500,
                                       meas_rate=args.meas_rate,
                                       noise_std=args.noise_std)
        splits = create_splits(dl)
        save_dataset(splits, args.system, 'data')

    train_d, val_d, test_d = splits['train'], splits['val'], splits['test']
    print(f"Data: {len(train_d)} train, {len(val_d)} val, {len(test_d)} test")

    # Normalizer
    norm = DataNormalizer()
    norm.fit(train_d)

    # Model
    if args.model == 'pigacn':
        model = PIGACN(in_channels=4, out_channels=2, d_model=10, n_heads=2, n_layers=3)
    else:
        model = GCN(in_dim=4, out_dim=2)

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}, params={n_params:,}")

    # Train
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-12)
    best_val = float('inf')
    best_state = None

    use_physics = not args.no_physics
    for ep in tqdm(range(args.epochs), desc='Training'):
        tl = train_epoch(model, train_d, optimizer, norm, args.batch_size, device, use_physics)
        vm = evaluate(model, val_d, norm, device, use_physics)
        if vm['mae'] < best_val:
            best_val = vm['mae']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 50 == 0:
            print(f"  ep {ep}: train={tl:.6f} val_mae={vm['mae']:.6f}")

    if best_state:
        model.load_state_dict(best_state)

    test_m = evaluate(model, test_d, norm, device, use_physics)
    print(f"\nTest: MAE={test_m['mae']:.6f}, RMSE={test_m['rmse']:.6f}, MAPE={test_m['mape']:.2f}%")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / f"{args.model}_{args.system}.pt")
    with open(out / f"{args.model}_metrics_{args.system}.json", 'w') as f:
        json.dump(test_m, f, indent=2)

    print("Done!")


if __name__ == '__main__':
    main()
