"""
Data generator for distribution network state estimation.

Generates time-series measurement data using DistFlow power flow.
Supports multiple test systems and creates train/val/test splits.
Outputs PyG Data objects.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path
from tqdm import tqdm

from .test_systems import get_test_system
from .utils import build_edge_index_from_parent, build_edge_attr_from_Y, build_admittance_matrix


def run_distflow(parent, r, x, p_load, q_load, v0=1.0, max_iter=10):
    """
    Run backward-forward sweep (DistFlow) for a radial distribution network.

    Uses iterative BFS with exact nonlinear DistFlow to handle larger voltage drops.

    Args:
        parent: (n_buses,) parent array, parent[0] = -1
        r: (n_branches,) line resistances (p.u.)
        x: (n_branches,) line reactances (p.u.)
        p_load: (n_buses,) active power injections (p.u.), negative = generation
        q_load: (n_buses,) reactive power injections (p.u.)
        v0: substation voltage magnitude (p.u.)
        max_iter: number of BFS iterations for nonlinear correction

    Returns:
        V: (n_buses,) voltage magnitudes (p.u.)
        theta: (n_buses,) voltage angles (rad)
        P_flow: (n_branches,) line active power flows
        Q_flow: (n_branches,) line reactive power flows
    """
    n_buses = len(parent)
    n_branches = n_buses - 1

    # Build child list for tree traversal
    children = [[] for _ in range(n_buses)]
    branch_idx = {}
    for i in range(1, n_buses):
        p = parent[i]
        if p >= 0:
            children[p].append(i)
            branch_idx[(p, i)] = i - 1

    # Topological order for BFS
    queue = [0]
    order = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in children[u]:
            queue.append(v)

    # Initialize voltages
    V = np.ones(n_buses) * v0
    theta = np.zeros(n_buses)

    # Iterative BFS for nonlinear accuracy
    p_load = np.asarray(p_load, dtype=np.float64).ravel()
    q_load = np.asarray(q_load, dtype=np.float64).ravel()
    assert p_load.ndim == 1 and q_load.ndim == 1, f"p_load ndim={p_load.ndim}, q_load ndim={q_load.ndim}"
    assert p_load.dtype == np.float64, f"p_load dtype={p_load.dtype}"

    for _ in range(max_iter):
        # Backward sweep (iterative post-order, no recursion)
        P_flow = np.zeros(n_branches, dtype=np.float64)
        Q_flow = np.zeros(n_branches, dtype=np.float64)

        # Post-order traversal: process leaves first
        visited = np.zeros(n_buses, dtype=bool)
        for u in reversed(order):
            p_acc = float(p_load[u])
            q_acc = float(q_load[u])
            for v in children[u]:
                b_idx = branch_idx[(u, v)]
                p_child = P_flow[b_idx]
                q_child = Q_flow[b_idx]
                I_sq = (p_child**2 + q_child**2) / (V[v]**2 + 1e-8)
                p_acc += p_child + r[b_idx] * I_sq
                q_acc += q_child + x[b_idx] * I_sq
            if parent[u] >= 0:
                b_idx = branch_idx[(parent[u], u)]
                P_flow[b_idx] = p_acc
                Q_flow[b_idx] = q_acc

        # Forward sweep
        for u in order:
            for v in children[u]:
                b_idx = branch_idx[(u, v)]
                P = P_flow[b_idx]
                Q = Q_flow[b_idx]
                dV = (r[b_idx] * P + x[b_idx] * Q) / (V[u] + 1e-8)
                dV_perp = (x[b_idx] * P - r[b_idx] * Q) / (V[u] + 1e-8)
                V_new = V[u] - dV
                # Clamp to prevent collapse
                V[v] = max(0.85, min(1.15, V_new))
                theta[v] = theta[u] - dV_perp / (V[u] * V[v] + 1e-8)

    return V, theta, P_flow, Q_flow


def generate_load_profiles(n_buses, n_steps, base_p, base_q, noise_scale=0.15):
    """
    Generate realistic time-varying load profiles.

    Uses sinusoidal patterns to mimic daily load variation,
    plus random noise for individual bus variability.

    Args:
        n_buses: number of buses
        n_steps: number of time steps
        base_p: (n_buses,) base active power demand
        base_q: (n_buses,) base reactive power demand
        noise_scale: standard deviation of multiplicative noise

    Returns:
        P: (n_buses, n_steps) active power demand
        Q: (n_buses, n_steps) reactive power demand
    """
    t = np.linspace(0, 2 * np.pi, n_steps)

    # Global load pattern: peak at morning and evening
    global_pattern = (
        0.5 + 0.15 * np.sin(t - np.pi / 4)   # morning peak
        + 0.2 * np.sin(2 * t + np.pi / 3)    # evening peak
        + 0.05 * np.cos(3 * t)               # minor variation
    )
    global_pattern = np.clip(global_pattern / global_pattern.max() * 1.2, 0.1, 1.0)

    # Individual bus variations
    n_buses_non_root = n_buses - 1
    P = np.zeros((n_buses, n_steps))
    Q = np.zeros((n_buses, n_steps))

    for i in range(1, n_buses):
        # Each bus has its own random phase shift and amplitude
        phase_shift = np.random.uniform(0, 2 * np.pi)
        amp_mod = np.random.uniform(0.7, 1.3)
        bus_pattern = amp_mod * np.interp(
            (t + phase_shift) % (2 * np.pi),
            t,
            global_pattern
        )
        noise = 1 + np.random.normal(0, noise_scale, n_steps)
        P[i] = base_p[i-1] * bus_pattern * noise
        Q[i] = base_q[i-1] * bus_pattern * noise

    # Substation (bus 0) has zero load, acts as slack
    P[0] = 0
    Q[0] = 0

    return P, Q


def generate_dataset(test_system_name='33', n_steps=500, noise_scale=0.15,
                     observable_ratio=0.3, seed=42, custom_params=None):
    """
    Generate a complete dataset for training DSSE models.

    Args:
        test_system_name: '33', '118', or '1746'
        n_steps: number of time steps to simulate
        noise_scale: load variability
        observable_ratio: fraction of buses with measurements
        seed: random seed
        custom_params: optional parameter dict, overriding the named system

    Returns:
        data_list: list of PyG Data objects, each = one time step
        params: test system parameters dict
    """
    np.random.seed(seed)
    if custom_params is not None:
        params = custom_params
    else:
        params = get_test_system(test_system_name)

    n_buses = params['n_buses']
    parent = params['parent']
    r = params['r_pu']
    x = params['x_pu']
    base_p = params['p_pu']
    base_q = params['q_pu']

    # Build graph structure (same for all time steps)
    edge_index = build_edge_index_from_parent(parent)
    G, B, Ybus_mag = build_admittance_matrix(r, x)

    # Compute normalized Ybus edge weights for physics-informed GCN
    # Use symmetric normalization: D^(-1/2) * W * D^(-1/2)
    edge_attr_raw = build_edge_attr_from_Y(Ybus_mag, edge_index)  # (E, 1)

    # Symmetric normalization
    row, col = edge_index.numpy()
    deg = np.zeros(n_buses)
    for r_idx, w in zip(row, edge_attr_raw.numpy()):
        deg[r_idx] += abs(w) + 1e-8
    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    edge_attr_norm = np.array([
        edge_attr_raw[i, 0] * deg_inv_sqrt[row[i]] * deg_inv_sqrt[col[i]]
        for i in range(len(row))
    ], dtype=np.float32)
    edge_attr = torch.tensor(edge_attr_norm, dtype=torch.float32).unsqueeze(-1)

    # Generate time-varying loads
    P_load, Q_load = generate_load_profiles(n_buses, n_steps, base_p, base_q, noise_scale)

    # Run DistFlow for each time step
    print(f"Running DistFlow for {test_system_name}-bus system ({n_steps} steps)...")
    V_all = np.zeros((n_buses, n_steps))
    theta_all = np.zeros((n_buses, n_steps))

    for t in tqdm(range(n_steps), desc="DistFlow"):
        V, theta, _, _ = run_distflow(parent, r, x, P_load[:, t], Q_load[:, t])
        V_all[:, t] = V
        theta_all[:, t] = theta

    # Select observable buses
    n_obs = max(2, int(n_buses * observable_ratio))
    obs_buses = np.sort(np.random.choice(np.arange(1, n_buses), n_obs - 1, replace=False))
    obs_buses = np.concatenate([[0], obs_buses])  # always include substation

    # Build mask
    obs_mask = np.zeros(n_buses, dtype=bool)
    obs_mask[obs_buses] = True

    # Build node features and targets
    data_list = []
    edge_index_t = edge_index.clone()
    edge_attr_t = edge_attr.clone()

    for t in range(n_steps):
        # Features: [V_meas, P_meas, Q_meas, obs_mask, V_prev]
        # Measurements only at observable buses
        V_meas = np.zeros(n_buses)
        P_meas = np.zeros(n_buses)
        Q_meas = np.zeros(n_buses)

        V_meas[obs_buses] = V_all[obs_buses, t] + np.random.normal(0, 0.001, len(obs_buses))
        P_meas[obs_buses] = P_load[obs_buses, t] + np.random.normal(0, 0.001, len(obs_buses))
        Q_meas[obs_buses] = Q_load[obs_buses, t] + np.random.normal(0, 0.001, len(obs_buses))

        x = np.stack([
            V_meas,
            P_meas,
            Q_meas,
            obs_mask.astype(float),
        ], axis=1)  # (n_buses, 4)

        # Target: full state
        y = np.stack([
            V_all[:, t],
            theta_all[:, t],
            P_load[:, t],
            Q_load[:, t],
        ], axis=1)  # (n_buses, 4)

        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.float32),
            edge_index=edge_index_t,
            edge_attr=edge_attr_t,
            obs_mask=torch.tensor(obs_mask, dtype=torch.bool),
            n_buses=n_buses,
        )
        data_list.append(data)

    return data_list, params


def create_splits(data_list, train_ratio=0.8, val_ratio=0.1):
    """Split data into train/val/test sets."""
    n = len(data_list)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    indices = np.random.permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        'train': [data_list[i] for i in train_idx],
        'val': [data_list[i] for i in val_idx],
        'test': [data_list[i] for i in test_idx],
    }


def save_dataset(splits, test_system_name, data_dir='data'):
    """Save dataset splits to disk."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for split_name, data_list in splits.items():
        path = data_dir / f"{test_system_name}bus_{split_name}.pt"
        torch.save(data_list, path)
        print(f"Saved {len(data_list)} samples to {path}")


def load_dataset(test_system_name, data_dir='data'):
    """Load saved dataset splits."""
    data_dir = Path(data_dir)
    splits = {}
    for split_name in ['train', 'val', 'test']:
        path = data_dir / f"{test_system_name}bus_{split_name}.pt"
        if path.exists():
            splits[split_name] = torch.load(path, weights_only=False)
    return splits if splits else None


# ============ Main script ============

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate DSSE datasets')
    parser.add_argument('--system', type=str, default='33', choices=['33', '118', '1746'],
                        help='Test system name')
    parser.add_argument('--n_steps', type=int, default=500,
                        help='Number of time steps')
    parser.add_argument('--obs_ratio', type=float, default=0.3,
                        help='Observable bus ratio')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Output directory')

    args = parser.parse_args()

    print(f"Generating {args.system}-bus dataset...")
    print(f"  n_steps={args.n_steps}, obs_ratio={args.obs_ratio}")

    data_list, params = generate_dataset(
        test_system_name=args.system,
        n_steps=args.n_steps,
        observable_ratio=args.obs_ratio,
    )

    splits = create_splits(data_list)
    save_dataset(splits, args.system, args.data_dir)

    # Quick stats
    for split_name, dl in splits.items():
        print(f"  {split_name}: {len(dl)} samples")
    print(f"  n_buses={params['n_buses']}, n_edges={data_list[0].edge_index.shape[1]}")
    print("Done!")
