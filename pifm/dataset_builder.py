"""
Synthetic dataset builder for the PI-FM experiments.

Extracted verbatim from the original training script so that the model modules
do not have to import a training script in order to build data. The function
body is unchanged; only the module it lives in differs.
"""
import numpy as np
import torch
from tqdm import tqdm

def generate_paper_dataset(test_system_name='33', n_steps=500, meas_rate=0.6,
                           noise_std=0.001, seed=42):
    """
    Generate dataset matching the paper's setup:
    - Input: P_inj, Q_inj at all buses + P_flow, Q_flow at 30% branches
    - Output: Vm, Va
    - Measurement noise: Gaussian, noise_std default 0.1% std
    """
    from .test_systems import get_test_system
    from .data_generator import generate_load_profiles, run_distflow

    np.random.seed(seed)
    params = get_test_system(test_system_name)
    n_buses = params['n_buses']
    parent = params['parent']
    r = params['r_pu']
    x = params['x_pu']

    # Generate loads
    P_load, Q_load = generate_load_profiles(n_buses, n_steps, params['p_pu'], params['q_pu'])

    edge_index = build_edge_index_from_parent(parent)
    G, B, Ybus_mag = build_admittance_matrix(r, x)

    # Normalize Ybus for edge_attr
    row, col = edge_index.numpy()
    deg = np.zeros(n_buses)
    for ri, w in zip(row, Ybus_mag[row, col]):
        deg[ri] += abs(w) + 1e-8
    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    n_edges = len(row)
    edge_attr_arr = np.array([
        Ybus_mag[row[i], col[i]] * deg_inv_sqrt[row[i]] * deg_inv_sqrt[col[i]]
        for i in range(n_edges)
    ], dtype=np.float32)
    edge_attr = torch.tensor(edge_attr_arr, dtype=torch.float32).unsqueeze(-1)

    # Run DistFlow
    from torch_geometric.data import Data
    data_list = []

    for t in tqdm(range(n_steps), desc=f'Gen {test_system_name}-node'):
        V, theta, P_f, Q_f = run_distflow(parent, r, x, P_load[:, t], Q_load[:, t])

        # PAPER INPUT: P_inj + Q_inj + 30% line P_flow + 30% line Q_flow
        n_branches = len(r)
        n_line_meas = max(1, int(0.3 * n_branches))
        line_idx = np.random.choice(n_branches, n_line_meas, replace=False)

        # Build measurement vector
        x_feat = np.zeros((n_buses, 4), dtype=np.float32)
        # All buses: P/Q injection
        x_feat[:, 0] = P_load[:, t]
        x_feat[:, 1] = Q_load[:, t]
        # 30% lines: P_flow, Q_flow (place on first bus of each line)
        x_feat[:, 2] = 0  # will be filled for 30% of buses
        x_feat[:, 3] = 0

        # Map line flows to nodes
        for li in line_idx:
            child = li + 1  # branch i connects parent[child] to child
            if child < n_buses:
                x_feat[child, 2] = P_f[li]
                x_feat[child, 3] = Q_f[li]

        # Add measurement noise (paper: 0.1% std default)
        x_feat += np.random.normal(0, noise_std * np.abs(x_feat).clip(1e-6), x_feat.shape)

        # Measurement mask: 60% of nodes have real measurements
        n_meas = max(2, int(meas_rate * n_buses))
        meas_idx = np.random.choice(np.arange(1, n_buses), n_meas - 1, replace=False)
        meas_idx = np.concatenate([[0], meas_idx])
        meas_mask = np.zeros(n_buses, dtype=bool)
        meas_mask[meas_idx] = True

        # Unmeasured nodes: set injection to 0
        x_feat[~meas_mask, 0] = 0
        x_feat[~meas_mask, 1] = 0

        y = np.stack([V, theta], axis=1).astype(np.float32)  # (N, 2)

        data = Data(
            x=torch.tensor(x_feat, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.float32),
            edge_index=edge_index.clone(),
            edge_attr=edge_attr.clone(),
            obs_mask=torch.tensor(meas_mask, dtype=torch.bool),
            n_buses=n_buses,
        )
        data_list.append(data)

    return data_list, params
