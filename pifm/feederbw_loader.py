"""
FeederBW real load data loader + IEEE 123-bus topology builder.

Combines:
1. IEEE 123-bus topology from Excel data (standard test feeder)
2. FeederBW real load profiles (40 feeders, 2 years, 1-min resolution)
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch_geometric.data import Data

from .data_generator import run_distflow, generate_load_profiles
from .utils import build_edge_index_from_parent


# ============================================================
# ACSR conductor impedances (approx. p.u. at 100 MVA, 4.16 kV)
# R_base = V^2 / S = 4.16^2 / 100 = 0.173 ohm
# X_base = same
# ============================================================

# Resistance (ohm/mile) and Reactance (ohm/mile) for ACSR conductors
ACSR_OHM_PER_MILE = {
    # Config #: (R ohm/mile, X ohm/mile)
    1:  (0.306, 0.472),   # 336,400 ACSR, 3-phase
    2:  (0.306, 0.472),
    3:  (0.306, 0.472),
    4:  (0.306, 0.472),
    5:  (0.306, 0.472),
    6:  (0.306, 0.472),
    7:  (0.306, 0.472),   # 2-phase
    8:  (0.306, 0.472),
    9:  (0.607, 0.528),   # 1/0 ACSR, 1-phase
    10: (0.607, 0.528),
    11: (0.607, 0.528),
}

# Base values for IEEE 123-bus
V_BASE = 4.16  # kV
S_BASE = 100.0  # MVA
Z_BASE = V_BASE ** 2 / S_BASE  # = 0.173 ohm


def build_ieee123_topology(data_dir='data/IEEE123/feeder123'):
    """Parse IEEE 123-bus Excel data and build topology params."""
    # Read line data
    lines = pd.read_excel(f'{data_dir}/line data.xls', header=None)
    lines = lines.iloc[3:]  # skip headers
    lines.columns = ['node_a', 'node_b', 'length_ft', 'config']
    lines = lines.dropna(subset=['node_a', 'node_b'])

    # Convert to 0-based indexing
    node_a = lines['node_a'].astype(int).values - 1
    node_b = lines['node_b'].astype(int).values - 1
    length_ft = lines['length_ft'].astype(float).values
    config = lines['config'].astype(int).values

    # Compute R and X in p.u.
    r_pu = np.zeros(len(length_ft))
    x_pu = np.zeros(len(length_ft))
    for i, cfg in enumerate(config):
        r_ohm_mi, x_ohm_mi = ACSR_OHM_PER_MILE.get(cfg, (0.5, 0.5))
        # Convert feet to miles, then to p.u.
        length_mi = length_ft[i] / 5280.0
        r_ohm = r_ohm_mi * length_mi
        x_ohm = x_ohm_mi * length_mi
        r_pu[i] = r_ohm / Z_BASE
        x_pu[i] = x_ohm / Z_BASE

    # === Add normally-closed switches as zero-impedance lines ===
    # IEEE 123 has switches connecting feeder sections; normally-closed ones
    # are effectively lines with negligible impedance.
    try:
        sw_data = pd.read_excel(f'{data_dir}/switch data.xls', header=None)
        sw_data = sw_data.iloc[3:]  # skip headers
        for _, row in sw_data.iterrows():
            try:
                sw_a = int(row.iloc[0]) - 1
                sw_b = int(row.iloc[1]) - 1
                sw_status = str(row.iloc[2]).strip().lower() if pd.notna(row.iloc[2]) else 'open'
                if sw_status == 'closed':
                    node_a = np.append(node_a, sw_a)
                    node_b = np.append(node_b, sw_b)
                    r_pu = np.append(r_pu, 1e-6)  # negligible impedance
                    x_pu = np.append(x_pu, 1e-6)
            except (ValueError, TypeError, IndexError):
                continue
    except Exception:
        pass  # switch data file not found or unreadable

    # Remap sparse bus IDs to dense 0-based indexing
    all_buses = set(node_a) | set(node_b)
    bus_map = {b: i for i, b in enumerate(sorted(all_buses))}
    n_buses = len(all_buses)

    # Remap node connections
    node_a_mapped = np.array([bus_map[a] for a in node_a])
    node_b_mapped = np.array([bus_map[b] for b in node_b])

    # Build undirected adjacency for BFS
    adj = [[] for _ in range(n_buses)]
    for a, b in zip(node_a_mapped, node_b_mapped):
        adj[a].append(b)
        adj[b].append(a)  # undirected!

    # Build parent array via BFS from root (bus 0 = substation, find the actual root)
    parent = -np.ones(n_buses, dtype=int)
    # Find root: the bus that appears in node_a but not in node_b
    from_buses = set(node_b_mapped)
    to_buses = set(node_a_mapped)
    root_candidates = to_buses - from_buses
    root = min(root_candidates) if root_candidates else 0

    visited = np.zeros(n_buses, dtype=bool)
    queue = [root]
    visited[root] = True
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                queue.append(v)

    # Build branch index: branch i connects node_b_mapped[i] to node_a_mapped[i]
    branch_idx = {}
    for i, (a, b) in enumerate(zip(node_a_mapped, node_b_mapped)):
        branch_idx[(a, b)] = i

    # === Re-index r_pu/x_pu to match new BFS bus ordering ===
    # DistFlow assumes r[i-1] and x[i-1] correspond to branch parent[i]→i
    r_pu_new = np.zeros(n_buses - 1)
    x_pu_new = np.zeros(n_buses - 1)
    for child in range(1, n_buses):
        p = parent[child]
        orig_branch = branch_idx.get((p, child))
        if orig_branch is None:
            # Try reverse direction
            orig_branch = branch_idx.get((child, p))
        if orig_branch is not None:
            r_pu_new[child - 1] = r_pu[orig_branch]
            x_pu_new[child - 1] = x_pu[orig_branch]
        else:
            # Fallback: use average values
            r_pu_new[child - 1] = r_pu.mean()
            x_pu_new[child - 1] = x_pu.mean()
    r_pu = r_pu_new
    x_pu = x_pu_new

    # Spot loads (optional: read from Excel, otherwise use zeros)
    try:
        loads = pd.read_excel(f'{data_dir}/spot loads data.xls', header=None)
        print(f"  Spot loads: {loads.shape}")
    except Exception:
        loads = None

    # Assign base load per bus (remapped)
    p_pu = np.zeros(n_buses)
    q_pu = np.zeros(n_buses)

    if loads is not None:
        try:
            # Data starts at row 4 (rows 0-3 are headers).
            # Col 0: Node, Col 1: Model (e.g. "Y-PQ"), Cols 2/4/6: kW per phase, Cols 3/5/7: kVAr per phase
            load_data = loads.iloc[4:]
            for _, row in load_data.iterrows():
                try:
                    orig_bus = int(row.iloc[0]) - 1
                except (ValueError, TypeError):
                    continue
                # Sum three-phase loads (columns 2/4/6 for kW, 3/5/7 for kVAr)
                p_kw = 0.0
                q_kvar = 0.0
                for col_idx in [2, 4, 6]:
                    if pd.notna(row.iloc[col_idx]) if col_idx < len(row) else False:
                        try:
                            p_kw += float(row.iloc[col_idx])
                        except (ValueError, TypeError):
                            pass
                for col_idx in [3, 5, 7]:
                    if pd.notna(row.iloc[col_idx]) if col_idx < len(row) else False:
                        try:
                            q_kvar += float(row.iloc[col_idx])
                        except (ValueError, TypeError):
                            pass
                if orig_bus in bus_map:
                    bus = bus_map[orig_bus]
                    p_pu[bus] = p_kw / (S_BASE * 1000)
                    q_pu[bus] = q_kvar / (S_BASE * 1000)
        except Exception:
            pass
    else:
        p_pu[1:] = 0.002
        q_pu[1:] = 0.001

    # === Re-root topology so bus 0 is the root (DistFlow requirement) ===
    root = int(np.where(parent == -1)[0][0]) if (parent == -1).any() else 0
    if root != 0:
        # BFS from root to get new ordering
        new_order = [-1] * n_buses
        queue = [root]
        idx = 0
        while queue:
            u = queue.pop(0)
            new_order[u] = idx
            idx += 1
            for v in range(n_buses):
                if parent[v] == u:
                    queue.append(v)

        # Build remapped arrays
        old_to_new = new_order
        parent_new = -np.ones(n_buses, dtype=int)
        p_pu_new = np.zeros(n_buses)
        q_pu_new = np.zeros(n_buses)
        r_pu_new = np.zeros(n_buses - 1)
        x_pu_new = np.zeros(n_buses - 1)

        for old_bus in range(n_buses):
            new_bus = old_to_new[old_bus]
            if new_bus < 0:
                continue
            p_pu_new[new_bus] = p_pu[old_bus]
            q_pu_new[new_bus] = q_pu[old_bus]
            if parent[old_bus] >= 0:
                parent_new[new_bus] = old_to_new[parent[old_bus]]
                branch_idx_old = old_bus - 1  # old branch index
                if branch_idx_old < len(r_pu):
                    r_pu_new[new_bus - 1] = r_pu[branch_idx_old]
                    x_pu_new[new_bus - 1] = x_pu[branch_idx_old]

        parent = parent_new
        p_pu = p_pu_new
        q_pu = q_pu_new
        r_pu = r_pu_new
        x_pu = x_pu_new

    return {
        'n_buses': n_buses,
        'n_branches': len(r_pu),
        'parent': parent,
        'r_pu': r_pu,
        'x_pu': x_pu,
        'p_pu': p_pu,
        'q_pu': q_pu,
        'V_base': V_BASE,
        'S_base': S_BASE,
    }


def load_feederbw_profiles(feeder_ids=None, n_feeder_samples=123,
                           data_dir='data/FeederBW/extracted/feeder_measurement_data_001-040'):
    """
    Load FeederBW load profiles and sample for use as node loads.

    Args:
        feeder_ids: list of feeder indices to load (default: 1-40)
        n_feeder_samples: number of time series to extract (for n_nodes)
        data_dir: path to FeederBW parquet files

    Returns:
        P_profiles: (n_nodes, n_steps) real power in p.u.
        Q_profiles: (n_nodes, n_steps) reactive power in p.u.
    """
    import glob
    import os

    files = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))
    print(f"  Loading {len(files)} FeederBW files...")

    # Load first few feeders to get time series
    all_P = []
    all_Q = []
    n_files = min(len(files), 40)

    for i in range(n_files):
        df = pd.read_parquet(files[i])
        # Extract P and Q, downsample to hourly for manageability
        df_resampled = df.iloc[::60]  # 1-min → 1-hour

        P_kw = df_resampled['active_power_kW'].values.astype(float)
        Q_kvar = df_resampled['reactive_power_kVar'].values.astype(float)

        # Normalize to p.u. (base: S_BASE=100 MVA)
        P_pu = P_kw / (S_BASE * 1000)
        Q_pu = Q_kvar / (S_BASE * 1000)

        all_P.append(P_pu)
        all_Q.append(Q_pu)

    # Stack and sample for n_nodes
    n_steps_total = min(len(p) for p in all_P)
    n_nodes = n_feeder_samples

    P_profiles = np.zeros((n_nodes, n_steps_total))
    Q_profiles = np.zeros((n_nodes, n_steps_total))

    # Use each feeder as a "node load profile"
    for i in range(min(n_nodes, len(all_P))):
        P_profiles[i] = all_P[i][:n_steps_total]
        Q_profiles[i] = all_Q[i][:n_steps_total]

    # If we need more nodes than feeders, duplicate with small noise
    for i in range(len(all_P), n_nodes):
        src = i % len(all_P)
        P_profiles[i] = all_P[src][:n_steps_total] * (0.8 + 0.4 * np.random.random())
        Q_profiles[i] = all_Q[src][:n_steps_total] * (0.8 + 0.4 * np.random.random())

    print(f"  FeederBW profiles: {n_nodes} nodes x {n_steps_total} steps")
    return P_profiles, Q_profiles


def generate_feederbw_dataset(topology_params, n_steps=500, obs_ratio=0.3, seed=42):
    """
    Generate dataset using IEEE topology + FeederBW real loads.
    """
    np.random.seed(seed)
    params = topology_params

    n_buses = params['n_buses']
    parent = params['parent']
    r = params['r_pu']
    x = params['x_pu']

    # Build graph structure
    edge_index = build_edge_index_from_parent(parent)

    # Load FeederBW profiles (assign to nodes)
    P_load_raw, Q_load_raw = load_feederbw_profiles(n_feeder_samples=n_buses)
    n_available = P_load_raw.shape[1]
    n_steps = min(n_steps, n_available)

    P_load = P_load_raw[:, :n_steps]
    Q_load = Q_load_raw[:, :n_steps]

    # Normalize edge weights
    from .data_generator import build_admittance_matrix, build_edge_attr_from_Y
    G, B, Ybus_mag = build_admittance_matrix(r, x)
    edge_attr_raw = build_edge_attr_from_Y(Ybus_mag, edge_index)

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

    # Run DistFlow
    print(f"  Running DistFlow ({n_steps} steps)...")
    from tqdm import tqdm
    V_all = np.zeros((n_buses, n_steps))
    theta_all = np.zeros((n_buses, n_steps))

    for t in tqdm(range(n_steps), desc="DistFlow"):
        V, theta, _, _ = run_distflow(parent, r, x, P_load[:, t], Q_load[:, t])
        V_all[:, t] = V
        theta_all[:, t] = theta

    # Select observable buses
    n_obs = max(2, int(n_buses * obs_ratio))
    obs_buses = np.sort(np.random.choice(np.arange(1, n_buses), n_obs - 1, replace=False))
    obs_buses = np.concatenate([[0], obs_buses])
    obs_mask = np.zeros(n_buses, dtype=bool)
    obs_mask[obs_buses] = True

    # Build PyG Data objects
    data_list = []
    for t in range(n_steps):
        V_meas = np.zeros(n_buses)
        P_meas = np.zeros(n_buses)
        Q_meas = np.zeros(n_buses)
        V_meas[obs_buses] = V_all[obs_buses, t] + np.random.normal(0, 0.001, len(obs_buses))
        P_meas[obs_buses] = P_load[obs_buses, t] + np.random.normal(0, 0.001, len(obs_buses))
        Q_meas[obs_buses] = Q_load[obs_buses, t] + np.random.normal(0, 0.001, len(obs_buses))

        x = np.stack([V_meas, P_meas, Q_meas, obs_mask.astype(float)], axis=1)
        y = np.stack([V_all[:, t], theta_all[:, t], P_load[:, t], Q_load[:, t]], axis=1)

        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.float32),
            edge_index=edge_index.clone(),
            edge_attr=edge_attr.clone(),
            obs_mask=torch.tensor(obs_mask, dtype=torch.bool),
            n_buses=n_buses,
        )
        data_list.append(data)

    return data_list, params


if __name__ == '__main__':
    print("Building IEEE 123-bus topology...")
    params = build_ieee123_topology()
    print(f"  Topology: {params['n_buses']} buses, {params['n_branches']} branches")

    print("Generating dataset with FeederBW loads...")
    data_list, _ = generate_feederbw_dataset(params, n_steps=500, obs_ratio=0.3)

    from .data_generator import create_splits, save_dataset
    splits = create_splits(data_list)
    save_dataset(splits, '123bw', 'data')  # 123-bus + FeederBW
    print("Done!")
