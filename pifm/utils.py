"""
Utility functions for graph and admittance construction.
"""

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def build_admittance_matrix(r, x):
    """
    Build bus admittance matrix Ybus from line parameters.

    Args:
        r: array of line resistances (p.u.), length n_branches
        x: array of line reactances (p.u.), length n_branches

    Returns:
        G: conductance matrix (n_buses x n_buses)
        B: susceptance matrix (n_buses x n_buses)
        Y: admittance magnitude matrix
    """
    n_buses = len(r) + 1  # radial network: n_branches = n_buses - 1
    G = np.zeros((n_buses, n_buses))
    B = np.zeros((n_buses, n_buses))

    z = r**2 + x**2
    g = r / (z + 1e-12)
    b = -x / (z + 1e-12)

    for i, (gi, bi) in enumerate(zip(g, b)):
        j = i + 1  # child node (assuming sequential ordering)
        G[i, i] += gi
        G[j, j] += gi
        G[i, j] -= gi
        G[j, i] -= gi
        B[i, i] += bi
        B[j, j] += bi
        B[i, j] -= bi
        B[j, i] -= bi

    Y = np.sqrt(G**2 + B**2)
    return G, B, Y


def build_edge_index_from_parent(parent):
    """
    Build PyG edge_index from parent array.

    Args:
        parent: array of length n_buses, parent[i] = idx of parent bus
                (root node has parent[root] = -1 or parent[root] = root)

    Returns:
        edge_index: (2, 2*n_edges) long tensor
        edge_attr: (2*n_edges, 1) float tensor (Y magnitude)
    """
    n_buses = len(parent)
    edges = []
    for i in range(n_buses):
        if parent[i] >= 0 and parent[i] != i:
            edges.append([parent[i], i])
            edges.append([i, parent[i]])  # bidirectional

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def build_edge_attr_from_Y(Y, edge_index):
    """
    Extract edge attributes from admittance matrix for given edge_index.

    Args:
        Y: (n_buses, n_buses) admittance magnitude matrix
        edge_index: (2, n_edges) long tensor

    Returns:
        edge_attr: (n_edges, 1) float tensor
    """
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    vals = Y[src, dst]
    return torch.tensor(vals, dtype=torch.float32).unsqueeze(-1)


def gauss_newton_edge_weight(Ybus, V_est):
    """
    Compute physics-informed edge weights using Gauss-Newton approximation.

    The Gauss-Newton Hessian H ≈ J^T W J, where J relates measurements to states.
    For distribution networks, the edge-wise contribution to the Hessian is
    proportional to the admittance magnitude between buses.

    Args:
        Ybus: (n_buses, n_buses) admittance magnitude matrix
        V_est: (n_buses,) estimated voltage magnitudes (or None for flat start)

    Returns:
        W_edge: (n_edges, 1) physics-informed edge weights
    """
    n_buses = Ybus.shape[0]
    if V_est is None:
        V_est = np.ones(n_buses)

    # Gauss-Newton edge weight: |Y_ij| * |V_i| * |V_j|
    # This approximates the contribution of each line to the measurement Jacobian
    edge_weights = []
    for i in range(n_buses):
        for j in range(i + 1, n_buses):
            if Ybus[i, j] > 1e-10:
                w = Ybus[i, j] * V_est[i] * V_est[j]
                edge_weights.append(w)
    return np.array(edge_weights)


def normalize_adjacency(edge_index, edge_weight, n_nodes):
    """
    Symmetric normalization of adjacency: D^{-1/2} A D^{-1/2}.

    Args:
        edge_index: (2, n_edges)
        edge_weight: (n_edges,)
        n_nodes: int

    Returns:
        norm_edge_weight: (n_edges,)
    """
    row, col = edge_index
    deg = np.zeros(n_nodes)
    for r, w in zip(row.numpy(), edge_weight.numpy()):
        deg[r] += abs(w)

    deg_inv_sqrt = np.power(deg + 1e-8, -0.5)
    norm_weights = []
    for r, c, w in zip(row.numpy(), col.numpy(), edge_weight.numpy()):
        norm_weights.append(w * deg_inv_sqrt[r] * deg_inv_sqrt[c])

    return torch.tensor(norm_weights, dtype=torch.float32)


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
