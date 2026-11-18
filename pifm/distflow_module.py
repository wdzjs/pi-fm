"""
DistFlow coarse estimation module.

Provides an independent, callable DistFlow-based coarse state estimator.
Given measurement vector x (P_inj, Q_inj, P_flow, Q_flow), runs backward-
forward sweep to produce a coarse estimate of (Vm, theta).

Usage:
    from pifm.test_systems import get_test_system
    from pifm.distflow_module import DistFlowCoarse

    params = get_test_system('33')
    df = DistFlowCoarse(params['parent'], params['r_pu'], params['x_pu'])
    y_phys = df(x_tensor)  # (N, 2) = [Vm, theta]
"""

import numpy as np
import torch


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
    assert p_load.ndim == 1 and q_load.ndim == 1, \
        f"p_load ndim={p_load.ndim}, q_load ndim={q_load.ndim}"
    assert p_load.dtype == np.float64, \
        f"p_load dtype={p_load.dtype}"

    for _ in range(max_iter):
        # Backward sweep (iterative post-order, no recursion)
        P_flow = np.zeros(n_branches, dtype=np.float64)
        Q_flow = np.zeros(n_branches, dtype=np.float64)

        # Post-order traversal: process leaves first
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


class DistFlowCoarse:
    """
    DistFlow-based coarse state estimator.

    Given a partial measurement vector x = [P_inj, Q_inj, P_flow, Q_flow],
    runs DistFlow with zero-filled missing injections to produce a coarse
    estimate of the full system state (Vm, theta).

    Attributes:
        parent: (n_buses,) parent array for radial topology
        r: (n_branches,) line resistances (p.u.)
        xline: (n_branches,) line reactances (p.u.)
        n_buses: number of buses
    """

    def __init__(self, parent, r, xline):
        """
        Args:
            parent: (n_buses,) parent array, parent[0] = -1
            r: (n_branches,) line resistances (p.u.)
            xline: (n_branches,) line reactances (p.u.)
        """
        self.parent = np.asarray(parent, dtype=int)
        self.r = np.asarray(r, dtype=np.float64)
        self.xline = np.asarray(xline, dtype=np.float64)
        self.n_buses = len(parent)

    def __call__(self, x, edge_index=None, edge_attr=None, n_buses=None):
        """
        Run DistFlow coarse estimation.

        Args:
            x: (N, 4) measurement vector [P_inj, Q_inj, P_flow, Q_flow].
               Missing injections should already be zero-filled.
               Can be torch.Tensor or np.ndarray.
            edge_index: (2, E) - unused, for API compatibility.
            edge_attr: (E, 1) - unused, for API compatibility.
            n_buses: int - unused (uses self.n_buses).

        Returns:
            y_phys: (N, 2) torch.Tensor, coarse state estimate [Vm, theta].
        """
        # Convert to numpy
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)

        N = x_np.shape[0]
        if N != self.n_buses:
            raise ValueError(
                f"Expected x with {self.n_buses} nodes, got {N}"
            )

        # Extract P_inj and Q_inj (columns 0 and 1)
        p_load = x_np[:, 0].astype(np.float64)
        q_load = x_np[:, 1].astype(np.float64)

        # Run DistFlow
        V, theta, _, _ = run_distflow(
            self.parent, self.r, self.xline,
            p_load, q_load, v0=1.0, max_iter=10
        )

        # Return as torch tensor (N, 2)
        y_phys = np.stack([V, theta], axis=1).astype(np.float32)
        # If input was on GPU, move result to same device
        if isinstance(x, torch.Tensor):
            return torch.tensor(y_phys, dtype=torch.float32, device=x.device)
        return torch.tensor(y_phys, dtype=torch.float32)

    def to(self, device):
        """No-op for API compatibility (DistFlow runs on CPU)."""
        return self


def distflow_coarse(x, edge_index, edge_attr, n_buses):
    """
    Standalone function for DistFlow coarse estimation.

    Requires that the system parameters are available globally or
    that a DistFlowCoarse instance has been created beforehand.
    This is a thin wrapper — prefer using DistFlowCoarse class directly
    for production use.

    Args:
        x: (N, 4) measurement vector [P_inj, Q_inj, P_flow, Q_flow].
        edge_index: (2, E) graph connectivity.
        edge_attr: (E, 1) normalized Ybus edge weights.
        n_buses: number of buses.

    Returns:
        y_phys: (N, 2) coarse state estimate [Vm, theta].

    Note:
        This function requires _GLOBAL_DISTFLOW to be set via
        set_distflow_instance() before calling.
    """
    if _GLOBAL_DISTFLOW is None:
        raise RuntimeError(
            "Global DistFlowCoarse instance not set. "
            "Call set_distflow_instance(df) first, or use DistFlowCoarse directly."
        )
    return _GLOBAL_DISTFLOW(x, edge_index, edge_attr, n_buses)


_GLOBAL_DISTFLOW = None


def set_distflow_instance(df):
    """Set the global DistFlowCoarse instance for use by distflow_coarse()."""
    global _GLOBAL_DISTFLOW
    _GLOBAL_DISTFLOW = df
