"""
Test system definitions for distribution network state estimation.

Defines 33-bus, 118-bus, and 1746-node test systems.
For OpenDSS-based simulation.
"""

import numpy as np
import os

# ============================================================
# 33-Bus System (Baran & Wu, 1989)
# ============================================================

# Line data: from_bus, to_bus, R (ohm), X (ohm), P_load (kW), Q_load (kvar)
# Base: 12.66 kV, 100 MVA
IEEE33_LINE_DATA = [
    # from, to, R, X, P_load(kW), Q_load(kvar)
    (0, 1, 0.0922, 0.0470, 100, 60),
    (1, 2, 0.4930, 0.2511, 90, 40),
    (2, 3, 0.3660, 0.1864, 120, 80),
    (3, 4, 0.3811, 0.1941, 60, 30),
    (4, 5, 0.8190, 0.7070, 60, 20),
    (5, 6, 0.1872, 0.6188, 200, 100),
    (6, 7, 1.7114, 1.2351, 200, 100),
    (7, 8, 1.0300, 0.7400, 60, 20),
    (8, 9, 1.0400, 0.7400, 60, 20),
    (9, 10, 0.1966, 0.0650, 45, 30),
    (10, 11, 0.3744, 0.1238, 60, 35),
    (11, 12, 1.4680, 1.1550, 60, 35),
    (12, 13, 0.5416, 0.7129, 120, 80),
    (13, 14, 0.5910, 0.5260, 60, 10),
    (14, 15, 0.7463, 0.5450, 60, 20),
    (15, 16, 1.2890, 1.7210, 60, 20),
    (16, 17, 0.7320, 0.5740, 90, 40),
    (1, 18, 0.1640, 0.1565, 90, 40),
    (18, 19, 1.5042, 1.3554, 90, 40),
    (19, 20, 0.4095, 0.4784, 90, 40),
    (20, 21, 0.7089, 0.9373, 90, 40),
    (2, 22, 0.4512, 0.3083, 90, 50),
    (22, 23, 0.8980, 0.7091, 420, 200),
    (23, 24, 0.8960, 0.7011, 420, 200),
    (5, 25, 0.2030, 0.1034, 60, 25),
    (25, 26, 0.2842, 0.1447, 60, 25),
    (26, 27, 1.0590, 0.9337, 60, 20),
    (27, 28, 0.8042, 0.7006, 120, 70),
    (28, 29, 0.5075, 0.2585, 200, 600),
    (29, 30, 0.9744, 0.9630, 150, 70),
    (30, 31, 0.3105, 0.3619, 210, 100),
    (31, 32, 0.3410, 0.5302, 60, 40),
]


def get_ieee33_params():
    """Return IEEE 33-bus system parameters."""
    lines = np.array(IEEE33_LINE_DATA)
    n_buses = 33

    # Build parent array (radial)
    parent = -np.ones(n_buses, dtype=int)
    for row in lines:
        f, t = int(row[0]), int(row[1])
        parent[t] = f

    # Extract line parameters
    r_ohm = lines[:, 2].copy()
    x_ohm = lines[:, 3].copy()
    p_kw = lines[:, 4].copy()
    q_kvar = lines[:, 5].copy()

    # Convert to p.u. (base 100 MVA, 12.66 kV)
    V_base = 12.66  # kV
    S_base = 100.0  # MVA
    Z_base = V_base**2 / S_base

    r_pu = r_ohm / Z_base
    x_pu = x_ohm / Z_base
    p_pu = p_kw / (S_base * 1000)  # kW -> p.u.
    q_pu = q_kvar / (S_base * 1000)

    return {
        'n_buses': n_buses,
        'n_branches': len(lines),
        'parent': parent,
        'r_pu': r_pu,
        'x_pu': x_pu,
        'p_pu': p_pu,
        'q_pu': q_pu,
        'V_base': V_base,
        'S_base': S_base,
        'line_data': lines,
    }


# ============================================================
# 118-Bus System (Synthetic)
# ============================================================

def generate_synthetic_118bus():
    """
    Generate a synthetic 118-bus radial distribution network.

    Uses branching pattern to create a realistic radial topology.
    Returns same format as get_ieee33_params().
    """
    np.random.seed(42)
    n_buses = 118
    parent = -np.ones(n_buses, dtype=int)

    # Build radial topology: each node connects to a previous node
    # Main feeder: 0-1-2-...-39 (40 buses)
    for i in range(1, 40):
        parent[i] = i - 1

    # Laterals: each connected to a main-feeder bus
    branch_starts = [2, 5, 8, 12, 15, 18, 22, 25, 28, 32, 35, 38]
    next_bus = 40
    for main_bus in branch_starts:
        branch_len = np.random.randint(4, 10)
        prev = main_bus
        for _ in range(branch_len):
            if next_bus < n_buses:
                parent[next_bus] = prev
                prev = next_bus
                next_bus += 1

    # Fill remaining buses
    while next_bus < n_buses:
        main_bus = np.random.randint(1, 40)
        parent[next_bus] = main_bus
        next_bus += 1

    n_branches = n_buses - 1
    # Generate realistic line parameters (scaled down for larger systems)
    # Use lower impedance to prevent voltage collapse
    r_pu = np.random.uniform(0.0005, 0.008, n_branches)
    x_pu = np.random.uniform(0.0003, 0.006, n_branches)
    # Loads: scale each bus to have ~1-2% of total load, with total P ≈ 0.1-0.2 p.u.
    p_pu = np.random.uniform(0.0002, 0.003, n_buses)
    q_pu = np.random.uniform(0.0001, 0.002, n_buses)

    return {
        'n_buses': n_buses,
        'n_branches': n_branches,
        'parent': parent,
        'r_pu': r_pu,
        'x_pu': x_pu,
        'p_pu': p_pu,
        'q_pu': q_pu,
        'V_base': 12.66,
        'S_base': 100.0,
    }


# ============================================================
# 1746-Node System (Synthetic, scaled from smaller topology)
# ============================================================

def generate_synthetic_1746node():
    """
    Generate a synthetic 1746-node radial distribution network.

    Creates a realistic large-scale distribution system by recursively
    branching from a main feeder.
    """
    np.random.seed(123)
    n_buses = 1746
    parent = -np.ones(n_buses, dtype=int)

    # Main feeder: 100 buses in a chain
    main_feeder_len = 100
    for i in range(1, main_feeder_len):
        parent[i] = i - 1

    # Recursively add branches from main feeder
    next_bus = main_feeder_len
    branch_points = list(range(5, main_feeder_len - 5, 5))

    for main_bus in branch_points:
        if next_bus >= n_buses:
            break
        # Each branch point spawns a sub-feeder
        sub_len = min(np.random.randint(10, 50), n_buses - next_bus)
        prev = main_bus
        for _ in range(sub_len):
            parent[next_bus] = prev
            prev = next_bus
            next_bus += 1
            if next_bus >= n_buses:
                break

        # Add sub-laterals from this sub-feeder
        if next_bus < n_buses:
            for sub_point in np.random.choice(range(prev - sub_len + 1, prev), min(3, max(1, sub_len // 10)), replace=False):
                if next_bus >= n_buses:
                    break
                sub_sub_len = min(np.random.randint(5, 15), n_buses - next_bus)
                sp = sub_point
                for _ in range(sub_sub_len):
                    parent[next_bus] = sp
                    sp = next_bus
                    next_bus += 1
                    if next_bus >= n_buses:
                        break

    # Fill any remaining buses
    while next_bus < n_buses:
        p = np.random.randint(1, next_bus)
        parent[next_bus] = p
        next_bus += 1

    n_branches = n_buses - 1
    r_pu = np.random.uniform(0.0001, 0.003, n_branches)
    x_pu = np.random.uniform(0.0001, 0.002, n_branches)
    p_pu = np.random.uniform(0.00005, 0.0005, n_buses)
    q_pu = np.random.uniform(0.00002, 0.0003, n_buses)

    return {
        'n_buses': n_buses,
        'n_branches': n_branches,
        'parent': parent,
        'r_pu': r_pu,
        'x_pu': x_pu,
        'p_pu': p_pu,
        'q_pu': q_pu,
        'V_base': 12.66,
        'S_base': 100.0,
    }


# ============================================================
# Registry
# ============================================================

TEST_SYSTEMS = {
    '33': get_ieee33_params,
    '118': generate_synthetic_118bus,
    '1746': generate_synthetic_1746node,
}


def get_ieee123bw_params():
    """IEEE 123-bus topology + FeederBW real load profiles."""
    from .feederbw_loader import build_ieee123_topology
    return build_ieee123_topology()


# Register 123bw (uses IEEE 123 topology; loads from FeederBW in data_generator)
TEST_SYSTEMS['123bw'] = get_ieee123bw_params


def get_test_system(name: str) -> dict:
    """Get test system parameters by name."""
    if name not in TEST_SYSTEMS:
        raise ValueError(f"Unknown test system: {name}. Available: {list(TEST_SYSTEMS.keys())}")
    return TEST_SYSTEMS[name]()
