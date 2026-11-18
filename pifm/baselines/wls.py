"""
WLS state estimation baselines, Tier A (DistFlow-WLS) and Tier B (GN-WLS).

Two classical estimators, used as the non-learned reference.

Tier A: DistFlow-WLS -- WLS with DistFlow-initialized state.
  - Initialize V, theta from DistFlow backward-forward sweep
  - Gauss-Newton refinement via scipy.optimize.least_squares (LM, 20 evals)
  - Represents physics-informed initialization approach

Tier B: GN-WLS -- Standard Gauss-Newton Weighted Least Squares.
  - Initialize from flat start (V=1, theta=0)
  - Full LM optimization via scipy.optimize.least_squares (100 evals)
  - Represents the classical DSSE approach

Both use the same AC measurement model. Key design decisions:
  - Slack bus (bus 0) excluded from P_inj/Q_inj measurements
    (it has P_load=0, not a meaningful injection measurement)
  - Branch flow measurements included for measured branches only
  - Solver: scipy.optimize.least_squares with 'lm' method

Usage:
    python -m pifm.baselines.wls --system 33 --method both
    python -m pifm.baselines.wls --system 123bw --method both
"""

import numpy as np
from scipy.optimize import least_squares
from pathlib import Path
import json
import argparse
import time
from tqdm import tqdm

from ..test_systems import get_test_system
from ..data_generator import load_dataset, create_splits
from ..distflow_module import run_distflow
from ..dataset_builder import generate_paper_dataset


# ============================================================
# Ybus Construction (uses parent array, correct for any BFS order)
# ============================================================

def build_ybus(parent, r_pu, x_pu):
    """Build G + jB admittance matrix from parent array."""
    N = len(parent)
    G = np.zeros((N, N), dtype=np.float64)
    B = np.zeros((N, N), dtype=np.float64)

    for child in range(1, N):
        p = int(parent[child])
        if p < 0:
            continue
        br = child - 1
        r = float(r_pu[br])
        x = float(x_pu[br])
        z2 = r * r + x * x + 1e-12
        g = r / z2
        b_val = -x / z2

        G[p, p] += g
        G[child, child] += g
        G[p, child] -= g
        G[child, p] -= g

        B[p, p] += b_val
        B[child, child] += b_val
        B[p, child] -= b_val
        B[child, p] -= b_val

    return G, B


# ============================================================
# AC Measurement Function h(x): (V, theta) -> [P_inj, Q_inj, P_flow, Q_flow]
# ============================================================

def compute_measurements(V, theta, G, B, parent):
    """
    Compute measurement predictions from state (V, theta) using AC equations.

    Returns:
        P_inj: (N,) nodal active power injections (p.u.)
        Q_inj: (N,) nodal reactive power injections (p.u.)
        P_flow: (N-1,) branch active power flows from parent to child (p.u.)
        Q_flow: (N-1,) branch reactive power flows from parent to child (p.u.)
    """
    N = len(V)
    theta_diff = theta[:, None] - theta[None, :]
    cos_d = np.cos(theta_diff)
    sin_d = np.sin(theta_diff)
    VV = V[:, None] * V[None, :]

    P_inj = np.sum(VV * (G * cos_d + B * sin_d), axis=1)
    Q_inj = np.sum(VV * (G * sin_d - B * cos_d), axis=1)

    N_br = N - 1
    P_flow = np.zeros(N_br, dtype=np.float64)
    Q_flow = np.zeros(N_br, dtype=np.float64)

    for child in range(1, N):
        p = int(parent[child])
        if p < 0:
            continue
        br = child - 1
        d = theta[p] - theta[child]
        cd = np.cos(d)
        sd = np.sin(d)

        g_ij = -G[p, child]
        b_ij = -B[p, child]
        Vp, Vc = V[p], V[child]

        P_flow[br] = Vp * Vp * g_ij - Vp * Vc * (g_ij * cd + b_ij * sd)
        Q_flow[br] = -Vp * Vp * b_ij - Vp * Vc * (g_ij * sd - b_ij * cd)

    return P_inj, Q_inj, P_flow, Q_flow


# ============================================================
# Measurement Extraction (v3-format -> WLS measurement vector)
# ============================================================

def extract_measurements(data):
    """
    Extract measurements from v3-format PyG Data object.

    Sign convention: data stores P_load (positive=consumption), but the AC
    measurement function h(x) computes P_inj = P_gen - P_load (negative at
    load buses). We NEGATE P_load to get P_inj for the measurement model.

    Slack bus (bus 0): P_load=0 is NOT an informative measurement of P_inj
    (which equals total network load at the slack). We exclude it.

    Returns:
        meas_indices: dict mapping 'P_inj','Q_inj','P_flow','Q_flow' to index lists
        z: (n_meas,) measurement values in AC convention
    """
    N = data.x.shape[0]
    x_np = data.x.cpu().numpy().astype(np.float64)
    obs_mask = data.obs_mask.cpu().numpy()

    meas_indices = {'P_inj': [], 'Q_inj': [], 'P_flow': [], 'Q_flow': []}
    z_list = []

    # P_inj / Q_inj at observed buses (EXCLUDE slack bus 0)
    for i in range(1, N):  # skip slack bus
        if obs_mask[i]:
            meas_indices['P_inj'].append(i)
            z_list.append(-float(x_np[i, 0]))  # P_inj = -P_load
            meas_indices['Q_inj'].append(i)
            z_list.append(-float(x_np[i, 1]))  # Q_inj = -Q_load

    # P_flow / Q_flow at measured branches (stored on child bus)
    for i in range(1, N):
        if abs(x_np[i, 2]) > 1e-12:
            br = i - 1
            meas_indices['P_flow'].append(br)
            z_list.append(float(x_np[i, 2]))
        if abs(x_np[i, 3]) > 1e-12:
            br = i - 1
            meas_indices['Q_flow'].append(br)
            z_list.append(float(x_np[i, 3]))

    return meas_indices, np.array(z_list, dtype=np.float64)


# ============================================================
# WLS via scipy.optimize.least_squares
# ============================================================

def state_to_Vtheta(x, N):
    """Convert flattened state x to V and theta arrays."""
    V = np.ones(N, dtype=np.float64)
    theta = np.zeros(N, dtype=np.float64)
    V[1:] = x[N-1:]
    theta[1:] = x[:N-1]
    return V, theta


def Vtheta_to_state(V, theta):
    """Convert V, theta to flattened state (excluding slack bus)."""
    N = len(V)
    x = np.zeros(2 * (N - 1), dtype=np.float64)
    x[:N-1] = theta[1:]
    x[N-1:] = V[1:]
    return x


def residual_fun(x, meas_indices, z, G, B, parent):
    """
    Residual function for scipy least_squares: f(x) = z - h(x).
    """
    N = len(parent)
    V, theta = state_to_Vtheta(x, N)
    P_inj, Q_inj, P_flow, Q_flow = compute_measurements(V, theta, G, B, parent)

    r = np.zeros(len(z), dtype=np.float64)
    z_idx = 0
    for mtype in ['P_inj', 'Q_inj', 'P_flow', 'Q_flow']:
        for idx in meas_indices[mtype]:
            if mtype == 'P_inj':
                pred = P_inj[idx]
            elif mtype == 'Q_inj':
                pred = Q_inj[idx]
            elif mtype == 'P_flow':
                pred = P_flow[idx]
            else:
                pred = Q_flow[idx]
            r[z_idx] = z[z_idx] - pred
            z_idx += 1
    return r


def solve_wls(V_init, theta_init, G, B, parent, meas_indices, z,
              max_nfev=50, verbose=False):
    """
    Solve WLS state estimation using scipy.optimize.least_squares (LM method).

    Args:
        V_init, theta_init: initial state guess
        G, B: admittance matrices
        parent: parent array
        meas_indices: measurement indices dict
        z: measurement vector
        max_nfev: max function evaluations
        verbose: print convergence info

    Returns:
        V_est, theta_est: estimated state
        info: dict with n_iter, converged, cost, time_ms
    """
    N = len(V_init)
    x0 = Vtheta_to_state(V_init, theta_init)

    t0 = time.time()

    result = least_squares(
        residual_fun,
        x0,
        method='lm',
        max_nfev=max_nfev,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        args=(meas_indices, z, G, B, parent),
        verbose=0,
    )

    elapsed = time.time() - t0

    V_est, theta_est = state_to_Vtheta(result.x, N)

    # Clamp to valid range
    V_est = np.clip(V_est, 0.85, 1.15)
    theta_est = np.clip(theta_est, -0.5, 0.5)

    return V_est, theta_est, {
        'n_iter': result.nfev,
        'converged': result.success,
        'cost': float(2.0 * result.cost),
        'time_ms': elapsed * 1000,
        'message': str(result.message),
    }


# ============================================================
# DistFlow Initialization
# ============================================================

def distflow_initialize(params, meas_indices, z):
    """
    Initialize state using DistFlow with measured loads (zero-fill unmeasured).
    """
    parent = params['parent']
    N = len(parent)

    p_load = np.zeros(N, dtype=np.float64)
    q_load = np.zeros(N, dtype=np.float64)

    z_idx = 0
    for mtype in ['P_inj', 'Q_inj', 'P_flow', 'Q_flow']:
        for idx in meas_indices[mtype]:
            if mtype == 'P_inj':
                p_load[idx] = -float(z[z_idx])  # P_inj -> P_load
            elif mtype == 'Q_inj':
                q_load[idx] = -float(z[z_idx])  # Q_inj -> Q_load
            z_idx += 1

    V, theta, _, _ = run_distflow(
        parent, params['r_pu'], params['x_pu'],
        p_load, q_load, v0=1.0, max_iter=10
    )
    return V.astype(np.float64), theta.astype(np.float64)


# ============================================================
# Tier A: DistFlow-WLS
# ============================================================

def distflow_wls_estimate(params, meas_indices, z, max_nfev=20, verbose=False):
    """
    Tier A: DistFlow-WLS -- DistFlow initialization + WLS refinement.
    """
    t0 = time.time()
    G, B = build_ybus(params['parent'], params['r_pu'], params['x_pu'])

    # DistFlow initialization
    V_init, theta_init = distflow_initialize(params, meas_indices, z)

    if verbose:
        r0 = residual_fun(Vtheta_to_state(V_init, theta_init),
                          meas_indices, z, G, B, params['parent'])
        print(f"  DistFlow init cost: {0.5 * np.sum(r0**2):.6e}")

    # WLS refinement
    V_est, theta_est, info = solve_wls(
        V_init, theta_init, G, B, params['parent'],
        meas_indices, z, max_nfev=max_nfev, verbose=verbose
    )

    total_time = time.time() - t0
    info['total_time_ms'] = total_time * 1000
    return V_est, theta_est, info


# ============================================================
# Tier B: GN-WLS
# ============================================================

def gnwls_estimate(params, meas_indices, z, max_nfev=100, verbose=False):
    """
    Tier B: GN-WLS -- flat start + full WLS optimization.
    """
    t0 = time.time()
    G, B = build_ybus(params['parent'], params['r_pu'], params['x_pu'])
    N = len(params['parent'])

    # Flat start
    V_init = np.ones(N, dtype=np.float64)
    theta_init = np.zeros(N, dtype=np.float64)

    if verbose:
        r0 = residual_fun(Vtheta_to_state(V_init, theta_init),
                          meas_indices, z, G, B, params['parent'])
        print(f"  Flat start cost: {0.5 * np.sum(r0**2):.6e}")

    # WLS optimization
    V_est, theta_est, info = solve_wls(
        V_init, theta_init, G, B, params['parent'],
        meas_indices, z, max_nfev=max_nfev, verbose=verbose
    )

    total_time = time.time() - t0
    info['total_time_ms'] = total_time * 1000
    return V_est, theta_est, info


# ============================================================
# Batch Evaluation
# ============================================================

def evaluate_wls_on_dataset(method, data_list, params, verbose=False):
    """Run WLS on all samples and aggregate metrics."""
    estimate_fn = distflow_wls_estimate if method == 'distflow_wls' else gnwls_estimate

    mae_sum, rmse_sum = 0.0, 0.0
    maeV_sum, maeT_sum = 0.0, 0.0
    n_iter_sum, time_sum = 0.0, 0.0
    n_conv, n_fail = 0, 0
    n_samples = len(data_list)

    for data in tqdm(data_list, desc=f'Eval {method}'):
        try:
            meas_indices, z = extract_measurements(data)
            if len(z) < 2:
                n_fail += 1
                continue

            V_est, theta_est, info = estimate_fn(
                params, meas_indices, z, verbose=verbose
            )

            y_true = data.y.cpu().numpy().astype(np.float64)
            V_true = y_true[:, 0]
            theta_true = y_true[:, 1]

            err_V = np.abs(V_est - V_true)
            err_T = np.abs(theta_est - theta_true)
            err_T = np.minimum(err_T, 2.0 * np.pi - err_T)

            all_err = np.concatenate([err_V, err_T])
            mae_sum += np.mean(all_err)
            rmse_sum += np.mean(all_err ** 2)
            maeV_sum += np.mean(err_V)
            maeT_sum += np.mean(err_T)

            n_iter_sum += info.get('n_iter', 0)
            time_sum += info.get('total_time_ms', info.get('time_ms', 0))
            if info.get('converged', False):
                n_conv += 1

        except Exception as e:
            n_fail += 1
            if verbose:
                print(f"  Failed: {e}")

    n_valid = n_samples - n_fail
    if n_valid == 0:
        return {'error': 'All samples failed', 'n_failed': n_fail}

    return {
        'mae': mae_sum / n_valid,
        'rmse': np.sqrt(rmse_sum / n_valid),
        'mae_V': maeV_sum / n_valid,
        'mae_theta': maeT_sum / n_valid,
        'avg_nfev': n_iter_sum / n_valid,
        'avg_time_ms': time_sum / n_valid,
        'convergence_rate': n_conv / n_valid,
        'n_samples': n_valid,
        'n_failed': n_fail,
    }


# ============================================================
# DistFlow-only baseline (reference)
# ============================================================

def evaluate_distflow_only(data_list, params):
    """DistFlow coarse estimate (zero-fill unmeasured)."""
    N = params['n_buses']
    parent = params['parent']
    r_pu = params['r_pu']
    x_pu = params['x_pu']

    mae_sum, rmse_sum = 0.0, 0.0
    n_samples = len(data_list)

    for data in data_list:
        x_np = data.x.cpu().numpy().astype(np.float64)
        p_load = x_np[:, 0].copy()
        q_load = x_np[:, 1].copy()

        V, theta, _, _ = run_distflow(parent, r_pu, x_pu, p_load, q_load,
                                       v0=1.0, max_iter=10)
        V = V.astype(np.float64)
        theta = theta.astype(np.float64)

        y_true = data.y.cpu().numpy().astype(np.float64)
        V_true = y_true[:, 0]
        theta_true = y_true[:, 1]

        err_V = np.abs(V - V_true)
        err_T = np.abs(theta - theta_true)
        err_T = np.minimum(err_T, 2.0 * np.pi - err_T)

        all_err = np.concatenate([err_V, err_T])
        mae_sum += np.mean(all_err)
        rmse_sum += np.mean(all_err ** 2)

    return {
        'mae': mae_sum / n_samples,
        'rmse': np.sqrt(rmse_sum / n_samples),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='WLS state estimation baselines')
    parser.add_argument('--system', type=str, default='33',
                        choices=['33', '118', '123bw', '1746'])
    parser.add_argument('--meas_rate', type=float, default=0.6)
    parser.add_argument('--method', type=str, default='both',
                        choices=['distflow_wls', 'gnwls', 'both', 'distflow_only'])
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    np.random.seed(args.seed)

    print(f"{'='*60}")
    print("WLS state estimation baselines (scipy LM solver)")
    print(f"  System: {args.system}, meas_rate: {args.meas_rate}")
    print(f"  Method: {args.method}")
    print(f"{'='*60}\n")

    # Load data
    splits = load_dataset(args.system, 'data')
    if splits is not None and abs(args.meas_rate - 0.6) < 1e-6:
        print(f"Loaded v3 dataset for '{args.system}'")
        _, _, test_d = splits['train'], splits['val'], splits['test']
    else:
        print(f"Generating dataset (meas_rate={args.meas_rate})...")
        dl, _ = generate_paper_dataset(args.system, n_steps=500,
                                        meas_rate=args.meas_rate, seed=args.seed)
        splits = create_splits(dl)
        _, _, test_d = splits['train'], splits['val'], splits['test']

    N = test_d[0].n_buses
    print(f"  Test: {len(test_d)} samples, {N} buses")

    # System params
    params = get_test_system(args.system)

    # Sanity check
    mi0, z0 = extract_measurements(test_d[0])
    print(f"  Sample 0: {len(z0)} measurements "
          f"(P_inj:{len(mi0['P_inj'])} Q_inj:{len(mi0['Q_inj'])} "
          f"P_f:{len(mi0['P_flow'])} Q_f:{len(mi0['Q_flow'])})")

    # Run
    all_results = {}
    rate_str = f"_{int(args.meas_rate*100)}meas" if abs(args.meas_rate - 0.6) > 1e-6 else ""

    if args.method in ('distflow_only', 'both'):
        print(f"\n{'='*40}")
        print("DistFlow-only baseline")
        print(f"{'='*40}")
        df_r = evaluate_distflow_only(test_d, params)
        all_results['distflow_only'] = df_r
        print(f"  MAE  = {df_r['mae']:.6f}")
        print(f"  RMSE = {df_r['rmse']:.6f}")

    if args.method in ('distflow_wls', 'both'):
        print(f"\n{'='*40}")
        print("Tier A: DistFlow-WLS (DistFlow init + scipy LM)")
        print(f"{'='*40}")
        ta = evaluate_wls_on_dataset('distflow_wls', test_d, params, verbose=args.verbose)
        all_results['tier_a_distflow_wls'] = ta
        if 'error' not in ta:
            print(f"  MAE  = {ta['mae']:.6f}")
            print(f"  RMSE = {ta['rmse']:.6f}")
            print(f"  MAE_V = {ta['mae_V']:.6f}, MAE_theta = {ta['mae_theta']:.6f}")
            print(f"  Avg nfev: {ta['avg_nfev']:.1f}, Time: {ta['avg_time_ms']:.1f} ms")
            print(f"  Converged: {ta['convergence_rate']*100:.1f}%")

    if args.method in ('gnwls', 'both'):
        print(f"\n{'='*40}")
        print("Tier B: GN-WLS (flat start + scipy LM)")
        print(f"{'='*40}")
        tb = evaluate_wls_on_dataset('gnwls', test_d, params, verbose=args.verbose)
        all_results['tier_b_gnwls'] = tb
        if 'error' not in tb:
            print(f"  MAE  = {tb['mae']:.6f}")
            print(f"  RMSE = {tb['rmse']:.6f}")
            print(f"  MAE_V = {tb['mae_V']:.6f}, MAE_theta = {tb['mae_theta']:.6f}")
            print(f"  Avg nfev: {tb['avg_nfev']:.1f}, Time: {tb['avg_time_ms']:.1f} ms")
            print(f"  Converged: {tb['convergence_rate']*100:.1f}%")

    # Summary
    print(f"\n{'='*60}")
    print("WLS Results Summary")
    print(f"{'='*60}")
    print(f"\n{'Method':<25} {'MAE':>10}")
    print("-" * 37)
    for name, r in all_results.items():
        label = {
            'distflow_only': 'DistFlow-only',
            'tier_a_distflow_wls': 'Tier A: DistFlow-WLS',
            'tier_b_gnwls': 'Tier B: GN-WLS',
        }.get(name, name)
        if 'error' not in r:
            print(f"{label:<25} {r['mae']:>10.6f}")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = {}
    for k, v in all_results.items():
        if 'error' not in v:
            clean[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                         for kk, vv in v.items()}
        else:
            clean[k] = v

    path = out_dir / f"wls_{args.system}{rate_str}.json"
    with open(path, 'w') as f:
        json.dump(clean, f, indent=2)
    print(f"\nResults saved to {path}")
    print("Done!")


if __name__ == '__main__':
    main()
