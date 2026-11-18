"""
WLAV (weighted least absolute value) state estimation baselines.

Implements two approaches for L1-norm state estimation:
  1. IRLS-WLAV: Iteratively Reweighted Least Squares
     - Each iteration solves a WLS with weights w_i = 1/max(|r_i|, epsilon)
     - Approximates L1 minimization via successive WLS solves
  2. Huber M-estimator: Robust regression with Huber loss
     - Uses scipy.optimize.minimize with Huber rho function
     - Delta threshold controls L2/L1 transition
     - More stable than pure L1, similarly robust to outliers

Usage:
    python -m pifm.baselines.wlav --system 33
"""

import numpy as np
from scipy.optimize import least_squares, minimize
from pathlib import Path
import json
import time
from tqdm import tqdm

from ..test_systems import get_test_system
from ..data_generator import load_dataset, create_splits
from ..dataset_builder import generate_paper_dataset
from .wls import (
    build_ybus, compute_measurements, extract_measurements,
    state_to_Vtheta, Vtheta_to_state, residual_fun
)


# ============================================================
# Approach 1: IRLS-WLAV
# ============================================================

def irls_wlav_estimate(params, meas_indices, z, max_iter=20, epsilon=1e-6,
                        verbose=False):
    """
    WLAV via Iteratively Reweighted Least Squares (IRLS).

    Minimizes: Σ |z_i - h_i(x)|

    Each IRLS iteration solves:
        x_{k+1} = argmin Σ w_i (z_i - h_i(x))^2
        w_i = 1 / max(|r_i|, ε)

    Args:
        params: system parameters dict
        meas_indices: measurement indices dict
        z: measurement vector
        max_iter: max IRLS iterations (default 20)
        epsilon: small constant for numerical stability
        verbose: print convergence info

    Returns:
        V_est, theta_est: estimated state
        info: dict with n_iter, converged, cost, etc.
    """
    N = len(params['parent'])
    G, B = build_ybus(params['parent'], params['r_pu'], params['x_pu'])
    parent = params['parent']

    # Initialize from flat start
    V = np.ones(N, dtype=np.float64)
    theta = np.zeros(N, dtype=np.float64)

    n_meas = len(z)
    weights = np.ones(n_meas, dtype=np.float64)

    t0 = time.time()
    total_inner_iters = 0
    converged = False

    for irls_iter in range(max_iter):
        # Solve WLS with current weights
        def weighted_residual(x):
            r = residual_fun(x, meas_indices, z, G, B, parent)
            return np.sqrt(weights) * r

        x0 = Vtheta_to_state(V, theta)
        result = least_squares(
            weighted_residual,
            x0,
            method='lm',
            max_nfev=30,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            verbose=0,
        )
        total_inner_iters += result.nfev

        V_new, theta_new = state_to_Vtheta(result.x, N)
        V_new = np.clip(V_new, 0.85, 1.15)
        theta_new = np.clip(theta_new, -0.5, 0.5)

        # Compute residuals
        r = residual_fun(Vtheta_to_state(V_new, theta_new),
                        meas_indices, z, G, B, parent)

        # Update weights for next IRLS iteration
        # w_i = 1/max(|r_i|, epsilon)  -- standard IRLS for L1
        abs_r = np.abs(r)
        weights_new = 1.0 / np.maximum(abs_r, epsilon)

        # Check convergence: relative change in weights
        weight_change = np.max(np.abs(weights_new - weights)) / (np.max(weights) + 1e-10)
        weights = weights_new
        V, theta = V_new, theta_new

        if verbose and irls_iter % 5 == 0:
            cost = np.sum(abs_r)
            print(f"  IRLS iter {irls_iter}: cost={cost:.6e}, "
                  f"weight_change={weight_change:.6e}")

        if weight_change < 1e-6 and irls_iter >= 3:
            converged = True
            if verbose:
                print(f"  IRLS converged at iter {irls_iter}")
            break

    elapsed = time.time() - t0

    # Final cost
    r_final = residual_fun(Vtheta_to_state(V, theta),
                           meas_indices, z, G, B, parent)
    cost = float(np.sum(np.abs(r_final)))

    return V, theta, {
        'n_irls_iter': irls_iter + 1,
        'total_inner_nfev': total_inner_iters,
        'converged': converged,
        'cost': cost,
        'time_ms': elapsed * 1000,
    }


# ============================================================
# Approach 2: Huber M-estimator
# ============================================================

def huber_rho(r, delta):
    """Huber loss function rho(r)."""
    abs_r = np.abs(r)
    # rho(r) = r^2/2 for |r| <= delta
    # rho(r) = delta * (|r| - delta/2) for |r| > delta
    quad_mask = abs_r <= delta
    result = np.zeros_like(r)
    result[quad_mask] = 0.5 * r[quad_mask] ** 2
    result[~quad_mask] = delta * (abs_r[~quad_mask] - 0.5 * delta)
    return result


def huber_psi(r, delta):
    """Huber psi function (derivative of rho)."""
    return np.clip(r, -delta, delta)


def huber_weights(r, delta):
    """Huber weights: psi(r)/r, with limit at r=0."""
    abs_r = np.abs(r)
    w = np.ones_like(r)
    large = abs_r > delta
    w[large] = delta / abs_r[large]
    return w


def huber_cost(x, meas_indices, z, G, B, parent, delta):
    """Total Huber cost for scipy.optimize.minimize."""
    r = residual_fun(x, meas_indices, z, G, B, parent)
    return np.sum(huber_rho(r, delta))


def huber_gradient(x, meas_indices, z, G, B, parent, delta):
    """
    Gradient of Huber cost via numerical approximation.
    Works well enough for scipy BFGS/L-BFGS-B.
    """
    eps = 1e-8
    grad = np.zeros_like(x)
    f0 = huber_cost(x, meas_indices, z, G, B, parent, delta)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        grad[i] = (huber_cost(x_plus, meas_indices, z, G, B, parent, delta) - f0) / eps
    return grad


def huber_m_estimate(params, meas_indices, z, delta=0.01, method='L-BFGS-B',
                      max_iter=100, verbose=False):
    """
    Huber M-estimator for robust state estimation.

    Minimizes: Σ ρ(z_i - h_i(x))
    where ρ is the Huber loss function.

    Args:
        params: system parameters dict
        meas_indices: measurement indices dict
        z: measurement vector
        delta: Huber threshold (default 0.01 p.u., ~1% of rated values)
        method: optimization method ('L-BFGS-B' or 'BFGS')
        max_iter: max optimization iterations
        verbose: print convergence info

    Returns:
        V_est, theta_est: estimated state
        info: dict with n_iter, converged, cost, etc.
    """
    N = len(params['parent'])
    G, B = build_ybus(params['parent'], params['r_pu'], params['x_pu'])
    parent = params['parent']

    # Flat start
    V_init = np.ones(N, dtype=np.float64)
    theta_init = np.zeros(N, dtype=np.float64)
    x0 = Vtheta_to_state(V_init, theta_init)

    t0 = time.time()

    # Bounds: V in [0.85, 1.15], theta in [-0.5, 0.5]
    n_state = 2 * (N - 1)
    bounds = []
    for i in range(N - 1):
        bounds.append((-0.5, 0.5))  # theta
    for i in range(N - 1):
        bounds.append((0.85, 1.15))  # V

    result = minimize(
        huber_cost,
        x0,
        args=(meas_indices, z, G, B, parent, delta),
        method=method,
        bounds=bounds,
        jac=False,  # Let the optimizer use finite differences
        options={'maxiter': max_iter, 'disp': False},
    )

    elapsed = time.time() - t0

    V_est, theta_est = state_to_Vtheta(result.x, N)
    V_est = np.clip(V_est, 0.85, 1.15)
    theta_est = np.clip(theta_est, -0.5, 0.5)

    r_final = residual_fun(Vtheta_to_state(V_est, theta_est),
                           meas_indices, z, G, B, parent)

    return V_est, theta_est, {
        'n_iter': result.nit,
        'n_fev': result.nfev,
        'converged': result.success,
        'cost_L1': float(np.sum(np.abs(r_final))),
        'cost_Huber': float(result.fun),
        'time_ms': elapsed * 1000,
        'message': str(result.message),
    }


# ============================================================
# Approach 3: IRLS-Huber (combined)
# ============================================================

def irls_huber_estimate(params, meas_indices, z, delta=0.01, max_iter=20,
                         verbose=False):
    """
    WLAV via IRLS using Huber weights (more stable than pure L1 IRLS).

    Uses Huber weight function instead of 1/|r|:
        w_i = min(1, delta/|r_i|)

    This gives L2 behavior for small residuals and L1 for large residuals.

    Args:
        params: system parameters dict
        meas_indices: measurement indices dict
        z: measurement vector
        delta: Huber threshold
        max_iter: max IRLS iterations
        verbose: print convergence info

    Returns:
        V_est, theta_est: estimated state
        info: dict with n_iter, converged, cost, etc.
    """
    N = len(params['parent'])
    G, B = build_ybus(params['parent'], params['r_pu'], params['x_pu'])
    parent = params['parent']

    V = np.ones(N, dtype=np.float64)
    theta = np.zeros(N, dtype=np.float64)

    n_meas = len(z)
    weights = np.ones(n_meas, dtype=np.float64)

    t0 = time.time()
    total_inner_iters = 0
    converged = False

    for irls_iter in range(max_iter):
        def weighted_residual(x):
            r = residual_fun(x, meas_indices, z, G, B, parent)
            return np.sqrt(weights) * r

        x0 = Vtheta_to_state(V, theta)
        result = least_squares(
            weighted_residual,
            x0,
            method='lm',
            max_nfev=30,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            verbose=0,
        )
        total_inner_iters += result.nfev

        V_new, theta_new = state_to_Vtheta(result.x, N)
        V_new = np.clip(V_new, 0.85, 1.15)
        theta_new = np.clip(theta_new, -0.5, 0.5)

        r = residual_fun(Vtheta_to_state(V_new, theta_new),
                        meas_indices, z, G, B, parent)
        weights_new = huber_weights(r, delta)

        weight_change = np.max(np.abs(weights_new - weights)) / (np.max(weights) + 1e-10)
        weights = weights_new
        V, theta = V_new, theta_new

        if verbose and irls_iter % 5 == 0:
            cost = np.sum(np.abs(r))
            print(f"  IRLS-Huber iter {irls_iter}: cost={cost:.6e}, "
                  f"weight_change={weight_change:.6e}")

        if weight_change < 1e-6 and irls_iter >= 3:
            converged = True
            if verbose:
                print(f"  IRLS-Huber converged at iter {irls_iter}")
            break

    elapsed = time.time() - t0

    r_final = residual_fun(Vtheta_to_state(V, theta),
                           meas_indices, z, G, B, parent)
    cost = float(np.sum(np.abs(r_final)))

    return V, theta, {
        'n_irls_iter': irls_iter + 1,
        'total_inner_nfev': total_inner_iters,
        'converged': converged,
        'cost_L1': cost,
        'time_ms': elapsed * 1000,
    }


# ============================================================
# Batch Evaluation
# ============================================================

def evaluate_wlav_on_dataset(estimate_fn, data_list, params, method_name,
                              verbose=False):
    """Run WLAV on all samples and aggregate metrics."""
    mae_sum, rmse_sum = 0.0, 0.0
    maeV_sum, maeT_sum = 0.0, 0.0
    n_iter_sum, time_sum = 0.0, 0.0
    n_conv, n_fail = 0, 0
    n_samples = len(data_list)

    for data in tqdm(data_list, desc=f'Eval {method_name}'):
        try:
            meas_indices, z = extract_measurements(data)
            if len(z) < 2:
                n_fail += 1
                continue

            V_est, theta_est, info = estimate_fn(params, meas_indices, z)

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

            # Use appropriate iteration count
            n_iter = info.get('n_irls_iter', info.get('n_iter', 0))
            n_iter_sum += n_iter
            time_sum += info.get('time_ms', 0)
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
        'avg_iter': n_iter_sum / n_valid,
        'avg_time_ms': time_sum / n_valid,
        'convergence_rate': n_conv / n_valid,
        'n_samples': n_valid,
        'n_failed': n_fail,
    }


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='WLAV state estimation baselines')
    parser.add_argument('--system', type=str, default='33',
                        choices=['33', '118', '123bw'])
    parser.add_argument('--meas_rate', type=float, default=0.6)
    parser.add_argument('--method', type=str, default='all',
                        choices=['irls_l1', 'huber', 'irls_huber', 'all'])
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    np.random.seed(args.seed)

    print(f"{'='*60}")
    print("WLAV (L1-norm) state estimation baselines")
    print(f"  System: {args.system}-bus, meas_rate: {args.meas_rate}")
    print(f"  Method: {args.method}")
    print(f"{'='*60}\n")

    # Load data (same as WLS baseline for fair comparison)
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

    # Run all methods
    all_results = {}

    if args.method in ('irls_l1', 'all'):
        print(f"\n{'='*40}")
        print("Approach 1: IRLS-WLAV (pure L1 via 1/|r| weights)")
        print(f"{'='*40}")

        # Wrapper with verbose support
        def irls_fn(p, mi, z):
            return irls_wlav_estimate(p, mi, z, max_iter=20, verbose=args.verbose)

        r1 = evaluate_wlav_on_dataset(irls_fn, test_d, params, 'IRLS-WLAV',
                                       verbose=args.verbose)
        all_results['IRLS_WLAV'] = r1
        if 'error' not in r1:
            print(f"  MAE  = {r1['mae']:.6f}")
            print(f"  RMSE = {r1['rmse']:.6f}")
            print(f"  MAE_V = {r1['mae_V']:.6f}, MAE_theta = {r1['mae_theta']:.6f}")
            print(f"  Avg IRLS iter: {r1['avg_iter']:.1f}, Time: {r1['avg_time_ms']:.1f} ms")
            print(f"  Converged: {r1['convergence_rate']*100:.1f}%")

    if args.method in ('huber', 'all'):
        print(f"\n{'='*40}")
        print("Approach 2: Huber M-estimator (L-BFGS-B optimizer)")
        print(f"{'='*40}")

        def huber_fn(p, mi, z):
            return huber_m_estimate(p, mi, z, delta=0.01, verbose=args.verbose)

        r2 = evaluate_wlav_on_dataset(huber_fn, test_d, params, 'Huber-M',
                                       verbose=args.verbose)
        all_results['Huber_M'] = r2
        if 'error' not in r2:
            print(f"  MAE  = {r2['mae']:.6f}")
            print(f"  RMSE = {r2['rmse']:.6f}")
            print(f"  MAE_V = {r2['mae_V']:.6f}, MAE_theta = {r2['mae_theta']:.6f}")
            print(f"  Avg iter: {r2['avg_iter']:.1f}, Time: {r2['avg_time_ms']:.1f} ms")
            print(f"  Converged: {r2['convergence_rate']*100:.1f}%")

    if args.method in ('irls_huber', 'all'):
        print(f"\n{'='*40}")
        print("Approach 3: IRLS-Huber (Huber weights via IRLS)")
        print(f"{'='*40}")

        def irlsh_fn(p, mi, z):
            return irls_huber_estimate(p, mi, z, delta=0.01, max_iter=20,
                                        verbose=args.verbose)

        r3 = evaluate_wlav_on_dataset(irlsh_fn, test_d, params, 'IRLS-Huber',
                                       verbose=args.verbose)
        all_results['IRLS_Huber'] = r3
        if 'error' not in r3:
            print(f"  MAE  = {r3['mae']:.6f}")
            print(f"  RMSE = {r3['rmse']:.6f}")
            print(f"  MAE_V = {r3['mae_V']:.6f}, MAE_theta = {r3['mae_theta']:.6f}")
            print(f"  Avg IRLS iter: {r3['avg_iter']:.1f}, Time: {r3['avg_time_ms']:.1f} ms")
            print(f"  Converged: {r3['convergence_rate']*100:.1f}%")

    # Summary and comparison
    print(f"\n{'='*60}")
    print("WLAV Results Summary")
    print(f"{'='*60}")
    print(f"\n{'Method':<30} {'MAE':>10} {'RMSE':>10}")
    print("-" * 52)
    for name, r in all_results.items():
        label = {
            'IRLS_WLAV': 'IRLS-WLAV (pure L1)',
            'Huber_M': 'Huber M-estimator',
            'IRLS_Huber': 'IRLS-Huber',
        }.get(name, name)
        if 'error' not in r:
            print(f"{label:<30} {r['mae']:>10.6f} {r['rmse']:>10.6f}")

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean for JSON
    clean = {}
    for k, v in all_results.items():
        if 'error' not in v:
            clean[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                         for kk, vv in v.items()}
        else:
            clean[k] = v

    # Also save the primary result in the requested format
    # Pick the best method (lowest MAE) for the main output
    best_method = None
    best_mae = float('inf')
    for name, r in all_results.items():
        if 'error' not in r and r['mae'] < best_mae:
            best_mae = r['mae']
            best_method = name

    primary_output = {
        "method": "WLAV (Huber M-estimator, IRLS-Huber, IRLS-L1)",
        "system": f"{args.system}-bus",
        "meas_rate": args.meas_rate,
        "best_method": best_method,
        "mae": round(best_mae, 6) if best_method else None,
        "rmse": round(all_results[best_method]['rmse'], 6) if best_method else None,
        "all_results": clean,
        "references": refs,
    }

    path = out_dir / "e2_wlav_metrics.json"
    with open(path, 'w') as f:
        json.dump(primary_output, f, indent=2)
    print(f"\nResults saved to {path}")
    print("Done!")


if __name__ == '__main__':
    main()
