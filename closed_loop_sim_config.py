"""
closed_loop_sim_config.py — Stage 4 Shared Closed Loop (EKF / MHE)
==================================================================
Extends Stage 3 (offset-free NMPC under full state feedback) to the
realistic case: only y = [p; q; ω] is measured, with noise, and the
augmented state z = [x; d] must be estimated. This module is the ONE
closed loop both estimators run in; the entry scripts only pick the
estimator:

    simulate_ekf.py      Stage 4a  state_est_ekf.ExtendedKalmanFilter
    simulate_mhe.py      Stage 4b  state_est_mhe.MovingHorizonEstimator
    simulate_compare.py  loads both saved runs → comparison figure + table

Shared by both runs — the estimator is the only difference:
    augmented model     quadrotor_3d_model   (z = [x; d], ḋ = 0)
    measurement         sensor_simulator     (y = [p; q; ω], MOCAP_NOISE_STD, seed)
    controller          ocp_config_offsetfree + ss_target  (d̂ in the prediction,
                                                            target under d̂)
    disturbance         get_disturbance()    (constant from t = 0)
    reference, x0       X_REF, X0            (hover at (2, 1, 3) from rest at 0)
    initial prior       z̄₀ = project_measurement(y₀)   (v = 0, d = 0)
    rates               see below

Scenario (inherited from Stage 3 for direct comparability):
    d_fx = +0.5 N     (wind, world x)
    d_fy = -0.3 N     (wind, world y)
    d_fz = -2.943 N   (30% mass error)

Loop (multi-rate capable; currently single-rate, r = 1):
    Ts     = 0.05 s    plant, sensor, estimator       (base rate, index k)
    Ts_mpc = 0.05 s    NMPC, every r = Ts_mpc / Ts samples; thrust held (ZOH)
    The two horizons are independent: MPC 1 s (N = 20); MHE N = ⌈1.5·T_min/Ts⌉
    from the certificate (state_est_mhe) — it may well exceed the MPC's.

    y_k   = sensor(x_k)                                ← partial measurement
    ẑ_k   = estimator.step(y_k, u_{k-1})               ← EKF or MHE
    if k mod r = 0:
        (x_s, u_s) = ss_target(x_ref, d̂_k)            ← equilibrium under d̂
        u_k = MPC(x̂_k, x_s, u_s, d̂_k)                 ← MPC uses the ESTIMATE
    else u_k = u_{k-1}
    x_{k+1} = plant(x_k, u_k, d_true)                  ← ground truth

    Information timing: the EKF is a filter (ẑ_k uses y_k); the MHE is in
    prediction form as in the paper (ẑ_k uses y up to k-1).

Body-rate limit:
    The MHE certificate holds on |ωᵢ| ≤ OMEGA_MAX. The MPC keeps the plant
    inside it with a soft |ωᵢ| ≤ 0.9·OMEGA_MAX (both runs, same controller);
    the MHE additionally imposes the hard |ω̂ᵢ| ≤ OMEGA_MAX of the paper.

Computation time:
    Wall-clock time of every estimator call and every controller call
    (target calculator + MPC solve), compared with the real-time budgets
    Ts and Ts_mpc. OMP_NUM_THREADS = 1: acados is built with OpenMP, and on
    problems this small thread start-up costs more than it saves.

Results of each run:
    results/stage4_<ekf|mhe>[_nominal]/   figures + sim_data.npz (all arrays,
                                          reloaded by simulate_compare.py)
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')    # before the solver libraries load

import time
import numpy as np

from quadrotor_3d_model import (
    create_disturbance_plant, normalize_quaternion, project_measurement,
    OMEGA_MAX, NX, NU, ND, NZ,
)
from ocp_config_offsetfree import create_solver, set_disturbance_param, set_reference
from ss_target        import compute_ss_target, print_ss_target
from sensor_simulator import SensorSimulator, MOCAP_NOISE_STD
from state_est_ekf    import ExtendedKalmanFilter, default_ekf_R
from state_est_mhe    import MovingHorizonEstimator
from plot_utils       import (
    plot_states, plot_inputs, plot_3d_trajectory, plot_disturbance,
    plot_velocity_estimation, plot_solve_times,
)

OMEGA_MPC   = 0.9 * OMEGA_MAX       # MPC soft rate limit — margin to the certificate
ESTIMATORS  = ('ekf', 'mhe')
STAGE_LABEL = {'ekf': 'Stage 4a — EKF', 'mhe': 'Stage 4b — MHE'}

# Reference: hover at (2, 1, 3); initial state: origin, level, at rest
X_REF = np.array([
    2.0, 1.0, 3.0,           # px, py, pz
    0.0, 0.0, 0.0,           # vx, vy, vz
    1.0, 0.0, 0.0, 0.0,      # qw, qx, qy, qz
    0.0, 0.0, 0.0,           # p, q, r
])
X0 = np.zeros(NX)
X0[6] = 1.0


# ─────────────────────────────────────────────────────────────────
# Disturbance scenario  (same values as Stage 3)
# ─────────────────────────────────────────────────────────────────
def get_disturbance(t: float) -> np.ndarray:
    """Constant disturbance active from t = 0."""
    d = np.zeros(ND)
    d[0] =  0.5             # d_fx =  0.5 N   (wind in world x)
    d[1] = -0.3             # d_fy = -0.3 N   (wind in world y)
    d[2] = -0.981 * 3       # d_fz = -2.943 N (30% mass error, Δm·g)
    return d


# ─────────────────────────────────────────────────────────────────
# Estimator factory
# ─────────────────────────────────────────────────────────────────
def make_estimator(kind: str, Ts: float, z0: np.ndarray, sensor_noise: dict):
    """Both estimators start from the same prior z̄₀; the EKF's R is matched to the sensor."""
    if kind == 'ekf':
        return ExtendedKalmanFilter(Ts=Ts, z0=z0, R=default_ekf_R(sensor_noise))
    if kind == 'mhe':
        return MovingHorizonEstimator(Ts=Ts, z0=z0)
    raise ValueError(f'unknown estimator {kind!r}, choose from {ESTIMATORS}')


# ─────────────────────────────────────────────────────────────────
# Closed-loop simulation
# ─────────────────────────────────────────────────────────────────
def simulate(estimator:    str,
             x0:           np.ndarray = X0,
             x_ref:        np.ndarray = X_REF,
             N_mpc:        int   = 20,
             T_horizon:    float = 1.0,
             Ts:           float = 0.05,
             T_sim:        float = 10.0,
             sensor_noise: dict  = MOCAP_NOISE_STD,
             sensor_seed:  int   = 42,
             disturbance:  bool  = True) -> dict:
    """
    Closed-loop offset-free NMPC with the chosen estimator.

    Args:
        estimator:        'ekf' or 'mhe'
        N_mpc, T_horizon: MPC horizon; Ts_mpc = T_horizon / N_mpc
        Ts:               base rate of plant, sensor and estimator;
                          Ts_mpc must be an integer multiple of it
        sensor_noise:     σ dict for SensorSimulator (None = noise-free)
        disturbance:      apply get_disturbance() to the plant (else d = 0)

    Returns:
        result dict with X, X_hat, D_hat, U, D_true, X_s, T_est, T_mpc,
        x_ref, Ts, Ts_mpc, name, info
    """
    Ts_mpc = T_horizon / N_mpc
    r_mpc  = int(round(Ts_mpc / Ts))
    assert r_mpc >= 1 and np.isclose(r_mpc * Ts, Ts_mpc), \
        f'Ts_mpc = {Ts_mpc} s must be an integer multiple of Ts = {Ts} s'
    n_sim = int(round(T_sim / Ts))

    # ── Controller, plant, sensor ──────────────────────────────
    print('Compiling acados NMPC solver (offset-free)...')
    mpc = create_solver(x_ref, N=N_mpc, T_horizon=T_horizon, omega_max=OMEGA_MPC)
    print('Creating disturbance plant simulator...')
    plant  = create_disturbance_plant(Ts)
    sensor = SensorSimulator(noise_std=sensor_noise, seed=sensor_seed)

    # ── Storage (all on the base grid Ts) ──────────────────────
    X      = np.zeros((n_sim + 1, NX))       # plant state (ground truth)
    Z_hat  = np.zeros((n_sim + 1, NZ))       # estimate ẑ = [x̂; d̂]
    U      = np.zeros((n_sim,     NU))       # applied (held) thrusts
    D_true = np.zeros((n_sim,     ND))
    X_s    = np.zeros((n_sim,     NX))       # steady-state target
    T_est  = np.full (n_sim, np.nan)         # wall-clock [s]
    T_mpc  = np.full (n_sim, np.nan)         # wall-clock [s], NaN = held input
    X[0]   = x0

    # ── Estimator: same prior z̄₀ from the first measurement ────
    y = sensor.measure(X[0])
    print(f'Building {estimator.upper()}...')
    est = make_estimator(estimator, Ts, project_measurement(y), sensor_noise)
    print(f'Ready — {est.name} at Ts = {Ts} s, MPC at Ts_mpc = {Ts_mpc} s '
          f'(every {r_mpc} samples), |ω| ≤ {OMEGA_MPC:.2f} rad/s in the MPC.\n')

    # ── Main loop ──────────────────────────────────────────────
    for k in range(n_sim):
        t = k * Ts

        # 1. Measure (y₀ was taken above to build the prior)
        if k > 0:
            y = sensor.measure(X[k])

        # 2. Estimate ẑ_k = [x̂_k; d̂_k]
        t0 = time.perf_counter()
        z_hat = est.step(y, U[k-1] if k > 0 else None)
        T_est[k] = time.perf_counter() - t0
        Z_hat[k] = z_hat
        d_hat    = z_hat[NX:]

        # 3. Offset-free NMPC every r_mpc base steps
        if k % r_mpc == 0:
            t0 = time.perf_counter()
            x_hat    = normalize_quaternion(z_hat[:NX].copy())   # MPC wants unit q
            x_s, u_s = compute_ss_target(x_ref, d_hat)
            set_disturbance_param(mpc, d_hat, N_mpc)
            set_reference        (mpc, x_s,   u_s, N_mpc)
            mpc.set(0, 'lbx', x_hat)
            mpc.set(0, 'ubx', x_hat)
            status   = mpc.solve()
            T_mpc[k] = time.perf_counter() - t0
            if status not in [0, 2]:
                print(f'  [step {k:3d}] MPC solver status {status} — stopping.')
                X, Z_hat, U, D_true, X_s, T_est, T_mpc = (
                    X[:k+1], Z_hat[:k+1], U[:k], D_true[:k], X_s[:k],
                    T_est[:k], T_mpc[:k])
                break
            u_k = mpc.get(0, 'u')
        U[k]   = u_k                             # ZOH between MPC updates
        X_s[k] = x_s

        # 4. Plant step with TRUE disturbance
        d_true    = get_disturbance(t) if disturbance else np.zeros(ND)
        D_true[k] = d_true
        plant.set('x', X[k])
        plant.set('u', U[k])
        plant.set('p', d_true)
        plant.solve()
        X[k+1] = normalize_quaternion(plant.get('x'))

        # Progress log (once per second)
        if k % int(round(1.0 / Ts)) == 0:
            print(f'  step {k:3d}/{n_sim}  t={t:5.2f}s  |  '
                  f'px={X[k,0]:+.3f}(x_s={x_s[0]:+.3f})  |  '
                  f'v̂x={z_hat[3]:+.3f}(true:{X[k,3]:+.3f})  |  '
                  f'd̂_fz={d_hat[2]:+.3f}(true:{d_true[2]:+.3f})')

    # Pad the last estimate slot for shape alignment with X in plots
    Z_hat[-1] = Z_hat[-2]

    if np.any(Z_hat[-1, NX:] != 0):
        print_ss_target(x_ref, X_s[-1], u_s, Z_hat[-1, NX:])

    print('\nSimulation complete.')
    return {
        'X': X, 'X_hat': Z_hat[:, :NX], 'D_hat': Z_hat[:, NX:],
        'U': U, 'D_true': D_true, 'X_s': X_s,
        'T_est': T_est, 'T_mpc': T_mpc, 'x_ref': x_ref,
        'Ts': Ts, 'Ts_mpc': Ts_mpc,
        'name': est.name, 'info': est.info(),
    }


# ─────────────────────────────────────────────────────────────────
# Result I/O  (sim_data.npz next to the figures of each run)
# ─────────────────────────────────────────────────────────────────
def results_dir(kind: str, disturbance: bool = True) -> str:
    return f'results/stage4_{kind}' + ('' if disturbance else '_nominal')


def save_result(result: dict, out_dir: str) -> str:
    path = os.path.join(out_dir, 'sim_data.npz')
    os.makedirs(out_dir, exist_ok=True)
    np.savez(path, **result)
    print(f'Data saved: {path}')
    return path


def load_result(out_dir: str) -> dict:
    path = os.path.join(out_dir, 'sim_data.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found — run the corresponding '
                                f'simulate_<ekf|mhe>.py first.')
    with np.load(path) as f:
        r = {k: f[k] for k in f.files}
    for k in ('Ts', 'Ts_mpc'):
        r[k] = float(r[k])
    for k in ('name', 'info'):
        r[k] = str(r[k])
    return r


# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────
def metrics(result: dict) -> dict:
    """Steady-state errors over the last 1 s, envelope, computation time [ms]."""
    X, X_hat, D_hat, D_true, Ts, x_ref = (result['X'], result['X_hat'], result['D_hat'],
                                          result['D_true'], result['Ts'], result['x_ref'])
    n = int(1.0 / Ts)
    ms = lambda T: T[~np.isnan(T)] * 1e3
    return {
        'pos_err':   np.mean(np.abs(X[-n:, :3] - x_ref[:3]), axis=0),
        'v_err':     np.mean(np.abs(X_hat[-n:, 3:6] - X[-n:, 3:6]), axis=0),
        'd_err':     np.mean(np.abs(D_hat[-n-1:-1] - D_true[-n:]), axis=0),
        'omega_max': np.abs(X[:, 10:13]).max(),
        'T_est':     ms(result['T_est']),
        'T_mpc':     ms(result['T_mpc']),
    }


def _time_line(name: str, T_ms: np.ndarray, budget_s: float) -> str:
    return (f'    {name}:  mean {T_ms.mean():6.2f}  median {np.median(T_ms):6.2f}  '
            f'p99 {np.percentile(T_ms, 99):6.2f}  max {T_ms.max():6.2f} ms   '
            f'budget {budget_s * 1e3:.0f} ms, over {np.mean(T_ms > budget_s * 1e3):5.1%}')


def print_summary(result: dict):
    m = metrics(result)
    name = result['name']

    print('\n' + '=' * 72)
    print(f'STAGE 4 — {name} + OFFSET-FREE NMPC — CLOSED-LOOP SUMMARY')
    print('=' * 72)
    print( '  Steady-state position error (last 1 s):')
    for i, ax in enumerate('xyz'):
        print(f'    Δp{ax} = {m["pos_err"][i]:.4f} m')
    print( '\n  Velocity estimation error (last 1 s):')
    for i, ax in enumerate('xyz'):
        print(f'    Δv{ax} = {m["v_err"][i]:.4f} m/s')
    print( '\n  Disturbance estimation error (last 1 s):')
    for i, nm in enumerate(['d_fx', 'd_fy', 'd_fz']):
        print(f'    Δ{nm} = {m["d_err"][i]:.4f} N')
    for i, nm in enumerate(['d_τx', 'd_τy', 'd_τz']):
        print(f'    Δ{nm} = {m["d_err"][3+i]:.6f} N·m')
    ok = '✓' if m['omega_max'] <= OMEGA_MAX else '✗'
    print(f'\n  max |ω| (true) = {m["omega_max"]:.3f} rad/s   '
          f'≤ {OMEGA_MAX} (certified envelope)  {ok}')
    print( '\n  Computation time (wall clock):')
    print(_time_line(f'{name:3s}', m['T_est'], result['Ts']))
    print(_time_line('MPC', m['T_mpc'], result['Ts_mpc']))
    print(f'    {result["info"]}')
    print('=' * 72)


# ─────────────────────────────────────────────────────────────────
# Entry point shared by simulate_ekf.py / simulate_mhe.py
# ─────────────────────────────────────────────────────────────────
def run(kind: str):
    """Parse CLI, simulate one estimator, save data + figures, print summary."""
    import argparse
    parser = argparse.ArgumentParser(description=f'{STAGE_LABEL[kind]} + offset-free NMPC')
    parser.add_argument('--no-disturbance', action='store_true',
                        help='run the plant without the Stage 3 disturbance')
    args = parser.parse_args()

    disturbance = not args.no_disturbance
    r     = simulate(kind, disturbance=disturbance)
    out   = results_dir(kind, disturbance)
    label = STAGE_LABEL[kind]
    x_ref = r['x_ref']

    save_result(r, out)
    plot_states             (r['X'], r['Ts'], x_ref,
                             title=f'{label} — Plant State',
                             save_path=f'{out}/states.png')
    plot_velocity_estimation(r['X'], r['X_hat'], r['Ts'],
                             title=f'{label} — Velocity Estimate vs Truth',
                             save_path=f'{out}/velocity_estimation.png')
    plot_disturbance        (r['D_hat'][:-1], r['D_true'], r['Ts'],
                             title=f'{label} — Disturbance Estimate',
                             save_path=f'{out}/disturbance.png')
    plot_inputs             (r['U'], r['Ts'],
                             title=f'{label} — Motor Thrusts',
                             save_path=f'{out}/inputs.png')
    plot_3d_trajectory      (r['X'], x_ref,
                             title=f'{label} — 3D Flight Path',
                             save_path=f'{out}/trajectory_3d.png')
    plot_solve_times        (r['T_est'], r['T_mpc'], r['Ts'], r['Ts_mpc'], r['name'],
                             title=f'{label} — Computation Time per Call',
                             save_path=f'{out}/solve_time.png')
    print_summary(r)
