"""
simulate_trajectory.py — Stage 6 Closed-Loop Trajectory-Tracking NMPC
=====================================================================
The quadrotor flies the helix of reference_generate.py — take-off hop, spiral-
out, constant-rate climb (R = 2 m, 5 m climb over 3 turns, 2 m/s) — with
the Stage 4b measurement and estimator:

    plant        no disturbance  (D_TRUE = 0, see there)
    measurement  y = [p; q; ω] with mocap noise  (sensor_simulator)
    estimator    certified Lyapunov MHE  ẑ = [x̂; d̂]   (state_est_mhe,
                 unchanged from Stage 4b; needs data/mhe_params.npz)
    controller   trajectory-tracking NMPC         (ocp_config_trajectory)

    The MHE certificate holds for any attitude, 0 ≤ T_total ≤ 4·F_MAX and
    |ωᵢ| ≤ OMEGA_MAX; the helix needs |ω_r| ≤ 0.55 rad/s, and the MPC's soft
    |ωᵢ| ≤ OMEGA_MPC = 0.9·OMEGA_MAX keeps the plant inside the envelope.

Loop (Ts = 0.05 s, MPC horizon 1 s, N = 20):

    y_k             = sensor(x_k)
    ẑ_k             = MHE(y_k, u_{k−1})   (prediction form: uses y up to k−1)
    (X_r, U_r)_k    = flatness( helix(t_k + i·Ts), d̂_k ),  i = 0..N
    u_k             = MPC(x̂_k, X_r, U_r, d̂_k)           (shifted warm start)
    x_{k+1}         = plant(x_k, u_k, d_true)

    d̂ (≈ 0 here, but still estimated) enters the flatness map too: the
    reference attitude and thrust are those that fly the helix UNDER the
    estimated disturbance, so the reference stays a trajectory of the
    prediction model f(·, ·, d̂).

--nominal: d_true = 0, x̂ = x (full state feedback), d̂ = 0 — the tracking
MPC alone, as a check of the controller without estimation.

Imports:
    reference_generate.py     → HelixReference (p, v, a, jerk, snap, yaw)
    reference_flatness.py     → flat_to_state_input  (→ x_r, u_r under d̂)
    ocp_config_trajectory.py  → create_solver() + runtime helpers
    state_est_mhe.py          → MovingHorizonEstimator  (Stage 4b)
    closed_loop_sim_config.py → OMEGA_MPC  (Stage 4 value)

Run:  python simulate_trajectory.py [--nominal]
      → results/stage6_trajectory/ (nominal: results/stage6_trajectory/nominal/)
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')    # before the solver libraries load

import argparse
import time
import numpy as np

from quadrotor_3d_model import (create_disturbance_plant, quat_normalize,
                                project_measurement, OMEGA_MAX, F_MAX, NX, NU, ND, NY)
from reference_generate     import HelixReference
from reference_flatness     import flat_to_state_input
import ocp_config_trajectory as traj
from closed_loop_sim_config import OMEGA_MPC
from sensor_simulator       import SensorSimulator, MOCAP_NOISE_STD
from state_est_mhe          import MovingHorizonEstimator
from plot_utils             import (plot_states, plot_inputs, plot_3d_trajectory,
                                    plot_tracking_error, plot_disturbance,
                                    plot_solve_times)

RESULTS_DIR = os.path.join('results', 'stage6_trajectory')

# ─────────────────────────────────────────────────────────────────
# Scenario
# ─────────────────────────────────────────────────────────────────
N         = 20
T_HORIZON = 1.0
TS        = T_HORIZON / N            # 0.05 s — plant, sensor, MHE and MPC
REF       = HelixReference()         # default helix (reference_generate.HelixParams)
T_SIM     = REF.duration             # ≈ 22.9 s: hop + ramp + 3 turns
N_SIM     = int(np.ceil(T_SIM / TS))

# No disturbance. With the certified weights the MHE needs ≈ 2 s to learn a
# constant d from its prior d = 0 (a Stage 4b property: the arrival weight on
# d is about one position sample) — the 30 % mass error of Stages 3–4 made the
# quadrotor sag 0.43 m, the wind alone drift 0.2 m during take-off. That is an
# estimator start-up effect, not part of trajectory tracking. The d̂ path
# (prediction model + flatness map) stays in place for a non-zero D_TRUE.
D_TRUE = np.array([0, 0, 0.0, 0.0, 0.0, 0.0])   # [N, N, N, N·m, N·m, N·m]

P = REF.params
T_CRUISE = P.T_hop + P.T_ramp        # constant-radius helix from here on
PHASES   = [(P.T_hop, 'spiral'), (T_CRUISE, 'cruise')]


# ─────────────────────────────────────────────────────────────────
# Closed Loop
# ─────────────────────────────────────────────────────────────────
def simulate(nominal: bool = False, sensor_seed: int = 42) -> dict:
    """
    Returns dict with X (n+1, 13), X_hat (n, 13), D_hat (n, 6), D_true (n, 6),
    X_r (n, 13), U_r (n, 4), U (n, 4), status (n,), T_est (n,), T_mpc (n,) [s],
    Y (n, 10) noisy measurements y_k = [p; q; ω] (None for --nominal).

    X_r[k], U_r[k] = node 0 of the reference handed to the MPC at t_k (under
    d̂_k). T_mpc covers flatness map + reference setting + solve. A failed
    solve (status ∉ {0, 2}) is counted; its first input is still applied,
    clipped to the motor limits (u_r if not finite).
    """
    x0 = flat_to_state_input(REF.sample(0.0)).x_r    # at rest at the origin
    print('Compiling acados NMPC solver (trajectory tracking)...')
    solver = traj.create_solver(N, T_HORIZON, omega_max=OMEGA_MPC)
    plant  = create_disturbance_plant(TS)

    if not nominal:
        sensor = SensorSimulator(noise_std=MOCAP_NOISE_STD, seed=sensor_seed)
        y      = sensor.measure(x0)
        print('Building MHE (certificate data/mhe_params.npz)...')
        est    = MovingHorizonEstimator(Ts=TS, z0=project_measurement(y))

    ref0 = flat_to_state_input(REF.horizon(0.0, N, TS))
    traj.init_guess(solver, ref0.x_r, ref0.u_r, N)

    X      = np.zeros((N_SIM + 1, NX)); X[0] = x0
    X_hat  = np.zeros((N_SIM, NX))
    D_hat  = np.zeros((N_SIM, ND))
    D_true = np.zeros((N_SIM, ND))
    X_r    = np.zeros((N_SIM, NX))
    U_r    = np.zeros((N_SIM, NU))
    U      = np.zeros((N_SIM, NU))
    status = np.zeros(N_SIM, dtype=int)
    T_est  = np.full(N_SIM, np.nan)
    T_mpc  = np.zeros(N_SIM)
    Y      = None if nominal else np.zeros((N_SIM, NY))

    for k in range(N_SIM):
        t = k * TS

        # 1. State + disturbance estimate
        if nominal:
            x_hat, d_hat = X[k].copy(), np.zeros(ND)
        else:
            if k > 0:
                y = sensor.measure(X[k])
            Y[k] = y                                         # y₀ was taken above (prior)
            t0 = time.perf_counter()
            z_hat = est.step(y, U[k - 1] if k > 0 else None)
            T_est[k] = time.perf_counter() - t0
            x_hat = quat_normalize(z_hat[:NX].copy())       # MPC wants unit q
            d_hat = z_hat[NX:]
        X_hat[k], D_hat[k] = x_hat, d_hat

        # 2. Reference over the horizon (flatness under d̂) + NMPC
        t0 = time.perf_counter()
        ref = flat_to_state_input(REF.horizon(t, N, TS), d_hat)
        if k > 0:
            traj.shift_warm_start(solver, ref.x_r, ref.u_r, N)
        traj.set_trajectory(solver, ref.x_r, ref.u_r, d_hat, N)
        traj.set_initial_state(solver, x_hat)
        status[k] = solver.solve()
        T_mpc[k]  = time.perf_counter() - t0

        u = solver.get(0, 'u')
        if not np.all(np.isfinite(u)):
            u = ref.u_r[0]
        U[k] = np.clip(u, 0.0, F_MAX)
        X_r[k], U_r[k] = ref.x_r[0], ref.u_r[0]

        # 3. Plant with the TRUE disturbance
        d_true    = np.zeros(ND) if nominal else D_TRUE
        D_true[k] = d_true
        plant.set('x', X[k])
        plant.set('u', U[k])
        plant.set('p', d_true)
        plant.solve()
        X[k + 1] = quat_normalize(plant.get('x'))

        if k % int(round(2.0 / TS)) == 0:
            print(f'  t={t:5.2f}s  |  ‖p − p_r‖ = '
                  f'{np.linalg.norm(X[k, :3] - X_r[k, :3]):.3f} m  |  '
                  f'd̂_fx = {d_hat[0]:+.3f} (true {d_true[0]:+.3f})  |  status {status[k]}')

    return dict(X=X, X_hat=X_hat, D_hat=D_hat, D_true=D_true, X_r=X_r,
                U_r=U_r, U=U, status=status, T_est=T_est, T_mpc=T_mpc, Y=Y,
                info='' if nominal else est.info())


# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────
def tracking_errors(res: dict):
    """‖p − p_r‖ [m] and attitude error angle [°] at t_k, k = 0..n−1."""
    X, X_r = res['X'][:-1], res['X_r']
    e_p = np.linalg.norm(X[:, :3] - X_r[:, :3], axis=1)
    dot = np.abs(np.sum(X[:, 6:10] * X_r[:, 6:10], axis=1))
    e_q = np.rad2deg(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))
    return e_p, e_q


def summarize(res: dict, label: str) -> None:
    e_p, e_q = tracking_errors(res)
    t      = np.arange(len(e_p)) * TS
    cruise = t >= T_CRUISE + 2.0          # skip the first 2 s after the ramp
    fail   = ~np.isin(res['status'], [0, 2])
    U, T_mpc = res['U'], res['T_mpc'] * 1e3
    sat    = np.mean(np.any((U <= 1e-6) | (U >= F_MAX - 1e-6), axis=1))
    rms    = lambda e: np.sqrt(np.mean(e ** 2))

    print('\n' + '═' * 66)
    print(f'  Stage 6 — Trajectory-tracking NMPC ({label})')
    print('═' * 66)
    print(f'    failed OCP calls          : {fail.sum()}/{len(fail)}')
    print(f'    ‖p − p_r‖  RMS / max (all): {1e3 * rms(e_p):7.1f} / {1e3 * e_p.max():7.1f} mm')
    print(f'    ‖p − p_r‖  RMS / max (t ≥ {T_CRUISE + 2:.0f} s): '
          f'{1e3 * rms(e_p[cruise]):7.1f} / {1e3 * e_p[cruise].max():7.1f} mm')
    print(f'    attitude err RMS (t ≥ {T_CRUISE + 2:.0f} s): {rms(e_q[cruise]):.2f}°')
    print(f'    max |ω|                   : {np.abs(res["X"][:, 10:]).max():.3f} rad/s'
          f'   (soft bound {OMEGA_MPC:.2f}, envelope {OMEGA_MAX})')
    print(f'    thrust range              : [{U.min():.3f}, {U.max():.3f}] N'
          f'   (limits [0, {F_MAX:.3f}]), saturated {sat:.1%} of samples')
    if not np.all(np.isnan(res['T_est'])):
        d_err = np.abs(res['D_hat'][cruise] - res['D_true'][cruise]).mean(axis=0)
        print(f'    |d̂ − d| mean (t ≥ {T_CRUISE + 2:.0f} s)  : force {np.round(d_err[:3], 4)} N,'
              f' torque {np.round(d_err[3:], 5)} N·m')
        print(f'    MHE time mean / max       : {1e3 * np.nanmean(res["T_est"]):.2f} / '
              f'{1e3 * np.nanmax(res["T_est"]):.2f} ms')
        print(f'    {res["info"]}')
    print(f'    MPC time mean / p99 / max : {T_mpc.mean():.2f} / {np.percentile(T_mpc, 99):.2f}'
          f' / {T_mpc.max():.2f} ms   (flatness + solve, budget {1e3 * TS:.0f} ms)')


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Stage 6 trajectory-tracking NMPC')
    ap.add_argument('--nominal', action='store_true',
                    help='no disturbance, full state feedback (tracking MPC alone)')
    args = ap.parse_args()

    label = 'nominal, full state' if args.nominal else 'MHE, noisy y'
    out   = os.path.join(RESULTS_DIR, 'nominal') if args.nominal else RESULTS_DIR
    res   = simulate(nominal=args.nominal)
    summarize(res, label)

    title = 'Stage 6 — Trajectory tracking' + (' (nominal)' if args.nominal else '')
    plot_tracking_error(res['X'], res['X_r'], TS,
                        f'{title}: tracking error', os.path.join(out, 'tracking_error.png'),
                        t_marks=PHASES)
    plot_states(res['X'], TS, res['X_r'],
                f'{title}: states vs reference x_r', os.path.join(out, 'states.png'),
                Y=res['Y'])
    plot_inputs(res['U'], TS, f'{title}: motor thrusts (dashed: u_r)',
                os.path.join(out, 'inputs.png'), U_ref=res['U_r'])
    plot_3d_trajectory(res['X'], res['X_r'],
                       f'{title}: 3D flight path', os.path.join(out, 'trajectory_3d.png'),
                       P_meas=None if res['Y'] is None else res['Y'][:, :3])
    if not args.nominal:
        plot_disturbance(res['D_hat'], res['D_true'], TS,
                         f'{title}: disturbance estimate', os.path.join(out, 'disturbance.png'))
        plot_solve_times(res['T_est'], res['T_mpc'], TS, TS, 'MHE',
                         f'{title}: computation time per call', os.path.join(out, 'solve_time.png'))
