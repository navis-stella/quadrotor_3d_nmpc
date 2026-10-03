"""
simulate_offsetfree.py — Stage 3 Closed-Loop Offset-Free NMPC Simulation
========================================================================
Stage 3:  Offset-Free NMPC with EKF Disturbance Estimation

Demonstrates the complete offset-free architecture:
    1. EKF estimates augmented state  z = [x; d]
    2. Steady-state target calculator computes (x_s, u_s) from d̂
    3. d̂ is injected into the MPC prediction model as a parameter
    4. MPC tracks the equilibrium (x_s, u_s) instead of fixed (x_ref, u_hover)
    5. Plant is simulated with the TRUE disturbance applied continuously

Validation scenario:
    Constant disturbance from t=0:
        d_fx = +0.5 N     (wind force in x, world frame)
        d_fy = -0.3 N     (wind force in y, world frame)
        d_fz = -2.943 N   (30% mass error)
    Stage 4 (closed_loop_sim_config) reuses these values for comparability.

    Phase 1 (0 ≤ t < T_ACTIVATE):
        EKF runs and converges d̂, but offset-free is OFF.
        Standard MPC with zero disturbance assumption.
        → Steady-state position offset visible.

    Phase 2 (t ≥ T_ACTIVATE):
        Offset-free activated: d̂ injected into MPC, target calculator
        updates reference to (x_s, u_s).
        → Position converges back to reference.

Imports:
    quadrotor_3d_model.py       → create_disturbance_plant(), quat_normalize(),
                                   f_hover, NX, NU, ND
    ocp_config_offsetfree.py    → create_solver(), set_disturbance_param(), set_reference()
    ekf_aug.py                  → AugEKF, get_default_ekf_tuning()
    ss_target.py                → compute_ss_target(), print_ss_target()
    plot_utils.py               → shared figures (t_switch marks the activation)
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')    # before the solver libraries load

import numpy as np

from quadrotor_3d_model import (
    create_disturbance_plant, quat_normalize,
    f_hover, NX, NU, ND,
)
from ocp_config_offsetfree import create_solver, set_disturbance_param, set_reference
from ekf_aug import AugEKF, get_default_ekf_tuning
from ss_target import compute_ss_target, print_ss_target
from plot_utils import plot_states, plot_inputs, plot_disturbance, plot_3d_trajectory

RESULTS_DIR = os.path.join('results', 'stage3_offsetfree')


# ─────────────────────────────────────────────────────────────────
# Disturbance Scenario
# ─────────────────────────────────────────────────────────────────
def get_disturbance(t: float) -> np.ndarray:
    """
    Return the true disturbance at time t.

    Scenario:
        Disturbance is always on from t=0 — constant wind + mass error.
        This lets the standard MPC stabilize first (with residual offset),
        then the offset-free mechanism activates to eliminate the error.

    Returns:
        d_true: (6,) = [d_fx, d_fy, d_fz, d_tx, d_ty, d_tz]
    """
    d = np.zeros(ND)
    d[0] = 0.5       # d_fx =  0.5 N    wind force in x (world frame)
    d[1] = -0.3      # d_fy = -0.3 N    wind force in y (world frame)
    d[2] = -0.981 * 3   # d_fz = -0.981 N * 3 30% mass error (Δm·g)
    return d

# Offset-free activation time
T_ACTIVATE = 4.0   # [s] offset-free MPC activates at this time


# ─────────────────────────────────────────────────────────────────
# Closed-Loop Simulation
# ─────────────────────────────────────────────────────────────────
def simulate(x0:        np.ndarray,
             x_ref:     np.ndarray,
             N:         int   = 20,
             T_horizon: float = 1.0,
             T_sim:     float = 10.0):
    """
    Closed-loop offset-free NMPC simulation.

    Demonstration scenario:
        Phase 1 (t < T_ACTIVATE):
            Disturbance present from the start, but offset-free is OFF.
            EKF runs (to build up d̂), but d̂ is NOT injected into MPC
            and reference stays at (x_ref, u_hover).
            → System stabilizes with visible steady-state offset.

        Phase 2 (t >= T_ACTIVATE):
            Offset-free activated: d̂ injected into MPC prediction model,
            target calculator updates reference to (x_s, u_s).
            → Steady-state error is eliminated.

    This shows the problem and the solution in one continuous simulation.

    Returns:
        X:      state trajectory           (n_sim+1, 13)
        U:      input trajectory           (n_sim,   4)
        D_hat:  estimated disturbance      (n_sim,   6)
        D_true: true disturbance           (n_sim,   6)
        X_s:    equilibrium targets        (n_sim,  13)
        Ts:     sample time [s]
    """
    Ts    = T_horizon / N
    n_sim = int(T_sim / Ts)

    # ── Build solver and plant ─────────────────────────────────
    print('Compiling acados solver (disturbance model)...')
    solver = create_solver(x_ref, N, T_horizon)
    print('Solver ready.')

    print('Creating disturbance plant simulator...')
    plant = create_disturbance_plant(Ts)
    print('Plant ready.')

    # ── Initialize EKF ─────────────────────────────────────────
    Q_x, Q_d, R_ekf = get_default_ekf_tuning()
    ekf = AugEKF(Ts=Ts, Q_x=Q_x, Q_d=Q_d, R=R_ekf, x0=x0.copy())
    print('EKF initialized.\n')

    # ── Storage ────────────────────────────────────────────────
    X      = np.zeros((n_sim + 1, NX))
    U      = np.zeros((n_sim,     NU))
    D_hat  = np.zeros((n_sim,     ND))
    D_true = np.zeros((n_sim,     ND))
    X_s    = np.zeros((n_sim,     NX))    # equilibrium targets
    X[0]   = x0

    for k in range(n_sim):
        t = k * Ts

        # ── 1. Measurement (full state access) ─────────────────
        y = X[k].copy()

        # ── 2. EKF update (always runs — builds d̂ even when inactive) ─
        ekf.update(y)

        x_hat = ekf.get_state()
        d_hat = ekf.get_disturbance()
        D_hat[k] = d_hat

        # ── 3. Offset-free activation switch ───────────────────
        offset_free_active = (t >= T_ACTIVATE)

        if offset_free_active:
            # OFFSET-FREE ON: inject d̂ into prediction + update reference
            x_s, u_s = compute_ss_target(x_ref, d_hat)
            set_disturbance_param(solver, d_hat, N)
            set_reference(solver, x_s, u_s, N)
        else:
            # OFFSET-FREE OFF: standard MPC (like Stage 2b)
            # d = 0 in prediction model, reference = (x_ref, u_hover)
            x_s = x_ref.copy()
            u_s = np.full(NU, f_hover)
            set_disturbance_param(solver, np.zeros(ND), N)
            set_reference(solver, x_ref, u_s, N)

        X_s[k] = x_s

        # ── 4. Solve MPC ───────────────────────────────────────
        solver.set(0, 'lbx', X[k])
        solver.set(0, 'ubx', X[k])

        status = solver.solve()
        if status not in [0, 2]:
            print(f'  [step {k:3d}] solver status {status} — stopping.')
            X      = X[:k+1]
            U      = U[:k]
            D_hat  = D_hat[:k]
            D_true = D_true[:k]
            X_s    = X_s[:k]
            break

        U[k] = solver.get(0, 'u')

        # ── 5. EKF predict (propagate ẑ forward) ───────────────
        ekf.predict(U[k])

        # ── 6. Plant step (with TRUE disturbance) ──────────────
        d_true = get_disturbance(t)
        D_true[k] = d_true

        plant.set('x', X[k])
        plant.set('u', U[k])
        plant.set('p', d_true)
        plant.solve()
        X[k+1] = quat_normalize(plant.get('x'))

        # ── Progress ───────────────────────────────────────────
        if k % 20 == 0:
            mode = "OFFSET-FREE" if offset_free_active else "STANDARD"
            print(f'  step {k:3d}/{n_sim}  t={t:5.2f}s  [{mode}]  |  '
                  f'px={X[k,0]:+.3f}  pz={X[k,2]:+.3f}  |  '
                  f'd̂_fx={d_hat[0]:+.4f}  d̂_fz={d_hat[2]:+.4f}')

    # Print target calculator result for the final disturbance
    if np.any(D_hat[-1] != 0):
        print_ss_target(x_ref, X_s[-1], u_s, D_hat[-1])

    print('\nSimulation complete.')
    return X, U, D_hat, D_true, X_s, Ts


# ─────────────────────────────────────────────────────────────────
# Summary Statistics
# ─────────────────────────────────────────────────────────────────
def print_summary(X, D_hat, D_true, Ts, x_ref, t_dist=0.0):
    """Print convergence diagnostics."""
    n = X.shape[0] - 1

    # Steady-state error (last 1 second)
    n_last = int(1.0 / Ts)
    pos_err = np.mean(np.abs(X[-n_last:, :3] - x_ref[:3]), axis=0)

    # Disturbance estimation error (last 1 second)
    d_err = np.mean(np.abs(D_hat[-n_last:] - D_true[-n_last:]), axis=0)

    # Time to 95% d̂ convergence (for d_fx, measured from t_dist)
    idx_dist = int(t_dist / Ts)
    d_fx_true = D_true[idx_dist, 0]
    if d_fx_true != 0:
        d_fx_err = np.abs(D_hat[idx_dist:, 0] - d_fx_true)
        converged = np.where(d_fx_err < 0.05 * abs(d_fx_true))[0]
        t_conv = converged[0] * Ts if len(converged) > 0 else float('inf')
    else:
        t_conv = 0.0

    print('\n' + '='*60)
    print('STAGE 3 — OFFSET-FREE NMPC SUMMARY')
    print('='*60)
    print(f'  Simulation time:     {n*Ts:.1f} s')
    print(f'  Disturbance onset:   t = {t_dist} s')
    print(f'  Offset-free active:  t = {T_ACTIVATE} s')
    print(f'\n  Steady-state position error (last 1s):')
    print(f'    Δpx = {pos_err[0]:.4f} m')
    print(f'    Δpy = {pos_err[1]:.4f} m')
    print(f'    Δpz = {pos_err[2]:.4f} m')

    # Phase-1 steady-state offset (standard MPC, before activation)
    i0, i1 = int(3.0 / Ts), int(T_ACTIVATE / Ts)
    pos_off = np.mean(X[i0:i1, :3] - x_ref[:3], axis=0)   # signed, not abs
    print(f'\n  Phase-1 position offset (t = 3–{T_ACTIVATE:.0f} s, standard MPC):')
    print(f'    Δpx = {pos_off[0]:+.4f} m')
    print(f'    Δpy = {pos_off[1]:+.4f} m')
    print(f'    Δpz = {pos_off[2]:+.4f} m')

    print(f'\n  Disturbance estimation error (last 1s):')
    print(f'    Δd_fx = {d_err[0]:.4f} N')
    print(f'    Δd_fy = {d_err[1]:.4f} N')
    print(f'    Δd_fz = {d_err[2]:.4f} N')
    print(f'    Δd_τx = {d_err[3]:.6f} N·m')
    print(f'    Δd_τy = {d_err[4]:.6f} N·m')
    print(f'    Δd_τz = {d_err[5]:.6f} N·m')
    print(f'\n  d̂_fx 95% convergence time: {t_conv:.2f} s after onset')
    print('='*60)


# ─────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # ── Reference: hover at (2, 1, 3) ──────────────────────────
    x_ref = np.array([
        2.0, 1.0, 3.0,            # px, py, pz
        0.0, 0.0, 0.0,            # vx, vy, vz
        1.0, 0.0, 0.0, 0.0,       # qw, qx, qy, qz
        0.0, 0.0, 0.0,            # p, q, r
    ])

    # ── Initial state: origin, level, stationary ────────────────
    x0    = np.zeros(NX)
    x0[6] = 1.0   # qw = 1

    # ── Run ─────────────────────────────────────────────────────
    X, U, D_hat, D_true, X_s, Ts = simulate(
        x0, x_ref, N=20, T_horizon=1.0, T_sim=10.0
    )

    # ── Plots ───────────────────────────────────────────────────
    marker = dict(t_switch=T_ACTIVATE, switch_label=f'offset-free ON (t = {T_ACTIVATE:g} s)')
    plot_states(X, Ts, x_ref, X_s=X_s, **marker,
                title='Stage 3 — Offset-Free NMPC: State Trajectory',
                save_path=os.path.join(RESULTS_DIR, 'states.png'))
    plot_inputs(U, Ts, **marker,
                title='Stage 3 — Offset-Free NMPC: Input Trajectory',
                save_path=os.path.join(RESULTS_DIR, 'inputs.png'))
    plot_disturbance(D_hat, D_true, Ts, **marker,
                     title='Stage 3 — EKF Disturbance Estimation',
                     save_path=os.path.join(RESULTS_DIR, 'disturbance.png'))
    plot_3d_trajectory(X, x_ref,
                       title='Stage 3 Offset-Free NMPC — 3D Flight Path',
                       save_path=os.path.join(RESULTS_DIR, 'trajectory_3d.png'))

    # ── Summary ─────────────────────────────────────────────────
    print_summary(X, D_hat, D_true, Ts, x_ref)
