"""
simulate_offsetfree.py — Closed-Loop Offset-Free NMPC Simulation
==================================================================
Stage 3:  Offset-Free NMPC with EKF Disturbance Estimation

Demonstrates the complete offset-free architecture:
    1. EKF estimates augmented state  z = [x; d]
    2. Steady-state target calculator computes (x_s, u_s) from d̂
    3. d̂ is injected into the MPC prediction model as a parameter
    4. MPC tracks the equilibrium (x_s, u_s) instead of fixed (x_ref, u_hover)
    5. Plant is simulated with the TRUE disturbance applied continuously

Validation scenario:
    Constant disturbance from t=0:
        d_fx = 0.5 N     (wind force in x, world frame)
        d_fz = -0.981 N  (10% mass error)

    Phase 1 (0 ≤ t < T_ACTIVATE):
        EKF runs and converges d̂, but offset-free is OFF.
        Standard MPC with zero disturbance assumption.
        → Steady-state position offset visible.

    Phase 2 (t ≥ T_ACTIVATE):
        Offset-free activated: d̂ injected into MPC, target calculator
        updates reference to (x_s, u_s).
        → Position converges back to reference.

Imports:
    quadrotor_3d_model.py       → create_disturbance_plant(), normalize_quaternion(),
                                   quat_to_euler(), f_hover, NX, NU, ND
    ocp_config_offsetfree.py    → create_solver(), set_disturbance_param(), set_reference()
    ekf_aug.py                  → AugEKF, get_default_ekf_tuning()
    ss_target.py                → compute_ss_target(), print_ss_target()
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from quadrotor_3d_model import (
    create_disturbance_plant, normalize_quaternion, quat_to_euler,
    f_hover, NX, NU, ND,
)
from ocp_config_offsetfree import create_solver, set_disturbance_param, set_reference
from ekf_aug import AugEKF, get_default_ekf_tuning
from ss_target import compute_ss_target, print_ss_target


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
    plant = create_disturbance_plant(T_horizon, N)
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
            # OFFSET-FREE OFF: standard MPC (like Stage 2)
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
        X[k+1] = normalize_quaternion(plant.get('x'))

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
# Plotting — State Trajectories
# ─────────────────────────────────────────────────────────────────
def plot_states(X, Ts, x_ref, X_s=None, t_activate=T_ACTIVATE,
                save_path='results/stage3_offsetfree/states.png'):
    """Plot state trajectories with offset-free activation marker."""
    t_x = np.arange(X.shape[0]) * Ts
    t_s = np.arange(X_s.shape[0]) * Ts if X_s is not None else None
    euler_traj = np.rad2deg(quat_to_euler(X[:, 6:10]))
    euler_ref  = np.rad2deg(quat_to_euler(x_ref[6:10]))
    euler_s    = np.rad2deg(quat_to_euler(X_s[:, 6:10])) if X_s is not None else None

    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    fig.suptitle('Stage 3 — Offset-Free NMPC: State Trajectory', fontsize=14)

    labels = [
        ['px [m]', 'py [m]', 'pz [m]'],
        ['vx [m/s]', 'vy [m/s]', 'vz [m/s]'],
        ['roll φ [°]', 'pitch θ [°]', 'yaw ψ [°]'],
        ['p [rad/s]', 'q [rad/s]', 'r [rad/s]'],
    ]

    for row in range(4):
        for col in range(3):
            ax = axes[row][col]

            if row == 0:
                ax.plot(t_x, X[:, col], 'b', lw=1.2, label='state')
                ax.axhline(x_ref[col], color='r', ls='--', lw=1, label='ref')
            elif row == 1:
                ax.plot(t_x, X[:, 3+col], 'b', lw=1.2, label='state')
                ax.axhline(x_ref[3+col], color='r', ls='--', lw=1, label='ref')
            elif row == 2:
                ax.plot(t_x, euler_traj[:, col], 'b', lw=1.2, label='state')
                ax.axhline(euler_ref[col], color='r', ls='--', lw=1, label='ref')
                if euler_s is not None:
                    ax.plot(t_s, euler_s[:, col], 'g--', lw=1,
                            alpha=0.8, label='eq. target')
            else:
                ax.plot(t_x, X[:, 10+col], 'b', lw=1.2, label='state')
                ax.axhline(x_ref[10+col], color='r', ls='--', lw=1, label='ref')

            # Shaded regions: standard MPC vs offset-free
            ylim = ax.get_ylim()
            ax.axvspan(0, t_activate, alpha=0.05, color='red',
                       label='standard MPC')
            ax.axvspan(t_activate, t_x[-1], alpha=0.05, color='green',
                       label='offset-free ON')
            ax.axvline(t_activate, color='green', ls='-.', lw=1,
                       alpha=0.8, label=f'activate t={t_activate}s')
            ax.set_ylabel(labels[row][col])
            ax.set_xlabel('t [s]')
            ax.legend(fontsize=5, loc='best')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────
# Plotting — Input Trajectories
# ─────────────────────────────────────────────────────────────────
def plot_inputs(U, Ts, t_activate=T_ACTIVATE,
                save_path='results/stage3_offsetfree/inputs.png'):
    """Plot motor thrusts with offset-free activation marker."""
    t_u = np.arange(U.shape[0]) * Ts

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle('Stage 3 — Offset-Free NMPC: Input Trajectory', fontsize=14)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    ax = axes[0]
    for i in range(NU):
        ax.step(t_u, U[:, i], where='post', color=colors[i], lw=1.2,
                label=f'f{i+1}')
    ax.axhline(f_hover, color='k', ls='--', lw=1, alpha=0.5, label='f_hover')
    ax.axvline(t_activate, color='green', ls='-.', lw=1, alpha=0.8)
    ax.set_ylabel('thrust [N]')
    ax.set_xlabel('t [s]')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.step(t_u, U.sum(axis=1), where='post', color='purple', lw=1.2,
            label='T_total')
    ax.axhline(4*f_hover, color='k', ls='--', lw=1, alpha=0.5, label='mg')
    ax.axvline(t_activate, color='green', ls='-.', lw=1, alpha=0.8,
               label=f'offset-free ON')
    ax.set_ylabel('total thrust [N]')
    ax.set_xlabel('t [s]')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────
# Plotting — Disturbance Estimation  (Stage 3 specific)
# ─────────────────────────────────────────────────────────────────
def plot_disturbance(D_hat, D_true, Ts, t_activate=T_ACTIVATE,
                     save_path='results/stage3_offsetfree/disturbance.png'):
    """Plot estimated vs true disturbance with activation marker."""
    n = D_hat.shape[0]
    t = np.arange(n) * Ts

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle('Stage 3 — EKF Disturbance Estimation', fontsize=14)

    force_labels  = ['d_fx [N]', 'd_fy [N]', 'd_fz [N]']
    torque_labels = ['d_τx [N·m]', 'd_τy [N·m]', 'd_τz [N·m]']

    for i in range(3):
        ax = axes[0][i]
        ax.plot(t, D_true[:, i], 'r--', lw=1.5, label='true')
        ax.plot(t, D_hat[:, i],  'b',   lw=1.2, label='estimated')
        ax.axvline(t_activate, color='green', ls='-.', lw=1, alpha=0.8,
                   label='offset-free ON')
        ax.set_ylabel(force_labels[i])
        ax.set_xlabel('t [s]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[1][i]
        ax.plot(t, D_true[:, 3+i], 'r--', lw=1.5, label='true')
        ax.plot(t, D_hat[:, 3+i],  'b',   lw=1.2, label='estimated')
        ax.axvline(t_activate, color='green', ls='-.', lw=1, alpha=0.8,
                   label='offset-free ON')
        ax.set_ylabel(torque_labels[i])
        ax.set_xlabel('t [s]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────
# Plotting — 3D Trajectory
# ─────────────────────────────────────────────────────────────────
def plot_3d_trajectory(X, x_ref,
                       save_path='results/stage3_offsetfree/trajectory_3d.png'):
    """Plot 3D flight path."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(X[:, 0], X[:, 1], X[:, 2], 'b-', lw=1.5, label='trajectory')
    ax.scatter(*X[0, :3],  color='green', s=100, marker='o', label='start')
    ax.scatter(*x_ref[:3], color='red',   s=100, marker='*', label='target')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title('Stage 3 Offset-Free NMPC — 3D Flight Path')
    ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Plot saved: {save_path}')


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
    plot_states(X, Ts, x_ref, X_s=X_s)
    plot_inputs(U, Ts)
    plot_disturbance(D_hat, D_true, Ts)
    plot_3d_trajectory(X, x_ref)

    # ── Summary ─────────────────────────────────────────────────
    print_summary(X, D_hat, D_true, Ts, x_ref)
