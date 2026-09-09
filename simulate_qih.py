"""
simulate_qih.py — Closed-Loop QIH-NMPC Simulation + Plotting
==============================================================
Stage 2.1:  Complete Quasi-Infinite Horizon NMPC

3D Quadrotor QIH-NMPC simulation (quaternion model, 13 states).

Key addition over Stage 1 simulation:
    - Extracts the predicted terminal state x_N at each step
    - Computes V_N = (x_N - x_ref)^T P (x_N - x_ref) and plots it
      against alpha to show when the terminal set constraint is
      active, satisfied, or violated (slack > 0).

This visualization demonstrates the QIH trade-off:
    The terminal constraint is violated (slack active) during most
    of the transient — the only phase where stability matters.
    By the time V_N < alpha, the drone is already near hover.
    → Stage 2.2 drops the constraint, keeping only P_DARE.

Imports:
    quadrotor_3d_model.py       → plant, utilities
    ocp_config_qih.py            → QIH-NMPC solver  (Stage 2.1)
    compute_qih_params.py        → offline P, alpha for plotting
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from quadrotor_3d_model         import (create_plant_simulator,
                                        normalize_quaternion,
                                        quat_to_euler, f_hover, NX, NU)
from ocp_config_qih             import create_solver
from compute_qih_params         import compute_qih_offline_parameters


# ─────────────────────────────────────────────────────────────────
# Closed-Loop Simulation
# ─────────────────────────────────────────────────────────────────
def simulate(x0:        np.ndarray,
             x_ref:     np.ndarray,
             N:         int   = 20,
             T_horizon: float = 1.0,
             T_sim:     float = 5.0):
    """
    Closed-loop QIH-NMPC simulation.

    Same structure as Stage 2.2, but additionally extracts the predicted
    terminal state x_N from the solver at each step.

    Returns:
        X:    state trajectory           (n_sim+1, 13)
        U:    input trajectory           (n_sim,   4)
        X_N:  predicted terminal states  (n_sim,  13)
        Ts:   sample time [s]
    """
    print('Compiling acados QIH-NMPC solver — takes ~30s on first run...')
    solver = create_solver(x_ref, N, T_horizon)
    print('Solver ready.')

    print('Creating plant simulator...')
    plant = create_plant_simulator(T_horizon, N)
    print('Plant ready.\n')

    Ts    = T_horizon / N
    n_sim = int(T_sim / Ts)

    X   = np.zeros((n_sim + 1, NX))
    U   = np.zeros((n_sim,     NU))
    X_N = np.zeros((n_sim,     NX))     # predicted terminal state
    X[0] = x0

    for k in range(n_sim):
        # set current state as initial condition
        solver.set(0, 'lbx', X[k])
        solver.set(0, 'ubx', X[k])

        # solve OCP
        status = solver.solve()
        if status not in [0, 2]:
            print(f'  [step {k:3d}] solver status {status} — stopping.')
            X   = X[:k+1]
            U   = U[:k]
            X_N = X_N[:k]
            break

        # extract first optimal input
        U[k] = solver.get(0, 'u')

        # extract predicted terminal state x_N
        X_N[k] = solver.get(N, 'x')

        # simulate plant one step
        plant.set('x', X[k])
        plant.set('u', U[k])
        plant.solve()
        X[k+1] = normalize_quaternion(plant.get('x'))

        if k % 20 == 0:
            print(f'  step {k:3d}/{n_sim}  |  '
                  f'px={X[k,0]:+.3f}  py={X[k,1]:+.3f}  pz={X[k,2]:+.3f}  |  '
                  f'qw={X[k,6]:+.4f}')

    print('\nSimulation complete.')
    return X, U, X_N, Ts


# ─────────────────────────────────────────────────────────────────
# Plotting — State Trajectories
# ─────────────────────────────────────────────────────────────────
def plot_states(X:     np.ndarray,
                Ts:    float,
                x_ref: np.ndarray,
                title: str = 'Stage 2.1 QIH-NMPC — State Trajectory',
                save_path: str = 'results/stage2_1_qih/states.png'):
    """Plot state trajectories (position, velocity, attitude, angular rates)."""
    t_x = np.arange(X.shape[0]) * Ts

    euler_traj = np.rad2deg(quat_to_euler(X[:, 6:10]))
    euler_ref  = np.rad2deg(quat_to_euler(x_ref[6:10]))

    fig, axes = plt.subplots(4, 3, figsize=(14, 12))
    fig.suptitle(title, fontsize=14)

    # Row 0: position
    pos_labels = ['px [m]', 'py [m]', 'pz [m]']
    for i in range(3):
        ax = axes[0][i]
        ax.plot(t_x, X[:, i], 'b', linewidth=1.2, label='state')
        ax.axhline(x_ref[i], color='r', linestyle='--', linewidth=1, label='ref')
        ax.set_ylabel(pos_labels[i])
        ax.set_xlabel('t [s]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Row 1: velocity
    vel_labels = ['vx [m/s]', 'vy [m/s]', 'vz [m/s]']
    for i in range(3):
        ax = axes[1][i]
        ax.plot(t_x, X[:, 3 + i], 'b', linewidth=1.2, label='state')
        ax.axhline(x_ref[3 + i], color='r', linestyle='--', linewidth=1, label='ref')
        ax.set_ylabel(vel_labels[i])
        ax.set_xlabel('t [s]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Row 2: attitude (Euler angles)
    att_labels = ['roll φ [°]', 'pitch θ [°]', 'yaw ψ [°]']
    for i in range(3):
        ax = axes[2][i]
        ax.plot(t_x, euler_traj[:, i], 'b', linewidth=1.2, label='state')
        ax.axhline(euler_ref[i], color='r', linestyle='--', linewidth=1, label='ref')
        ax.set_ylabel(att_labels[i])
        ax.set_xlabel('t [s]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Row 3: angular rates
    rate_labels = ['p [rad/s]', 'q [rad/s]', 'r [rad/s]']
    for i in range(3):
        ax = axes[3][i]
        ax.plot(t_x, X[:, 10 + i], 'b', linewidth=1.2, label='state')
        ax.axhline(x_ref[10 + i], color='r', linestyle='--', linewidth=1, label='ref')
        ax.set_ylabel(rate_labels[i])
        ax.set_xlabel('t [s]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────
# Plotting — Input Trajectories
# ─────────────────────────────────────────────────────────────────
def plot_inputs(U:     np.ndarray,
                Ts:    float,
                title: str = 'Stage 2.1 QIH-NMPC — Input Trajectory',
                save_path: str = 'results/stage2_1_qih/inputs.png'):
    """Plot motor thrusts and total thrust."""
    t_u = np.arange(U.shape[0]) * Ts

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(title, fontsize=14)

    # Left: individual motor thrusts
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    ax = axes[0]
    for i in range(NU):
        ax.step(t_u, U[:, i], where='post', color=colors[i],
                linewidth=1.2, label=f'f{i+1}')
    ax.axhline(f_hover, color='k', linestyle='--', linewidth=1,
               alpha=0.5, label='f_hover')
    ax.set_ylabel('thrust [N]')
    ax.set_xlabel('t [s]')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Right: total thrust
    ax = axes[1]
    ax.step(t_u, U.sum(axis=1), where='post', color='purple',
            linewidth=1.2, label='T_total')
    ax.axhline(4 * f_hover, color='k', linestyle='--', linewidth=1,
               alpha=0.5, label='mg')
    ax.set_ylabel('total thrust [N]')
    ax.set_xlabel('t [s]')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────
# Plotting — Terminal Constraint  (QIH-specific diagnostic)
# ─────────────────────────────────────────────────────────────────
def plot_terminal_constraint(X_N:    np.ndarray,
                             x_ref:  np.ndarray,
                             P:      np.ndarray,
                             alpha:  float,
                             Ts:     float,
                             save_path: str = 'results/stage2_1_qih/terminal_constraint.png'):
    """
    Plot V_N(k) = (x_N(k) - x_ref)^T P (x_N(k) - x_ref) over time,
    compared to the terminal set radius alpha.

    This is the key QIH diagnostic:
        - V_N <= alpha  →  terminal constraint satisfied, QIH guarantee active
        - V_N >  alpha  →  slack variable is nonzero, guarantee lost

    You will typically see V_N >> alpha during the initial transient,
    decaying below alpha only near steady state — demonstrating that
    the QIH terminal set is too small for practical operation.
    """
    n_sim = X_N.shape[0]
    t = np.arange(n_sim) * Ts

    # Compute V_N = (x_N - x_ref)^T P (x_N - x_ref) at each step
    V_N = np.zeros(n_sim)
    for k in range(n_sim):
        dx = X_N[k] - x_ref
        V_N[k] = dx @ P @ dx

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    fig.suptitle('Stage 2.1 — QIH Terminal Constraint Diagnostic', fontsize=14)

    # ── Top: V_N vs alpha (log scale) ──────────────────────────
    ax = axes[0]
    ax.semilogy(t, V_N, 'b', linewidth=1.2,
                label=r'$V_N = \Delta x_N^\top P \, \Delta x_N$')
    ax.axhline(alpha, color='r', linestyle='--', linewidth=1.5,
               label=rf'$\alpha = {alpha:.4f}$')
    ax.set_ylabel(r'$V_N$  (log scale)')
    ax.set_xlabel('t [s]')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Terminal cost V_N vs. allowed region α')

    # ── Bottom: V_N / alpha ratio ──────────────────────────────
    ax = axes[1]
    ratio = V_N / alpha
    ax.semilogy(t, ratio, 'b', linewidth=1.2)
    ax.axhline(1.0, color='r', linestyle='--', linewidth=1.5,
               label='boundary (ratio = 1)')
    ax.fill_between(t, 0, 1, alpha=0.1, color='green',
                    transform=ax.get_xaxis_transform(),
                    label='feasible region')
    ax.set_ylabel(r'$V_N / \alpha$  (log scale)')
    ax.set_xlabel('t [s]')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Constraint violation ratio (>1 means slack is active)')

    # Print summary
    n_violated = np.sum(V_N > alpha)
    print(f'\n  Terminal constraint summary:')
    print(f'    alpha               = {alpha:.6f}')
    print(f'    V_N range           = [{V_N.min():.4f},  {V_N.max():.4f}]')
    print(f'    max V_N / alpha     = {ratio.max():.1f}x')
    print(f'    steps violated      = {n_violated}/{n_sim}  '
          f'({100*n_violated/n_sim:.0f}%)')
    print(f'    steps feasible      = {n_sim - n_violated}/{n_sim}  '
          f'({100*(n_sim - n_violated)/n_sim:.0f}%)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────
# 3D Trajectory Plot
# ─────────────────────────────────────────────────────────────────
def plot_3d_trajectory(X:     np.ndarray,
                       x_ref: np.ndarray,
                       save_path: str = 'results/stage2_1_qih/trajectory_3d.png'):
    """Plot the 3D flight path of the quadrotor."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(X[:, 0], X[:, 1], X[:, 2], 'b-', linewidth=1.5, label='trajectory')
    ax.scatter(*X[0, :3],  color='green', s=100, marker='o', label='start')
    ax.scatter(*x_ref[:3], color='red',   s=100, marker='*', label='target')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title('Stage 2.1 QIH-NMPC — 3D Quadrotor Flight Path')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # ── Weight matrices (same as solver) ────────────────────────
    Q = np.diag([
        80.,  80.,  120.,
        10.,  10.,   15.,
        10., 120.,  120.,  80.,
         1.,   1.,    1.
    ])
    R = np.diag([0.1, 0.1, 0.1, 0.1])
    f_max = 3.0 * f_hover

    # ── Compute QIH parameters for plotting ─────────────────────
    print('Computing QIH offline parameters...\n')
    P_qih, alpha_qih, kappa, K_red = \
        compute_qih_offline_parameters(Q, R, f_max)

    # ── Reference: hover at (5, 0.5, 6.5), level attitude ──────
    x_ref = np.array([
        5.0, 0.5, 6.5,
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ])

    # ── Initial state: origin, level, stationary ────────────────
    x0    = np.zeros(NX)
    x0[6] = 1.0               # qw = 1

    # ── Run simulation ──────────────────────────────────────────
    X, U, X_N, Ts = simulate(x0, x_ref, N=20, T_horizon=1.0, T_sim=5.0)

    # ── Plots ───────────────────────────────────────────────────
    results_dir = os.path.join('results', 'stage2_1_qih')
    os.makedirs(results_dir, exist_ok=True)

    plot_states(X, Ts, x_ref)
    plot_inputs(U, Ts)
    plot_terminal_constraint(X_N, x_ref, P_qih, alpha_qih, Ts)
    plot_3d_trajectory(X, x_ref)
