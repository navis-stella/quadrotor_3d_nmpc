"""
plot_utils.py — Shared Plotting Functions for Quadrotor NMPC
=============================================================
Used by all simulation scripts (Stage 1 through Stage N).

All functions require explicit title and save_path — no stage-specific
defaults here, so each simulation script controls its own labeling.
"""

import numpy as np
import matplotlib.pyplot as plt

from quadrotor_3d_model import quat_to_euler, f_hover, NU, ND


# ─────────────────────────────────────────────────────────────────
# State Trajectories
# ─────────────────────────────────────────────────────────────────
def plot_states(X:         np.ndarray,
                Ts:        float,
                x_ref:     np.ndarray,
                title:     str,
                save_path: str):
    """
    Plot state trajectories (4 rows × 3 columns).

    The internal state uses quaternions, but the attitude row is plotted as
    Euler angles (roll/pitch/yaw in degrees) for human readability.

    Layout:
        Row 0:  px, py, pz             (position)
        Row 1:  vx, vy, vz             (velocity)
        Row 2:  φ,  θ,  ψ              (attitude — converted from quaternion)
        Row 3:  p,  q,  r              (angular rates)
    """
    t_x = np.arange(X.shape[0]) * Ts

    euler_traj = np.rad2deg(quat_to_euler(X[:, 6:10]))
    euler_ref  = np.rad2deg(quat_to_euler(x_ref[6:10]))

    fig, axes = plt.subplots(4, 3, figsize=(14, 12))
    fig.suptitle(title, fontsize=14)

    # ── Row 0: position ────────────────────────────────────────
    pos_labels = ['px [m]', 'py [m]', 'pz [m]']
    for i in range(3):
        ax = axes[0][i]
        ax.plot(t_x, X[:, i], 'b', linewidth=1.2, label='state')
        ax.axhline(x_ref[i], color='r', linestyle='--', linewidth=1, label='ref')
        ax.set_ylabel(pos_labels[i])
        ax.set_xlabel('t [s]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # ── Row 1: velocity ────────────────────────────────────────
    vel_labels = ['vx [m/s]', 'vy [m/s]', 'vz [m/s]']
    for i in range(3):
        ax = axes[1][i]
        ax.plot(t_x, X[:, 3 + i], 'b', linewidth=1.2, label='state')
        ax.axhline(x_ref[3 + i], color='r', linestyle='--', linewidth=1, label='ref')
        ax.set_ylabel(vel_labels[i])
        ax.set_xlabel('t [s]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # ── Row 2: attitude (Euler angles from quaternion) ─────────
    att_labels = ['roll φ [°]', 'pitch θ [°]', 'yaw ψ [°]']
    for i in range(3):
        ax = axes[2][i]
        ax.plot(t_x, euler_traj[:, i], 'b', linewidth=1.2, label='state')
        ax.axhline(euler_ref[i], color='r', linestyle='--', linewidth=1, label='ref')
        ax.set_ylabel(att_labels[i])
        ax.set_xlabel('t [s]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # ── Row 3: angular rates ───────────────────────────────────
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
# Input Trajectories
# ─────────────────────────────────────────────────────────────────
def plot_inputs(U:         np.ndarray,
                Ts:        float,
                title:     str,
                save_path: str):
    """
    Plot input (motor thrust) trajectories.

    Layout (1 row × 2 columns):
        Left:   individual motor thrusts f1-f4 overlaid
        Right:  total thrust T = f1+f2+f3+f4  vs  mg
    """
    t_u = np.arange(U.shape[0]) * Ts

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(title, fontsize=14)

    # ── Left: individual motor thrusts ─────────────────────────
    input_labels = ['f1', 'f2', 'f3', 'f4']
    colors       = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    ax = axes[0]
    for i in range(NU):
        ax.step(t_u, U[:, i], where='post', color=colors[i],
                linewidth=1.2, label=input_labels[i])
    ax.axhline(f_hover, color='k', linestyle='--', linewidth=1, alpha=0.5, label='f_hover')
    ax.set_ylabel('thrust [N]')
    ax.set_xlabel('t [s]')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Right: total thrust ────────────────────────────────────
    ax = axes[1]
    T_total = U.sum(axis=1)
    ax.step(t_u, T_total, where='post', color='purple', linewidth=1.2, label='T_total')
    ax.axhline(4 * f_hover, color='k', linestyle='--', linewidth=1, alpha=0.5, label='mg')
    ax.set_ylabel('total thrust [N]')
    ax.set_xlabel('t [s]')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────
# 3D Trajectory
# ─────────────────────────────────────────────────────────────────
def plot_3d_trajectory(X:         np.ndarray,
                       x_ref:     np.ndarray,
                       title:     str,
                       save_path: str):
    """Plot the 3D flight path of the quadrotor."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(X[:, 0], X[:, 1], X[:, 2], 'b-', linewidth=1.5, label='trajectory')
    ax.scatter(*X[0, :3],  color='green', s=100, marker='o', label='start')
    ax.scatter(*x_ref[:3], color='red',   s=100, marker='*', label='target')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────
# Terminal Constraint Diagnostic  (QIH-specific)
# ─────────────────────────────────────────────────────────────────
def plot_terminal_constraint(X_N:       np.ndarray,
                             x_ref:     np.ndarray,
                             P:         np.ndarray,
                             alpha:     float,
                             Ts:        float,
                             title:     str,
                             save_path: str):
    """
    Plot V_N(k) = (x_N(k) - x_ref)^T P (x_N(k) - x_ref) over time,
    compared to the terminal set radius alpha.

    This is the key QIH diagnostic:
        V_N <= alpha  →  terminal constraint satisfied
        V_N >  alpha  →  slack variable is nonzero
    """
    n_sim = X_N.shape[0]
    t = np.arange(n_sim) * Ts

    V_N = np.zeros(n_sim)
    for k in range(n_sim):
        dx = X_N[k] - x_ref
        V_N[k] = dx @ P @ dx

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    fig.suptitle(title, fontsize=14)

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
# Disturbance Estimation  (EKF stages)
# ─────────────────────────────────────────────────────────────────
def plot_disturbance(D_hat:     np.ndarray,
                     D_true:    np.ndarray,
                     Ts:        float,
                     title:     str,
                     save_path: str):
    """
    Plot estimated vs true disturbance over time (2 rows × 3 columns).

    Layout:
        Row 0:  d_fx, d_fy, d_fz   (force disturbances)
        Row 1:  d_τx, d_τy, d_τz   (torque disturbances)

    Used by any stage with EKF disturbance estimation.
    """
    n = D_hat.shape[0]
    t = np.arange(n) * Ts

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle(title, fontsize=14)

    force_labels  = ['d_fx [N]', 'd_fy [N]', 'd_fz [N]']
    torque_labels = ['d_τx [N·m]', 'd_τy [N·m]', 'd_τz [N·m]']

    for i in range(3):
        ax = axes[0][i]
        ax.plot(t, D_true[:, i], 'r--', lw=1.5, label='true')
        ax.plot(t, D_hat[:, i],  'b',   lw=1.2, label='estimated')
        ax.set_ylabel(force_labels[i])
        ax.set_xlabel('t [s]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[1][i]
        ax.plot(t, D_true[:, 3+i], 'r--', lw=1.5, label='true')
        ax.plot(t, D_hat[:, 3+i],  'b',   lw=1.2, label='estimated')
        ax.set_ylabel(torque_labels[i])
        ax.set_xlabel('t [s]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Plot saved: {save_path}')
