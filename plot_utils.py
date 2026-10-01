"""
plot_utils.py — All Stages Shared Plotting Functions
====================================================
Every figure of the project is drawn here; simulation scripts only choose
titles and paths.

    plot_states, plot_inputs, plot_3d_trajectory     all stages
    plot_disturbance                                 d̂ vs d        (Stage 3+)
    plot_velocity_estimation                         v̂ vs v        (Stage 4)
    plot_solve_times                                 estimator vs MPC compute time
    plot_estimator_comparison                        EKF vs MHE     (Stage 4)

All functions require explicit title and save_path — no stage-specific
defaults here, so each simulation script controls its own labeling.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from quadrotor_3d_model import quat_to_euler, f_hover, NU


def _ensure_dir(save_path: str) -> None:
    """Create the parent directory of save_path if it doesn't exist."""
    d = os.path.dirname(save_path)
    if d:
        os.makedirs(d, exist_ok=True)


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
    _ensure_dir(save_path)
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
    _ensure_dir(save_path)
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
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────
# Disturbance Estimation  (Stage 3+)
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

    Used by any stage with disturbance estimation (EKF or MHE).
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
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────
# Velocity Estimation  (Stage 4 — the unmeasured physical state)
# ─────────────────────────────────────────────────────────────────
def plot_velocity_estimation(X:         np.ndarray,
                             X_hat:     np.ndarray,
                             Ts:        float,
                             title:     str,
                             save_path: str):
    """
    Compare true vs estimated linear velocity — the key unmeasured state.

    Position, quaternion, and angular rate are measured directly, so their
    estimates track truth closely (uninteresting). Velocity is the state
    that must be inferred through the model coupling ṗ = v, and its
    convergence is the estimator's actual test.
    """
    t = np.arange(X.shape[0]) * Ts

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=14)

    labels = ['vx [m/s]', 'vy [m/s]', 'vz [m/s]']
    for i in range(3):
        ax = axes[i]
        ax.plot(t, X[:, 3+i],     'r--', lw=1.5, label='true')
        ax.plot(t, X_hat[:, 3+i], 'b',   lw=1.2, label='estimate')
        ax.set_ylabel(labels[i])
        ax.set_xlabel('t [s]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────
# Computation Time  (estimator vs MPC)
# ─────────────────────────────────────────────────────────────────
_C_MPC, _C_EST = '#1f77b4', '#d62728'


def plot_solve_times(T_est:     np.ndarray,
                     T_mpc:     np.ndarray,
                     Ts_est:    float,
                     Ts_mpc:    float,
                     est_name:  str,
                     title:     str,
                     save_path: str):
    """
    Per-call computation time of the estimator and the MPC against their
    real-time budgets (the sample time each one must finish within).

    T_est, T_mpc: wall-clock times [s] on the base grid Ts_est; NaN where
    that component did not run (the MPC runs every Ts_mpc / Ts_est steps).

    Left : time series — shows start-up transients and spikes
    Right: distribution (box plot, whiskers 1–99 %)
    """
    T_est, T_mpc = T_est * 1e3, T_mpc * 1e3
    t      = np.arange(len(T_est)) * Ts_est
    ok_est = ~np.isnan(T_est)
    ok_mpc = ~np.isnan(T_mpc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.5),
                                   gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle(title, fontsize=14)

    ax1.plot(t[ok_est], T_est[ok_est], '.', ms=3, color=_C_EST, label=est_name)
    ax1.plot(t[ok_mpc], T_mpc[ok_mpc], '.', ms=3, color=_C_MPC, label='MPC')
    ax1.axhline(Ts_est * 1e3, color=_C_EST, ls='--', lw=1,
                label=f'budget {est_name}  {Ts_est * 1e3:.0f} ms')
    ax1.axhline(Ts_mpc * 1e3, color=_C_MPC, ls='--', lw=1,
                label=f'budget MPC  {Ts_mpc * 1e3:.0f} ms')
    ax1.set_yscale('log')
    ax1.set_xlabel('t [s]')
    ax1.set_ylabel('computation time [ms]')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, which='both', alpha=0.3)

    ax2.boxplot([T_mpc[ok_mpc], T_est[ok_est]], whis=(1, 99),
                showfliers=True, flierprops={'markersize': 2})
    ax2.set_xticks([1, 2], ['MPC', est_name])
    ax2.set_yscale('log')
    ax2.set_ylabel('computation time [ms]')
    ax2.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────
# Estimator Comparison  (Stage 4 — EKF vs MHE, same closed loop)
# ─────────────────────────────────────────────────────────────────
def plot_estimator_comparison(results:   dict,
                              x_ref:     np.ndarray,
                              title:     str,
                              save_path: str):
    """
    Overlay the runs of several estimators on the identical closed loop.

    results: {name: result dict from closed_loop_sim_config.simulate (or load_result)}
    Panels:  tracking error ‖p − p_ref‖, velocity estimation error ‖v̂ − v‖,
             force-disturbance estimation error ‖d̂_f − d_f‖,
             estimator computation time — all on log scales
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
    fig.suptitle(title, fontsize=14)
    ax_p, ax_v, ax_d, ax_t = axes.ravel()

    for name, r in results.items():
        X, X_hat, D_hat, D_true, Ts = (r['X'], r['X_hat'], r['D_hat'],
                                       r['D_true'], r['Ts'])
        t   = np.arange(X.shape[0]) * Ts
        t_k = np.arange(D_true.shape[0]) * Ts
        ax_p.plot(t,   np.linalg.norm(X[:, :3] - x_ref[:3], axis=1),           lw=1.2, label=name)
        ax_v.plot(t,   np.linalg.norm(X_hat[:, 3:6] - X[:, 3:6], axis=1),      lw=1.0, label=name)
        ax_d.plot(t_k, np.linalg.norm(D_hat[:-1, :3] - D_true[:, :3], axis=1), lw=1.0, label=name)
        ax_t.plot(t_k, r['T_est'] * 1e3, '.', ms=2, label=name)

    Ts = next(iter(results.values()))['Ts']
    ax_t.axhline(Ts * 1e3, color='k', ls='--', lw=1, label=f'budget {Ts * 1e3:.0f} ms')
    for ax, lab in ((ax_p, '‖p − p_ref‖ [m]'), (ax_v, '‖v̂ − v‖ [m/s]'),
                    (ax_d, '‖d̂_f − d_f‖ [N]'), (ax_t, 'estimator time [ms]')):
        ax.set_ylabel(lab)
        ax.set_yscale('log')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8)
    ax_d.set_xlabel('t [s]')
    ax_t.set_xlabel('t [s]')

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Plot saved: {save_path}')
