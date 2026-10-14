"""
plot_utils.py — All Stages Shared Plotting Functions
====================================================
Every figure of the project is drawn here; simulation scripts only choose
titles and paths.

    plot_states, plot_inputs, plot_3d_trajectory     all stages
    plot_terminal_constraint                         V_N vs α       (Stage 2a)
    plot_disturbance                                 d̂ vs d        (Stage 3+)
    plot_velocity_estimation                         v̂ vs v        (Stage 4)
    plot_solve_times                                 estimator vs MPC compute time
    plot_estimator_comparison                        EKF vs MHE     (Stage 4)
    plot_artificial_reference                        p, p_a*, r     (Stage 5)
    plot_tracking_error                              ‖p − p_r‖, ∠(q, q_r)  (Stage 6)

All functions require explicit title and save_path — no stage-specific
defaults here, so each simulation script controls its own labeling.
Stage-specific extras are optional keyword arguments that default to None
(e.g. Stage 3's X_s / t_switch), so callers that omit them are unaffected.

Display (env var QUAD_PLOTS, read at each plot):
    show  (default)  every figure is saved at once and stays open WITHOUT
                     blocking; one plt.show() at interpreter exit opens all
                     windows together — the run is never held up by a window
    none             save only, each figure is closed (batch runs); also the
                     behavior on a non-interactive backend such as Agg
"""

import os
import atexit
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from quadrotor_3d_model import quat_to_euler, f_hover, NU


def _ensure_dir(save_path: str) -> None:
    """Create the parent directory of save_path if it doesn't exist."""
    d = os.path.dirname(save_path)
    if d:
        os.makedirs(d, exist_ok=True)


_NON_INTERACTIVE = {'agg', 'pdf', 'ps', 'svg', 'cairo', 'pgf', 'template'}
_show_at_exit    = False


def _show() -> None:
    """
    Hand the just-saved current figure to the display policy (QUAD_PLOTS).
    A blocking plt.show() per figure would stall the run until each window
    is closed; instead the figure stays open and _show_all, registered once
    with atexit, shows every open figure in a single blocking call.
    """
    global _show_at_exit
    mode = os.environ.get('QUAD_PLOTS', 'show').lower()
    if mode == 'none' or matplotlib.get_backend().lower() in _NON_INTERACTIVE:
        plt.close(plt.gcf())
        return
    if not _show_at_exit:
        # registered after pyplot's own atexit cleanup → runs before it (LIFO)
        atexit.register(_show_all)
        _show_at_exit = True


def _show_all() -> None:
    """Open all figures together; returns when the last window is closed."""
    if plt.get_fignums():
        print(f'\n{len(plt.get_fignums())} figure window(s) open — '
              'close them to exit (QUAD_PLOTS=none: save only).')
        plt.show()


def _mark_switch(ax, t_switch: float, label: str) -> None:
    """Shade t ≥ t_switch and draw the switching instant (no-op for None)."""
    if t_switch is None:
        return
    xlim = ax.get_xlim()
    ax.axvspan(t_switch, xlim[1], alpha=0.05, color='green')
    ax.axvline(t_switch, color='green', ls='-.', lw=1, alpha=0.8, label=label)
    ax.set_xlim(xlim)                   # the span must not widen the axis


# ─────────────────────────────────────────────────────────────────
# State Trajectories
# ─────────────────────────────────────────────────────────────────
def plot_states(X:            np.ndarray,
                Ts:           float,
                x_ref:        np.ndarray,
                title:        str,
                save_path:    str,
                X_s:          np.ndarray = None,
                t_switch:     float      = None,
                switch_label: str        = 'switch',
                Y:            np.ndarray = None):
    """
    Plot state trajectories (4 rows × 3 columns).

    The internal state uses quaternions, but the attitude row is plotted as
    Euler angles (roll/pitch/yaw in degrees) for human readability.

    x_ref is a constant set point (13,) — or, Stage 6, a time-varying
    reference (m, 13) on the same grid Ts, drawn as a dashed curve.

    Optional (Stage 3):
        X_s       (n, 13) equilibrium targets from ss_target — drawn on the
                  attitude row, the only one where they differ from x_ref
                  (p_s = p_ref, v_s = ω_s = 0)
        t_switch  [s] instant to mark (offset-free activation)

    Optional (Stage 6):
        Y         (n, 10) noisy measurements y = [p; q; ω] on the grid Ts —
                  dots on the position, attitude and rate rows (v is not
                  measured)

    Layout:
        Row 0:  px, py, pz             (position)
        Row 1:  vx, vy, vz             (velocity)
        Row 2:  φ,  θ,  ψ              (attitude — converted from quaternion)
        Row 3:  p,  q,  r              (angular rates)
    """
    t_x = np.arange(X.shape[0]) * Ts

    X_ref      = np.atleast_2d(x_ref)              # (1, 13) set point or (m, 13)
    t_r        = np.arange(X_ref.shape[0]) * Ts
    euler_traj = np.rad2deg(quat_to_euler(X[:, 6:10]))
    euler_ref  = np.rad2deg(quat_to_euler(X_ref[:, 6:10]))

    def ref_line(ax, r):
        if len(r) == 1:
            ax.axhline(r[0], color='r', linestyle='--', linewidth=1, label='ref')
        else:
            ax.plot(t_r, r, 'r--', linewidth=1, label='ref')
    if X_s is not None:
        t_s     = np.arange(X_s.shape[0]) * Ts
        euler_s = np.rad2deg(quat_to_euler(X_s[:, 6:10]))
    if Y is not None:
        t_y     = np.arange(Y.shape[0]) * Ts
        euler_y = np.rad2deg(quat_to_euler(Y[:, 3:7]))

    def meas_dots(ax, y):
        if Y is not None:
            ax.plot(t_y, y, '.', color='darkgoldenrod', ms=2.5, alpha=0.6, label='measured', zorder=1)

    fig, axes = plt.subplots(4, 3, figsize=(14, 12))
    fig.suptitle(title, fontsize=14)

    # ── Row 0: position ────────────────────────────────────────
    pos_labels = ['px [m]', 'py [m]', 'pz [m]']
    for i in range(3):
        ax = axes[0][i]
        meas_dots(ax, None if Y is None else Y[:, i])
        ax.plot(t_x, X[:, i], 'b', linewidth=1.2, label='state')
        ref_line(ax, X_ref[:, i])
        _mark_switch(ax, t_switch, switch_label)
        ax.set_ylabel(pos_labels[i])
        ax.set_xlabel('t [s]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # ── Row 1: velocity ────────────────────────────────────────
    vel_labels = ['vx [m/s]', 'vy [m/s]', 'vz [m/s]']
    for i in range(3):
        ax = axes[1][i]
        ax.plot(t_x, X[:, 3 + i], 'b', linewidth=1.2, label='state')
        ref_line(ax, X_ref[:, 3 + i])
        _mark_switch(ax, t_switch, switch_label)
        ax.set_ylabel(vel_labels[i])
        ax.set_xlabel('t [s]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # ── Row 2: attitude (Euler angles from quaternion) ─────────
    att_labels = ['roll φ [°]', 'pitch θ [°]', 'yaw ψ [°]']
    for i in range(3):
        ax = axes[2][i]
        meas_dots(ax, None if Y is None else euler_y[:, i])
        ax.plot(t_x, euler_traj[:, i], 'b', linewidth=1.2, label='state')
        ref_line(ax, euler_ref[:, i])
        if X_s is not None:
            ax.plot(t_s, euler_s[:, i], 'g--', linewidth=1, alpha=0.8, label='eq. target')
        _mark_switch(ax, t_switch, switch_label)
        ax.set_ylabel(att_labels[i])
        ax.set_xlabel('t [s]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # ── Row 3: angular rates ───────────────────────────────────
    rate_labels = ['p [rad/s]', 'q [rad/s]', 'r [rad/s]']
    for i in range(3):
        ax = axes[3][i]
        meas_dots(ax, None if Y is None else Y[:, 7 + i])
        ax.plot(t_x, X[:, 10 + i], 'b', linewidth=1.2, label='state')
        ref_line(ax, X_ref[:, 10 + i])
        _mark_switch(ax, t_switch, switch_label)
        ax.set_ylabel(rate_labels[i])
        ax.set_xlabel('t [s]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=150)
    _show()
    print(f'Plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────
# Input Trajectories
# ─────────────────────────────────────────────────────────────────
def plot_inputs(U:            np.ndarray,
                Ts:           float,
                title:        str,
                save_path:    str,
                t_switch:     float = None,
                switch_label: str   = 'switch',
                U_ref:        np.ndarray = None):
    """
    Plot input (motor thrust) trajectories.

    Layout (1 row × 2 columns):
        Left:   individual motor thrusts f1-f4 overlaid
        Right:  total thrust T = f1+f2+f3+f4  vs  mg

    Optional (Stage 3): t_switch [s] instant to mark (offset-free activation).
    Optional (Stage 6): U_ref (n, 4) feedforward u_r — dashed, same colors.
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
        if U_ref is not None:
            ax.step(t_u, U_ref[:, i], where='post', color=colors[i],
                    linestyle='--', linewidth=0.9, alpha=0.8)
    ax.axhline(f_hover, color='k', linestyle='--', linewidth=1, alpha=0.5, label='f_hover')
    _mark_switch(ax, t_switch, switch_label)
    ax.set_ylabel('thrust [N]')
    ax.set_xlabel('t [s]')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Right: total thrust ────────────────────────────────────
    ax = axes[1]
    T_total = U.sum(axis=1)
    ax.step(t_u, T_total, where='post', color='purple', linewidth=1.2, label='T_total')
    if U_ref is not None:
        ax.step(t_u, U_ref.sum(axis=1), where='post', color='purple',
                linestyle='--', linewidth=0.9, alpha=0.8, label='T_ref (u_r)')
    ax.axhline(4 * f_hover, color='k', linestyle='--', linewidth=1, alpha=0.5, label='mg')
    _mark_switch(ax, t_switch, switch_label)
    ax.set_ylabel('total thrust [N]')
    ax.set_xlabel('t [s]')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=150)
    _show()
    print(f'Plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────
# 3D Trajectory
# ─────────────────────────────────────────────────────────────────
def plot_3d_trajectory(X:         np.ndarray,
                       x_ref:     np.ndarray,
                       title:     str,
                       save_path: str,
                       P_meas:    np.ndarray = None):
    """
    Plot the 3D flight path of the quadrotor.

    x_ref: target set point (≥3,) — or, Stage 6, a reference path (m, ≥3),
    drawn as a dashed curve with its end point as the target.
    Optional (Stage 6): P_meas (n, 3) noisy position measurements (dark-yellow dots).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if P_meas is not None:
        ax.scatter(P_meas[:, 0], P_meas[:, 1], P_meas[:, 2], s=2, c='darkgoldenrod',
                   alpha=0.5, depthshade=False, label='measured p')

    ax.plot(X[:, 0], X[:, 1], X[:, 2], 'b-', linewidth=1.5, label='trajectory')
    if np.ndim(x_ref) == 2:
        ax.plot(x_ref[:, 0], x_ref[:, 1], x_ref[:, 2], 'r--', linewidth=1, label='reference')
        x_ref = x_ref[-1]
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
    _show()
    print(f'Plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────
# Terminal Constraint Diagnostic  (Stage 2a — QIH)
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

    dX  = X_N - x_ref
    V_N = np.einsum('ki,ij,kj->k', dX, P, dX)

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
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=150)
    _show()
    print(f'Plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────
# Disturbance Estimation  (Stage 3+)
# ─────────────────────────────────────────────────────────────────
def plot_disturbance(D_hat:        np.ndarray,
                     D_true:       np.ndarray,
                     Ts:           float,
                     title:        str,
                     save_path:    str,
                     t_switch:     float = None,
                     switch_label: str   = 'switch'):
    """
    Plot estimated vs true disturbance over time (2 rows × 3 columns).

    Layout:
        Row 0:  d_fx, d_fy, d_fz   (force disturbances)
        Row 1:  d_τx, d_τy, d_τz   (torque disturbances)

    Used by any stage with disturbance estimation (EKF or MHE).
    Optional (Stage 3): t_switch [s] instant to mark (offset-free activation).
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
        _mark_switch(ax, t_switch, switch_label)
        ax.set_ylabel(force_labels[i])
        ax.set_xlabel('t [s]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[1][i]
        ax.plot(t, D_true[:, 3+i], 'r--', lw=1.5, label='true')
        ax.plot(t, D_hat[:, 3+i],  'b',   lw=1.2, label='estimated')
        _mark_switch(ax, t_switch, switch_label)
        ax.set_ylabel(torque_labels[i])
        ax.set_xlabel('t [s]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=150)
    _show()
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
    _show()
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
    _show()
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
    _show()
    print(f'Plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────
# Artificial Reference  (Stage 5 — tracking MPC)
# ─────────────────────────────────────────────────────────────────
def plot_artificial_reference(X:         np.ndarray,
                              P_a:       np.ndarray,
                              R:         np.ndarray,
                              R_star:    np.ndarray,
                              Ts:        float,
                              y_lb:      np.ndarray,
                              y_ub:      np.ndarray,
                              title:     str,
                              save_path: str):
    """
    The key Stage 5 figure: how the artificial reference p_a* moves.

    X       (n+1, ≥13)  closed-loop states (position = X[:, :3])
    P_a     (n, 3)      optimal artificial reference p_a*(t) per MPC call
    R       (n, 3)      external reference r(t)
    R_star  (n, 3)      optimal reachable reference y_r*(t) = argmin_{Y_s} V_o
    y_lb/ub (3,)        bounds of Y_s (shaded outside)

    Expected (convergence to the optimal reachable steady state, Limon et
    al., IEEE TAC 2018): p_a* slides from p toward r and settles at y_r*;
    p follows p_a*. If r ∉ Y_s, p_a* stops at the boundary of Y_s.
    """
    t_x = np.arange(X.shape[0]) * Ts
    t_k = np.arange(P_a.shape[0]) * Ts

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(title, fontsize=14)
    labels = ['x [m]', 'y [m]', 'z [m]']

    for i in range(3):
        ax = axes[i]
        ax.step(t_k, R[:, i],      'r--', where='post', lw=1.3, label='r (reference)')
        ax.step(t_k, R_star[:, i], 'k:',  where='post', lw=1.5, label='y_r* (best reachable)')
        ax.plot(t_k, P_a[:, i],    'g',   lw=1.5, label='p_a* (artificial)')
        ax.plot(t_x, X[:, i],      'b',   lw=1.2, label='p (state)')
        lo, hi = ax.get_ylim()
        if y_ub[i] < hi:                          # shade only bounds in view
            ax.axhspan(y_ub[i], hi, color='gray', alpha=0.12, label='outside Y_s')
        if y_lb[i] > lo:
            ax.axhspan(lo, y_lb[i], color='gray', alpha=0.12)
        ax.set_ylim(lo, hi)
        ax.set_ylabel(labels[i])
        ax.set_xlabel('t [s]')
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=150)
    _show()
    print(f'Plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────
# Tracking Error  (Stage 6 — time-varying reference)
# ─────────────────────────────────────────────────────────────────
def plot_tracking_error(X:         np.ndarray,
                        X_r:       np.ndarray,
                        Ts:        float,
                        title:     str,
                        save_path: str,
                        t_marks:   list = None):
    """
    Position and attitude tracking error against a time-varying reference.

    X    (n+1, ≥13)  closed-loop states;  X_r (n, 13) reference at the same
                     instants (rows 0..n−1 are compared)
    t_marks          optional [(t, label), ...] phase boundaries to mark

    Left : ‖p − p_r‖ [m]                (log scale — shows the transient
                                          and the residual level together)
    Right: attitude error angle [°] = 2·arccos|⟨q, q_r⟩|  (double-cover safe)
    """
    n   = X_r.shape[0]
    t   = np.arange(n) * Ts
    e_p = np.linalg.norm(X[:n, :3] - X_r[:, :3], axis=1)
    q   = X[:n, 6:10] / np.linalg.norm(X[:n, 6:10], axis=1, keepdims=True)
    dot = np.abs(np.sum(q * X_r[:, 6:10], axis=1))
    e_q = np.rad2deg(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))

    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    fig.suptitle(title, fontsize=14)
    for ax, e, lab in ((axes[0], e_p, '‖p − p_r‖ [m]'),
                       (axes[1], e_q, 'attitude error [°]')):
        ax.plot(t, e, 'b', lw=1.2)
        for t_m, label in (t_marks or []):
            ax.axvline(t_m, color='k', ls=':', lw=1)
            ax.text(t_m, 1.0, f' {label}', transform=ax.get_xaxis_transform(),
                    fontsize=8, va='top')
        ax.set_ylabel(lab)
        ax.set_xlabel('t [s]')
        ax.grid(True, which='both', alpha=0.3)
    axes[0].set_yscale('log')

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=150)
    _show()
    print(f'Plot saved: {save_path}')
