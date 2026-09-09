"""
simulation_main.py — Closed-Loop Simulation + Plotting
========================================================
3D Quadrotor NMPC simulation (quaternion model, 13 states).

Imports:
    quadrotor_3d_model.py   → create_plant_simulator(), normalize_quaternion(),
                               quat_to_euler(), f_hover, NX, NU
    nmpc_solver_creator.py  → create_solver()

Extending in future stages:
    Stage 3 → calls EKF estimator at each step
    Stage 4 → obstacle avoidance constraints
    Stage 5 → time-varying reference trajectory
"""

import numpy as np
import matplotlib.pyplot as plt

from quadrotor_3d_model   import (create_plant_simulator, normalize_quaternion,
                                   quat_to_euler, f_hover, NX, NU)
from nmpc_solver_creator  import create_solver


# ─────────────────────────────────────────────────────────────────
# Closed-Loop Simulation
# ─────────────────────────────────────────────────────────────────
def simulate(x0:        np.ndarray,
             x_ref:     np.ndarray,
             N:         int   = 20,
             T_horizon: float = 1.0,
             T_sim:     float = 5.0):
    """
    Closed-loop NMPC simulation with AcadosSimSolver as plant.

    At each timestep:
        1. Fix current state as initial condition
        2. Solve OCP → optimal input sequence
        3. Apply u*[0] to AcadosSim plant
        4. Normalize quaternion to prevent drift
        5. Advance to next step (receding horizon)

    Args:
        x0:         initial state (13,)
                     [px,py,pz, vx,vy,vz, qw,qx,qy,qz, p,q,r]
        x_ref:      reference state (13,)
        N:          prediction horizon steps
        T_horizon:  prediction horizon duration [s]
        T_sim:      total simulation time [s]

    Returns:
        X:  state trajectory   (n_sim+1, 13)
        U:  input trajectory   (n_sim,   4)
        Ts: sample time [s]
    """
    print('Compiling acados solver — takes ~30s on first run...')
    solver = create_solver(x_ref, N, T_horizon)
    print('Solver ready.')

    print('Creating plant simulator...')
    plant = create_plant_simulator(T_horizon, N)
    print('Plant ready.\n')

    Ts    = T_horizon / N
    n_sim = int(T_sim / Ts)

    X = np.zeros((n_sim + 1, NX))
    U = np.zeros((n_sim,     NU))
    X[0] = x0

    for k in range(n_sim):
        # set current state as initial condition
        solver.set(0, 'lbx', X[k])
        solver.set(0, 'ubx', X[k])

        # solve OCP
        status = solver.solve()
        if status not in [0, 2]:
            print(f'  [step {k:3d}] solver status {status} — stopping.')
            break

        # extract first optimal input
        U[k] = solver.get(0, 'u')

        # simulate plant one step
        plant.set('x', X[k])
        plant.set('u', U[k])
        plant.solve()
        X[k+1] = normalize_quaternion(plant.get('x'))

        if k % 20 == 0:
            print(f'  step {k:3d}/{n_sim}  |  '
                  f'px={X[k,0]:+.3f}  py={X[k,1]:+.3f}  pz={X[k,2]:+.3f}  |  '
                  f'qw={X[k,6]:+.4f}  qx={X[k,7]:+.4f}  qy={X[k,8]:+.4f}  qz={X[k,9]:+.4f}')

    print('\nSimulation complete.')
    return X, U, Ts


# ─────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────
def plot_states(X:     np.ndarray,
                Ts:    float,
                x_ref: np.ndarray,
                title: str = 'Stage 1 — State Trajectory (Quaternion)',
                save_path: str = 'results_states.png'):
    """
    Plot state trajectories only.

    The internal state uses quaternions, but the attitude row is plotted as
    Euler angles (roll/pitch/yaw in degrees) for human readability.
    Conversion: quat_to_euler([qw, qx, qy, qz]) → [φ, θ, ψ] via ZYX.

    Layout (4 rows × 3 columns):
        Row 0:  px, py, pz             (position)
        Row 1:  vx, vy, vz             (velocity)
        Row 2:  φ,  θ,  ψ              (attitude — converted from quaternion)
        Row 3:  p,  q,  r              (angular rates)
    """
    t_x = np.arange(X.shape[0]) * Ts

    # ── Convert quaternion trajectory → Euler angles [deg] ─────
    euler_traj = np.rad2deg(quat_to_euler(X[:, 6:10]))   # (n_sim+1, 3)
    euler_ref  = np.rad2deg(quat_to_euler(x_ref[6:10]))  # (3,)

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


def plot_inputs(U:     np.ndarray,
                Ts:    float,
                title: str = 'Stage 1 — Input Trajectory',
                save_path: str = 'results_inputs.png'):
    """
    Plot input (motor thrust) trajectories only.

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
# 3D Trajectory Plot
# ─────────────────────────────────────────────────────────────────
def plot_3d_trajectory(X:     np.ndarray,
                       x_ref: np.ndarray,
                       save_path: str = 'trajectory_3d.png'):
    """Plot the 3D flight path of the quadrotor."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(X[:, 0], X[:, 1], X[:, 2], 'b-', linewidth=1.5, label='trajectory')
    ax.scatter(*X[0, :3],  color='green', s=100, marker='o', label='start')
    ax.scatter(*x_ref[:3], color='red',   s=100, marker='*', label='target')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title('3D Quadrotor Flight Path')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # reference: hover at (1.0, 0.5, 1.5), level attitude, zero velocity
    #   quaternion [1, 0, 0, 0] = identity rotation (level hover)
    x_ref = np.array([
        1.0, 0.5, 1.5,        # px, py, pz
        0.0, 0.0, 0.0,        # vx, vy, vz
        1.0, 0.0, 0.0, 0.0,   # qw, qx, qy, qz   (identity = level)
        0.0, 0.0, 0.0         # p, q, r
    ])

    # initial state: at origin, level, stationary
    #   qw = 1 for a valid unit quaternion at rest
    x0       = np.zeros(NX)
    x0[6]    = 1.0             # qw = 1  (identity quaternion)

    X, U, Ts = simulate(x0, x_ref, N=20, T_horizon=1.0, T_sim=5.0)

    plot_states(X, Ts, x_ref)
    plot_inputs(U, Ts)
    plot_3d_trajectory(X, x_ref)