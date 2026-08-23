"""
simulation_main.py — Closed-Loop Simulation + Plotting
========================================================
3D Quadrotor NMPC simulation.

Imports:
    quadrotor_3d_model.py   → create_plant_simulator(), NX, NU, f_hover
    nmpc_solver_creator.py  → create_solver()

Extending in future stages:
    Stage 3 → calls EKF estimator at each step
    Stage 5 → time-varying reference trajectory
"""

import numpy as np
import matplotlib.pyplot as plt

from quadrotor_3d_model   import create_plant_simulator, f_hover, NX, NU
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
        4. Advance to next step (receding horizon)

    Args:
        x0:         initial state (12,)
        x_ref:      reference state (12,)
        N:          prediction horizon steps
        T_horizon:  prediction horizon duration [s]
        T_sim:      total simulation time [s]

    Returns:
        X:  state trajectory   (n_sim+1, 12)
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
        X[k+1] = plant.get('x')

        if k % 20 == 0:
            print(f'  step {k:3d}/{n_sim}  |  '
                  f'px={X[k,0]:+.3f}  py={X[k,1]:+.3f}  pz={X[k,2]:+.3f}  |  '
                  f'phi={X[k,6]:+.4f}  theta={X[k,7]:+.4f}  psi={X[k,8]:+.4f}')

    print('\nSimulation complete.')
    return X, U, Ts


# ─────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────
def plot_results(X:     np.ndarray,
                 U:     np.ndarray,
                 Ts:    float,
                 x_ref: np.ndarray,
                 title: str = 'Stage 1 — 3D Quadrotor NMPC: Hover Stabilization',
                 save_path: str = 'results_stage1.png'):
    """
    Plot state and input trajectories.

    Layout:
        Row 1:  px, py, pz         (position)
        Row 2:  vx, vy, vz         (velocity)
        Row 3:  phi, theta, psi    (attitude)
        Row 4:  p, q, r            (angular rates)
        Row 5:  f1, f2, f3, f4     (motor thrusts)
    """
    t_x = np.arange(X.shape[0]) * Ts
    t_u = np.arange(U.shape[0]) * Ts

    state_labels = [
        'px [m]',    'py [m]',    'pz [m]',
        'vx [m/s]',  'vy [m/s]',  'vz [m/s]',
        'φ [rad]',   'θ [rad]',   'ψ [rad]',
        'p [rad/s]', 'q [rad/s]', 'r [rad/s]'
    ]
    input_labels = ['f1 [N]', 'f2 [N]', 'f3 [N]', 'f4 [N]']

    fig, axes = plt.subplots(5, 4, figsize=(16, 14))
    fig.suptitle(title, fontsize=14)

    # ── States: rows 0-3, columns 0-2 ──────────────────────────
    for i in range(NX):
        row = i // 3
        col = i % 3
        ax = axes[row][col]
        ax.plot(t_x, X[:, i], 'b', linewidth=1.2, label='state')
        ax.axhline(x_ref[i], color='r', linestyle='--', linewidth=1, label='ref')
        ax.set_ylabel(state_labels[i])
        ax.set_xlabel('t [s]')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    # hide unused subplot in rows 0-3 (column 3)
    for row in range(4):
        axes[row][3].set_visible(False)

    # ── Inputs: row 4, columns 0-3 ─────────────────────────────
    for i in range(NU):
        ax = axes[4][i]
        ax.step(t_u, U[:, i], where='post', color='g', linewidth=1.2, label=input_labels[i])
        ax.axhline(f_hover, color='r', linestyle='--', linewidth=1, label='f_hover')
        ax.set_ylabel(input_labels[i])
        ax.set_xlabel('t [s]')
        ax.legend(fontsize=6)
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
    x_ref = np.array([
        1.0, 0.5, 1.5,     # px, py, pz
        0.0, 0.0, 0.0,     # vx, vy, vz
        0.0, 0.0, 0.0,     # phi, theta, psi
        0.0, 0.0, 0.0      # p, q, r
    ])

    # initial state: at origin, level, stationary
    x0 = np.zeros(NX)

    X, U, Ts = simulate(x0, x_ref, N=20, T_horizon=1.0, T_sim=5.0)

    plot_results(X, U, Ts, x_ref)
    plot_3d_trajectory(X, x_ref)
