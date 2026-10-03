"""
simulate_qih.py — Stage 2a Closed-Loop QIH-NMPC Simulation
==========================================================
Stage 2a:  Complete Quasi-Infinite Horizon NMPC

3D Quadrotor QIH-NMPC simulation (quaternion model, 13 states).

Key addition over Stage 1 simulation:
    Extracts the predicted terminal state x_N at each step and computes
    V_N = (x_N - x_ref)^T P (x_N - x_ref) to show when the terminal
    set constraint is active, satisfied, or violated (slack > 0).

Imports:
    quadrotor_3d_model.py       → plant (p = 0 → nominal), utilities
    ocp_config_qih.py           → QIH-NMPC solver + get_qih_params()  (Stage 2a)
    plot_utils.py               → shared plotting functions
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')    # before the solver libraries load

import numpy as np

from quadrotor_3d_model         import (create_disturbance_plant,
                                        quat_normalize,
                                        NX, NU)
from ocp_config_qih             import create_solver, get_qih_params
from plot_utils                 import (plot_states, plot_inputs,
                                        plot_3d_trajectory,
                                        plot_terminal_constraint)

RESULTS_DIR = os.path.join('results', 'stage2a_qih')


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

    Same structure as other stages, but additionally extracts the predicted
    terminal state x_N from the solver at each step.

    Returns:
        X:    state trajectory           (n_sim+1, 13)
        U:    input trajectory           (n_sim,   4)
        X_N:  predicted terminal states  (n_sim,  13)
        Ts:   sample time [s]
    """
    Ts    = T_horizon / N
    n_sim = int(T_sim / Ts)

    print('Compiling acados QIH-NMPC solver — takes ~30s on first run...')
    solver = create_solver(x_ref, N, T_horizon)
    print('Solver ready.')

    print('Creating plant simulator...')
    plant = create_disturbance_plant(Ts)            # p = 0: no disturbance in Stage 2
    print('Plant ready.\n')

    X   = np.zeros((n_sim + 1, NX))
    U   = np.zeros((n_sim,     NU))
    X_N = np.zeros((n_sim,     NX))
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
        X[k+1] = quat_normalize(plant.get('x'))

        if k % 20 == 0:
            print(f'  step {k:3d}/{n_sim}  |  '
                  f'px={X[k,0]:+.3f}  py={X[k,1]:+.3f}  pz={X[k,2]:+.3f}  |  '
                  f'qw={X[k,6]:+.4f}')

    print('\nSimulation complete.')
    return X, U, X_N, Ts


# ─────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # ── Reference: hover at (5, 0.5, 6.5), level attitude ──────
    x_ref = np.array([
        5.0, 0.5, 6.5,
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ])

    # ── Initial state: origin, level, stationary ────────────────
    x0    = np.zeros(NX)
    x0[6] = 1.0

    # ── Run simulation ──────────────────────────────────────────
    X, U, X_N, Ts = simulate(x0, x_ref, N=20, T_horizon=1.0, T_sim=5.0)

    # ── QIH parameters — the same (cached) ones the solver used ─
    P_qih, alpha_qih, _, _ = get_qih_params()

    # ── Plots ───────────────────────────────────────────────────
    plot_states(X, Ts, x_ref,
                title='Stage 2a QIH-NMPC — State Trajectory',
                save_path=os.path.join(RESULTS_DIR, 'states.png'))
    plot_inputs(U, Ts,
                title='Stage 2a QIH-NMPC — Input Trajectory',
                save_path=os.path.join(RESULTS_DIR, 'inputs.png'))
    plot_terminal_constraint(X_N, x_ref, P_qih, alpha_qih, Ts,
                             title='Stage 2a — QIH Terminal Constraint Diagnostic',
                             save_path=os.path.join(RESULTS_DIR, 'terminal-constraint_diagnostic.png'))
    plot_3d_trajectory(X, x_ref,
                       title='Stage 2a QIH-NMPC — 3D Flight Path',
                       save_path=os.path.join(RESULTS_DIR, 'trajectory_3d.png'))
