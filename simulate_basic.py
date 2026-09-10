"""
simulate_basic.py — Closed-Loop Simulation + Plotting
========================================================
Stage 1:  Simple Terminal Cost (W_e = Q)

3D Quadrotor NMPC simulation (quaternion model, 13 states).

Imports:
    quadrotor_3d_model.py   → create_plant_simulator(), NX, NU
    ocp_config_basic.py     → create_solver()
    plot_utils.py           → plot_states(), plot_inputs(), plot_3d_trajectory()
"""

import os
import numpy as np

from quadrotor_3d_model        import (create_plant_simulator, normalize_quaternion,
                                       NX, NU)
from ocp_config_basic          import create_solver
from plot_utils                import plot_states, plot_inputs, plot_3d_trajectory

RESULTS_DIR = os.path.join('results', 'stage1_basic')

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
# Entry Point
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # reference: hover at (1.0, 0.5, 1.5), level attitude, zero velocity
    x_ref = np.array([
        1.0, 0.5, 1.5,        # px, py, pz
        0.0, 0.0, 0.0,        # vx, vy, vz
        1.0, 0.0, 0.0, 0.0,   # qw, qx, qy, qz   (identity = level)
        0.0, 0.0, 0.0         # p, q, r
    ])

    # initial state: at origin, level, stationary
    x0       = np.zeros(NX)
    x0[6]    = 1.0             # qw = 1  (identity quaternion)

    X, U, Ts = simulate(x0, x_ref, N=20, T_horizon=1.0, T_sim=5.0)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    plot_states(X, Ts, x_ref,
                title='Stage 1 — State Trajectory (Simple Terminal Cost)',
                save_path=os.path.join(RESULTS_DIR, 'states.png'))
    plot_inputs(U, Ts,
                title='Stage 1 — Input Trajectory (Simple Terminal Cost)',
                save_path=os.path.join(RESULTS_DIR, 'inputs.png'))
    plot_3d_trajectory(X, x_ref,
                       title='Stage 1 — 3D Flight Path (Simple Terminal Cost)',
                       save_path=os.path.join(RESULTS_DIR, 'trajectory_3d.png'))
