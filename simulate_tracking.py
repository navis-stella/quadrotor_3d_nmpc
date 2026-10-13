"""
simulate_tracking.py — Stage 5 Closed-Loop Tracking MPC with Artificial Reference
=================================================================================
The tracking MPC flies a piecewise-constant reference sequence on the
nominal plant (no disturbance, full state feedback):

    t =  0 s   r = ( 6, 0, 1)   6 m jump — not reachable within one 1 s horizon
    t =  5 s   r = ( 6, 6, 2)   another jump
    t = 10 s   r = (12, 6, 2)   OUTSIDE the room → no admissible steady state

Expected (Limon et al., IEEE TAC 2018):
    every OCP feasible          recursive feasibility — r is only a cost parameter
    p_a* slides toward r        and settles at y_r* = projection of r onto Y_s
                                (convergence to the optimal reachable steady state)

Imports:
    quadrotor_3d_model.py   → create_disturbance_plant() (p = 0 → nominal plant),
                               quat_normalize(), steady_state_map(), NX, NU
    ocp_config_tracking.py  → create_solver() + runtime helpers, Y_s, y_r*
    plot_utils.py           → plot_artificial_reference(), plot_states(),
                               plot_inputs(), plot_3d_trajectory()

Run:  python simulate_tracking.py   → results/stage5_tracking/
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')    # before the solver libraries load

import time
import numpy as np

from quadrotor_3d_model import (create_disturbance_plant, quat_normalize,
                                steady_state_map, F_MAX, NX, NU)
import ocp_config_tracking as track
from plot_utils import (plot_artificial_reference, plot_states, plot_inputs,
                        plot_3d_trajectory)

RESULTS_DIR = os.path.join('results', 'stage5_tracking')

# ─────────────────────────────────────────────────────────────────
# Scenario
# ─────────────────────────────────────────────────────────────────
N         = 20
T_HORIZON = 1.0
TS        = T_HORIZON / N            # 0.05 s
T_SIM     = 15.0
N_SIM     = int(round(T_SIM / TS))

REF_SCHEDULE = [                     # (t_start [s], r)
    ( 0.0, np.array([ 6.0, 0.0, 1.0])),
    ( 5.0, np.array([ 6.0, 6.0, 2.0])),
    (10.0, np.array([12.0, 6.0, 2.0])),
]
P0 = np.zeros(3)                     # start hovering at the origin


def reference_at(t: float) -> np.ndarray:
    """Piecewise-constant reference r(t)."""
    r = REF_SCHEDULE[0][1]
    for t_start, r_k in REF_SCHEDULE:
        if t >= t_start - 1e-9:
            r = r_k
    return r


# ─────────────────────────────────────────────────────────────────
# Closed Loop
# ─────────────────────────────────────────────────────────────────
def simulate() -> dict:
    """
    Returns dict with X (n+1, 13), U (n, 4), P_a (n, 3), R (n, 3),
    status (n,), T_solve (n,) [s].

    A failed solve (status ≠ 0) is counted, and its first input is still
    applied, clipped to the motor limits (hover if not finite).
    """
    x0, u_hover = steady_state_map(P0)
    plant  = create_disturbance_plant(TS)         # p = 0 → nominal plant
    solver = track.create_solver(N, T_HORIZON)
    track.init_guess(solver, x0, N)

    X   = np.zeros((N_SIM + 1, NX)); X[0] = x0
    U   = np.zeros((N_SIM, NU))
    P_a = np.zeros((N_SIM, 3))
    R   = np.zeros((N_SIM, 3))
    status  = np.zeros(N_SIM, dtype=int)
    T_solve = np.zeros(N_SIM)

    x = x0.copy()
    for k in range(N_SIM):
        r = reference_at(k * TS)
        R[k] = r
        track.set_initial_state(solver, x)
        track.set_reference(solver, r, N)

        t0 = time.perf_counter()
        status[k] = solver.solve()
        T_solve[k] = time.perf_counter() - t0

        u = solver.get(0, 'u')
        if not np.all(np.isfinite(u)):
            u = u_hover
        u = np.clip(u, 0.0, F_MAX)
        P_a[k] = track.get_artificial_reference(solver)

        plant.set('x', x)
        plant.set('u', u)
        plant.solve()
        x = quat_normalize(plant.get('x'))
        X[k + 1], U[k] = x, u

    return dict(X=X, U=U, P_a=P_a, R=R, status=status, T_solve=T_solve)


def summarize(res: dict) -> None:
    fail   = res['status'] != 0
    p_end  = res['X'][-1, :3]
    r_star = track.optimal_reachable_reference(res['R'][-1])
    in_room = (np.all(res['X'][:, :3] <= track.ROOM_UB + 1e-6)
               and np.all(res['X'][:, :3] >= track.ROOM_LB - 1e-6))
    print('\n' + '═' * 60)
    print('  Stage 5 — Tracking MPC with artificial reference')
    print('═' * 60)
    print(f'    failed OCP calls     : {fail.sum()}/{len(fail)}'
          + (f'   (first at t = {np.argmax(fail) * TS:.2f} s)' if fail.any() else ''))
    print(f'    position inside room : {in_room}')
    print(f'    final position       : {np.round(p_end, 3)}')
    print(f'    ‖p_end − y_r*‖       : {np.linalg.norm(p_end - r_star):.4f} m'
          f'   (y_r* = {r_star})')
    print(f'    solve time mean/max  : {1e3 * res["T_solve"].mean():.2f} / '
          f'{1e3 * res["T_solve"].max():.2f} ms   (budget {1e3 * TS:.0f} ms)')


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    res = simulate()
    summarize(res)

    R_star = np.array([track.optimal_reachable_reference(r) for r in res['R']])
    x_final, _ = steady_state_map(R_star[-1])

    plot_artificial_reference(res['X'], res['P_a'], res['R'], R_star,
                              TS, track.Y_LB, track.Y_UB,
                              'Stage 5 — Tracking MPC: artificial reference p_a*',
                              os.path.join(RESULTS_DIR, 'artificial_reference.png'))
    plot_states(res['X'], TS, x_final,
                'Stage 5 — Tracking MPC: states (ref = final y_r*)',
                os.path.join(RESULTS_DIR, 'states.png'))
    plot_inputs(res['U'], TS,
                'Stage 5 — Tracking MPC: motor thrusts',
                os.path.join(RESULTS_DIR, 'inputs.png'))
    plot_3d_trajectory(res['X'], x_final,
                       'Stage 5 — Tracking MPC: 3D trajectory',
                       os.path.join(RESULTS_DIR, 'trajectory_3d.png'))
