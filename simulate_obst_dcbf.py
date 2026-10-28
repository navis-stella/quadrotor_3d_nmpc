"""
simulate_obst_dcbf.py — Stage 7b Obstacle Avoidance with a Discrete-Time CBF
============================================================================
Same scene and closed loop as Stage 7a (obstacle_sim_config: helix through
three pillars, nominal plant, full state feedback); only the obstacle
constraint changes from the distance form to the discrete-time control
barrier function (DCBF) of Zeng, Zhang & Sreenath (ACC 2021):

    7a   hⱼ(x_i) ≥ 0                         i = 1..N
    7b   hⱼ(x_{i+1}) ≥ (1 − γ)·hⱼ(x_i)       i = 0..N−1

    hⱼ = squared horizontal distance to pillar j minus r_safe² (as in 7a);
    x_{i+1} is written as one RK4 step F(x_i, u_i) inside the constraint
    (ocp_config_obstacle.dcbf_pillars).

What γ does:
    The barrier value may shrink by at most the fraction γ per sample.
    Approaching a pillar head-on at speed v from distance d, h ≈ d² falls by
    about 2·v·Ts/d per step, so the DCBF row becomes active at

        d ≈ 2·v·Ts / γ          (v = 2 m/s, Ts = 0.05 s:  γ = 0.2 → 1 m,
                                 γ = 0.1 → 2 m,  γ = 0.05 → 4 m)

    — independent of the horizon. γ = 1 is the 7a constraint (active only
    when a predicted node reaches the circle); a small γ starts the
    avoidance earlier and spreads it out.

Runs:
    python simulate_obst_dcbf.py            main run, γ = GAMMA   → results/stage7b_obst_dcbf/
    python simulate_obst_dcbf.py --sweep    γ ∈ SWEEP: table + gamma_sweep.png
                                            (γ = 1 doubles as a check against 7a)
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')    # before the solver libraries load

import argparse
import numpy as np

import ocp_config_obstacle    as obs
import obstacle_sim_config    as osc
from obstacle_sim_config    import PILLARS, TS, MARGIN
from closed_loop_sim_config import OMEGA_MPC
from plot_utils             import plot_dcbf_sweep

RESULTS_DIR = os.path.join('results', 'stage7b_obst_dcbf')

GAMMA = 0.2                              # main run (trade-off: see --sweep)
SWEEP = [1.0, 0.5, 0.2, 0.1, 0.05]       # --sweep


# ─────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────
def swerve_lead(res: dict, dev: float = 0.01) -> float:
    """
    How long before the FIRST pillar crossing (P1, turn 1) the quadrotor
    leaves the reference by more than `dev` [s]. Before that crossing it
    tracks to < 1 mm (Stage 6), so this measures when the avoidance starts.
    """
    k_c  = osc.encounters(res['X_r'])[0][0]
    e_p  = np.linalg.norm(res['X'][:k_c, :3] - res['X_r'][:k_c, :3], axis=1)
    k_on = int(np.argmax(e_p > dev))
    return (k_c - k_on) * TS if e_p[k_on] > dev else 0.0


def sweep_row(res: dict) -> dict:
    X, X_r = res['X'], res['X_r']
    gap  = obs.clearance(X[:, :3], PILLARS).min(axis=1)
    e_p  = np.linalg.norm(X[:-1, :3] - X_r[:, :3], axis=1)
    w    = np.abs(X[:, 10:]).max(axis=1)
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (X[:, 7] ** 2 + X[:, 8] ** 2), -1, 1)))
    return dict(gamma=res['gamma'],
                fail=int(np.sum(~np.isin(res['status'], [0, 2]))),
                gap=1e3 * gap.min(), slack=res['slack'].max(),
                lead=swerve_lead(res),
                e_rms=1e3 * np.sqrt(np.mean(e_p ** 2)), e_max=1e3 * e_p.max(),
                w_max=w.max(), w_sat=np.mean(w > 0.97 * OMEGA_MPC),
                tilt=tilt.max(), t_mpc=1e3 * res['T_mpc'].mean())


def print_sweep(rows: list) -> None:
    print('\n' + '═' * 104)
    print('  Stage 7b — DCBF γ sweep (γ = 1 ≙ Stage 7a distance constraint)')
    print('═' * 104)
    print(f'  {"γ":>5} | {"fail":>4} | {"min gap":>8} | {"max slack":>9} | {"lead P1":>7} |'
          f' {"‖p−p_r‖ RMS/max":>16} | {"max|ω|":>6} | {"|ω|≈bound":>9} | {"tilt":>5} | {"MPC":>6}')
    print(f'  {"":>5} | {"":>4} | {"[mm]":>8} | {"[m²]":>9} | {"[s]":>7} |'
          f' {"[mm]":>16} | {"[rad/s]":>6} | {"[%]":>9} | {"[°]":>5} | {"[ms]":>6}')
    for r in rows:
        print(f'  {r["gamma"]:5.2f} | {r["fail"]:4d} | {r["gap"]:8.1f} | {r["slack"]:9.1e} |'
              f' {r["lead"]:7.2f} | {r["e_rms"]:7.0f} / {r["e_max"]:6.0f} | {r["w_max"]:6.2f} |'
              f' {100 * r["w_sat"]:9.1f} | {r["tilt"]:5.1f} | {r["t_mpc"]:6.2f}')


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Stage 7b DCBF obstacle avoidance')
    ap.add_argument('--sweep', action='store_true', help=f'run γ ∈ {SWEEP}')
    args = ap.parse_args()

    if args.sweep:
        runs = [osc.simulate(gamma=g, verbose=False) for g in SWEEP]
        print_sweep([sweep_row(r) for r in runs])
        curves = []
        for r in runs:
            t = np.arange(len(r['X_r'])) * TS
            curves.append((f'γ = {r["gamma"]:g}' + (' (≙ 7a)' if r['gamma'] == 1 else ''), t,
                           obs.clearance(r['X'][:-1, :3], PILLARS).min(axis=1),
                           np.linalg.norm(r['X'][:-1, :3] - r['X_r'][:, :3], axis=1),
                           np.abs(r['X'][:-1, 10:]).max(axis=1)))
        plot_dcbf_sweep(curves, (4.0, 11.0), MARGIN, OMEGA_MPC,
                        'Stage 7b — DCBF: effect of γ (first turn, P1 → P2 → P3)',
                        os.path.join(RESULTS_DIR, 'gamma_sweep.png'))
    else:
        res = osc.simulate(gamma=GAMMA)
        osc.summarize(res, f'Stage 7b — Obstacle avoidance, DCBF γ = {GAMMA:g} '
                           '(nominal, full state)')
        print_sweep([sweep_row(res)])
        osc.make_figures(res, RESULTS_DIR, f'Stage 7b — DCBF (γ = {GAMMA:g})')
