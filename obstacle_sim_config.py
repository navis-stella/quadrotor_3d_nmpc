"""
obstacle_sim_config.py — Stage 7 Shared Scene and Closed Loop (Distance / DCBF)
===============================================================================
The quadrotor flies the Stage 6 helix (R = 2 m, 5 m climb over 3 turns,
2 m/s) through a field of vertical pillars that stand ON the reference path.
The reference is left as it is — the controller has to leave it to pass a
pillar and rejoin it afterwards. This module is the ONE scene and closed
loop both obstacle constraints run in; the entry scripts only pick the
constraint:

    simulate_obst_dist.py   Stage 7a  distance constraint  hⱼ(p_i) ≥ 0
    simulate_obst_dcbf.py   Stage 7b  DCBF  hⱼ(x_{i+1}) ≥ (1 − γ)·hⱼ(x_i), γ sweep

    plant        nominal (d = 0)
    feedback     full state, x̂ = x, d̂ = 0
    reference    Stage 6 helix + flatness (x_r, u_r), unchanged
    controller   Stage 6 tracking NMPC + soft pillar constraints
                 (ocp_config_obstacle: gamma=None → distance, gamma=γ → DCBF)

    Full state feedback on purpose: the constraint acts on the state the
    MPC is given. With an estimator in the loop it would hold for x̂, not
    for x, and a near miss could not be told apart from estimation error —
    that needs a margin from the estimation error bound (robust MPC, later).

Scene:
    Three pillars of radius 0.2 m, each centred 0.15 m off the helix circle
    (P1 and P3 outside, P2 inside). Pillars are infinite in z, so the
    helix crosses each one once per turn — 9 encounters in 3 turns.

    Why off-centre: a pillar exactly on the circle makes the warm start
    (the reference) run through its axis, where ∇h = 0, and makes passing
    inside and outside equally good. With the offset the shorter way round
    is well defined: P1/P3 are passed on the inside of the helix, P2 on
    the outside — the one place the tracking cost and the constraint pull
    to the same side.

Loop (Ts = 0.05 s, MPC horizon 1 s, N = 20 — as Stage 6):

    (X_r, U_r)_k  = flatness( helix(t_k + i·Ts) ),  i = 0..N
    u_k           = MPC(x_k, X_r, U_r)       (soft pillar rows, 7a or 7b)
    x_{k+1}       = plant(x_k, u_k)

Imports:
    reference_generate.py     → HelixReference
    reference_flatness.py     → flat_to_state_input
    ocp_config_obstacle.py    → Pillar, create_solver, projected shift_warm_start, clearance
    ocp_config_trajectory.py  → runtime helpers (init_guess, set_trajectory, ...)
    closed_loop_sim_config.py → OMEGA_MPC  (same soft |ω| box as Stage 6)

Shared by both stages: the scene (PILLARS), the closed loop
simulate(gamma=), encounters, summarize and make_figures (six figures).

Results:  results/stage7a_obst_dist/, results/stage7b_obst_dcbf/
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')    # before the solver libraries load

import time
import numpy as np

from quadrotor_3d_model import (create_disturbance_plant, quat_normalize,
                                OMEGA_MAX, F_MAX, NX, NU, ND)
from reference_generate     import HelixReference
from reference_flatness     import flat_to_state_input
import ocp_config_trajectory as traj
import ocp_config_obstacle   as obs
from ocp_config_obstacle    import Pillar, R_DRONE, MARGIN
from closed_loop_sim_config import OMEGA_MPC
from plot_utils             import (plot_states, plot_inputs, plot_3d_trajectory,
                                    plot_tracking_error, plot_obstacle_topview,
                                    plot_obstacle_clearance)

# ─────────────────────────────────────────────────────────────────
# Scenario
# ─────────────────────────────────────────────────────────────────
N         = 20
T_HORIZON = 1.0
TS        = T_HORIZON / N            # 0.05 s
REF       = HelixReference()         # Stage 6 default helix
T_SIM     = REF.duration             # ≈ 22.9 s
N_SIM     = int(np.ceil(T_SIM / TS))

PILLAR_RADIUS = 0.2                  # [m] physical radius ρ
# (phase angle θ on the helix [rad], radial offset from the circle [m]);
# θ ∈ [2, 2π) is crossed three times during the cruise (θ runs 2 → 6π).
PILLAR_SITES  = [(2.6, +0.15),       # P1  outside the circle → passed inside
                 (4.4, -0.15),       # P2  inside the circle  → passed outside
                 (6.0, +0.15)]       # P3  outside the circle → passed inside


def pillars_on_helix(ref: HelixReference, sites, radius: float) -> list:
    """Pillars at helix phase θ, centred (R + offset) from the helix axis."""
    cx, cy, R = *ref.center, ref.params.R
    return [Pillar(cx + (R + off) * np.cos(th), cy + (R + off) * np.sin(th), radius)
            for th, off in sites]


PILLARS = pillars_on_helix(REF, PILLAR_SITES, PILLAR_RADIUS)
NP_OBS  = len(PILLARS)


# ─────────────────────────────────────────────────────────────────
# Closed Loop
# ─────────────────────────────────────────────────────────────────
def simulate(gamma: float = None, verbose: bool = True) -> dict:
    """
    gamma=None: distance constraint (7a);  gamma=γ: DCBF (7b).

    Returns dict with X (n+1, 13), X_r (n, 13), U (n, 4), U_r (n, 4),
    status (n,), T_mpc (n,) [s], slack (n,) largest pillar slack per solve,
    P_pred (n, N+1, 3) predicted positions of every solve, gamma.

    T_mpc covers flatness map + reference setting + solve. A failed solve
    (status ∉ {0, 2}) is counted; its first input is still applied, clipped
    to the motor limits (u_r if not finite).
    """
    x0 = flat_to_state_input(REF.sample(0.0)).x_r    # at rest at the origin
    kind = 'distance constraint' if gamma is None else f'DCBF, γ = {gamma:g}'
    print(f'Compiling acados NMPC solver (trajectory tracking + pillars, {kind})...')
    solver = obs.create_solver(PILLARS, N, T_HORIZON, omega_max=OMEGA_MPC, gamma=gamma)
    plant  = create_disturbance_plant(TS)

    ref0 = flat_to_state_input(REF.horizon(0.0, N, TS))
    traj.init_guess(solver, ref0.x_r, ref0.u_r, N)

    X      = np.zeros((N_SIM + 1, NX)); X[0] = x0
    X_r    = np.zeros((N_SIM, NX))
    U_r    = np.zeros((N_SIM, NU))
    U      = np.zeros((N_SIM, NU))
    status = np.zeros(N_SIM, dtype=int)
    T_mpc  = np.zeros(N_SIM)
    slack  = np.zeros(N_SIM)
    P_pred = np.zeros((N_SIM, N + 1, 3))
    d0     = np.zeros(ND)

    for k in range(N_SIM):
        t = k * TS

        # 1. Reference over the horizon + NMPC (full state feedback)
        t0  = time.perf_counter()
        ref = flat_to_state_input(REF.horizon(t, N, TS))
        if k > 0:
            obs.shift_warm_start(solver, ref.x_r, ref.u_r, N, PILLARS)
        traj.set_trajectory(solver, ref.x_r, ref.u_r, d0, N)
        traj.set_initial_state(solver, X[k])
        status[k] = solver.solve()
        T_mpc[k]  = time.perf_counter() - t0

        u = solver.get(0, 'u')
        if not np.all(np.isfinite(u)):
            u = ref.u_r[0]
        U[k] = np.clip(u, 0.0, F_MAX)
        X_r[k], U_r[k] = ref.x_r[0], ref.u_r[0]
        slack[k]  = obs.obstacle_slack(solver, N, NP_OBS, dcbf=gamma is not None)
        P_pred[k] = obs.predicted_positions(solver, N)

        # 2. Nominal plant
        plant.set('x', X[k])
        plant.set('u', U[k])
        plant.set('p', d0)
        plant.solve()
        X[k + 1] = quat_normalize(plant.get('x'))

        if verbose and k % int(round(2.0 / TS)) == 0:
            gap = obs.clearance(X[k, :3], PILLARS).min()
            print(f'  t={t:5.2f}s  |  ‖p − p_r‖ = '
                  f'{np.linalg.norm(X[k, :3] - X_r[k, :3]):.3f} m  |  '
                  f'min gap {gap:6.3f} m  |  status {status[k]}')

    return dict(X=X, X_r=X_r, U=U, U_r=U_r, status=status, T_mpc=T_mpc,
                slack=slack, P_pred=P_pred, gamma=gamma)


# ─────────────────────────────────────────────────────────────────
# Encounters
# ─────────────────────────────────────────────────────────────────
def encounters(X_r: np.ndarray) -> list:
    """
    [(k, j), ...] — the sample where the REFERENCE comes closest to pillar j,
    once per crossing (each maximal run of samples where the reference is
    inside the constraint circle, i.e. where it cannot be followed).
    """
    gap_r = obs.clearance(X_r[:, :3], PILLARS)
    out = []
    for j in range(NP_OBS):
        inside = gap_r[:, j] < MARGIN
        edges  = np.flatnonzero(np.diff(np.r_[0, inside.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            out.append((a + int(np.argmin(gap_r[a:b, j])), j))
    return sorted(out)


def first_sightings(res: dict) -> list:
    """
    [(t, P_pred), ...] — for each pillar, the first solve whose prediction
    comes as close to it as any prediction of the run does (to 1 mm): the
    pillar has just entered the horizon and the MPC starts to swerve. For
    the distance constraint (7a) that closest approach is the constraint
    circle; the DCBF (7b) keeps its predictions farther out. Shown in the
    top view.
    """
    out = []
    for j in range(NP_OBS):
        g = obs.clearance(res['P_pred'].reshape(-1, 3), [PILLARS[j]])
        g = g.reshape(N_SIM, N + 1).min(axis=1)
        k = int(np.argmax(g <= g.min() + 1e-3))
        out.append((k * TS, res['P_pred'][k]))
    return out


# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────
def summarize(res: dict, title: str) -> None:
    X, X_r, U = res['X'], res['X_r'], res['U']
    gap   = obs.clearance(X[:, :3], PILLARS)
    gap_b = obs.clearance_between_samples(X[:, :3], PILLARS)
    e_p   = np.linalg.norm(X[:-1, :3] - X_r[:, :3], axis=1)
    fail  = ~np.isin(res['status'], [0, 2])
    sat   = np.mean(np.any((U <= 1e-6) | (U >= F_MAX - 1e-6), axis=1))
    T_mpc = res['T_mpc'] * 1e3
    v_max = np.linalg.norm(X[:, 3:6], axis=1).max()
    r_min = min(o.r_safe() for o in PILLARS)

    print('\n' + '═' * 70)
    print(f'  {title}')
    print('═' * 70)
    print(f'    pillars                     : {NP_OBS} × ρ = {PILLAR_RADIUS} m,'
          f' r_drone = {R_DRONE} m, δ = {MARGIN} m')
    print(f'    reference crossings         : {len(encounters(X_r))}')
    print(f'    failed OCP calls            : {fail.sum()}/{len(fail)}')
    print(f'    min gap at samples          : {1e3 * gap.min():7.1f} mm   (δ = {1e3 * MARGIN:.0f} mm)')
    print(f'    min gap between samples     : {1e3 * gap_b.min():7.1f} mm   '
          f'(chord bound (v·Ts)²/(8 r_safe) = {1e3 * (v_max * TS) ** 2 / (8 * r_min):.1f} mm)')
    for j in range(NP_OBS):
        print(f'      P{j + 1} at ({PILLARS[j].cx:+.2f}, {PILLARS[j].cy:+.2f})       :'
              f' {1e3 * gap[:, j].min():7.1f} mm')
    print(f'    max pillar slack            : {res["slack"].max():.2e} m²')
    print(f'    ‖p − p_r‖  RMS / max        : {1e3 * np.sqrt(np.mean(e_p ** 2)):7.1f} /'
          f' {1e3 * e_p.max():7.1f} mm   (deliberate: the reference crosses pillars)')
    print(f'    max |ω|                     : {np.abs(X[:, 10:]).max():.3f} rad/s'
          f'   (soft bound {OMEGA_MPC:.2f}, envelope {OMEGA_MAX})')
    print(f'    thrust range                : [{U.min():.3f}, {U.max():.3f}] N'
          f'   (limits [0, {F_MAX:.3f}]), saturated {sat:.1%} of samples')
    print(f'    MPC time mean / p99 / max   : {T_mpc.mean():.2f} / {np.percentile(T_mpc, 99):.2f}'
          f' / {T_mpc.max():.2f} ms   (flatness + solve, budget {1e3 * TS:.0f} ms)')


# ─────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────
def make_figures(res: dict, out: str, title: str) -> None:
    """The six figures of one run (shared by Stages 7a and 7b)."""
    cyl   = np.array([[o.cx, o.cy, o.radius] for o in PILLARS])
    marks = [(k * TS, f'P{j + 1}') for k, j in encounters(res['X_r'])]

    plot_obstacle_topview(res['X'], res['X_r'], cyl, R_DRONE, MARGIN,
                          f'{title}: top view', os.path.join(out, 'topview.png'),
                          P_pred=first_sightings(res))
    plot_obstacle_clearance(obs.clearance(res['X'][:, :3], PILLARS),
                            obs.clearance_between_samples(res['X'][:, :3], PILLARS),
                            res['slack'], TS, MARGIN,
                            f'{title}: gap to the pillars', os.path.join(out, 'clearance.png'))
    plot_tracking_error(res['X'], res['X_r'], TS,
                        f'{title}: tracking error (dotted: reference crosses a pillar)',
                        os.path.join(out, 'tracking_error.png'), t_marks=marks)
    plot_3d_trajectory(res['X'], res['X_r'],
                       f'{title}: 3D flight path', os.path.join(out, 'trajectory_3d.png'),
                       cylinders=cyl)
    plot_states(res['X'], TS, res['X_r'],
                f'{title}: states vs reference x_r', os.path.join(out, 'states.png'))
    plot_inputs(res['U'], TS, f'{title}: motor thrusts (dashed: u_r)',
                os.path.join(out, 'inputs.png'), U_ref=res['U_r'])
