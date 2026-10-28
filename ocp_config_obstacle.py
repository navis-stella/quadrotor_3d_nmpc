"""
ocp_config_obstacle.py — Stage 7 Obstacle-Avoidance NMPC
========================================================
The Stage 6 trajectory-tracking NMPC (ocp_config_trajectory.create_ocp,
unchanged cost and dynamics) plus one state constraint per obstacle. The
obstacles are infinite vertical cylinders (pillars) — the reference is NOT
changed: it may run straight through a pillar, the constraint pushes the
predicted path around it and the tracking cost pulls it back afterwards.

Two forms of the obstacle constraint, chosen by create_solver(gamma=):

    7a  distance constraint     hⱼ(p_i) ≥ 0                    i = 1..N
    7b  discrete-time CBF       hⱼ(p_{i+1}) ≥ (1 − γ)·hⱼ(p_i)  i = 0..N−1
        (gamma=None → 7a; see "Stage 7b" below)

Stage 7a:

    min_u  Σ_{i<N} ‖ [e(x_i, x_r,i); u_i − u_r,i] ‖²_W  +  ‖ e(x_N, x_r,N) ‖²_P
           + Σ_{i=1..N} Σ_j  z₁·sⱼ,ᵢ + z₂·sⱼ,ᵢ²
    s.t.   x_{i+1} = f(x_i, u_i, d̂),   x_0 = x̂(t),   0 ≤ u_i ≤ F_MAX
           hⱼ(p_i) ≥ −sⱼ,ᵢ,  sⱼ,ᵢ ≥ 0      i = 1..N  (terminal node included)

Pillar constraint (squared distance in the horizontal plane):

    hⱼ(p) = (pₓ − cₓ,ⱼ)² + (p_y − c_y,ⱼ)² − r_safe,ⱼ²,
    r_safe,ⱼ = ρⱼ + r_drone + δ

    ρⱼ       physical pillar radius
    r_drone  the quadrotor as a ball (arm + propeller), so p is a point
    δ        margin: covers the path between two nodes (a chord of length
             ℓ = v·Ts with both ends on the circle enters it by ≈ ℓ²/(8 r_safe))
             plus anything the nominal model does not see

    Why the SQUARED distance: ‖p − c‖ is not differentiable at the axis;
    the square is smooth everywhere. Its gradient vanishes only ON the axis,
    so a warm start must never run exactly through a pillar centre (the
    scene offsets every pillar from the helix circle for that reason).

    Why it suits SQP / RTI: hⱼ is CONVEX in p, so its linearization at the
    current iterate p̄ underestimates it,
        hⱼ(p) ≥ hⱼ(p̄) + ∇hⱼ(p̄)ᵀ(p − p̄),
    i.e. the linearized constraint is a half-plane tangent to (a circle
    around) the pillar that lies inside the free space. A QP iterate that
    satisfies it is collision-free at the nodes — the separating-hyperplane
    view of X. Zhang, A. Liniger, F. Borrelli, "Optimization-based collision
    avoidance," IEEE TCST 29(3), 2021.

    A pillar has no top: only (pₓ, p_y) enter hⱼ, the constraint does not
    depend on z — the quadrotor must go around, never over.

Soft constraint (exact penalty):
    Linear + quadratic slack penalty, like the Stage 4 |ω| box. As long as
    z₁ exceeds the constraint multiplier, the soft problem has the same
    solution as the hard one whenever the hard one is feasible (E. Kerrigan,
    J. Maciejowski, "Soft constraints and exact penalty functions in model
    predictive control," UKACC Control 2000) — the slack only becomes
    non-zero when no collision-free solution exists, instead of the QP
    failing. Check: the reported slack stays ≈ 0 in the run.

    z₁ = 1e4, ten times the |ω| box weight: acados multiplies the slack
    penalties of the path nodes by cost_scaling (= Ts = 0.05 s, like the
    stage cost), so z₁ = 1e3 is only 50 per node — below the multiplier
    of a pillar that has just entered the horizon. The solver then traded
    up to 30 mm of the margin δ at the far nodes for tracking; with 1e4
    (500 per node) the slack stays at solver tolerance (~1e-11).

Nonconvex free space:
    The free space ℝ² \\ ∪ disks is not convex. The solver finds A local
    optimum (pass the pillar on the inside or the outside of the helix); the
    warm start chooses which. Stability does not need the global optimum
    (P. Scokaert, D. Mayne, J. Rawlings, "Suboptimal model predictive
    control (feasibility implies stability)," IEEE TAC 44(3), 1999), but
    nothing here certifies the Stage 6 terminal cost either (see there):
    the guarantee is practical — the 1 s horizon must see a pillar early
    enough to swerve.

Stage 7b — discrete-time control barrier function (DCBF):
    The distance constraint only acts once a pillar is inside the horizon;
    then the quadrotor swerves as hard as the rate bound allows. The DCBF
    constraint limits how fast the barrier value may DECREASE, one step at
    a time:

        hⱼ(x_{i+1}) ≥ (1 − γ)·hⱼ(x_i),    0 < γ ≤ 1

    By induction hⱼ(x_i) ≥ (1 − γ)^i·hⱼ(x_0) ≥ 0, so the predicted path is
    collision-free AND the safe set {hⱼ ≥ 0} is forward invariant under
    any input that satisfies the constraint at i = 0 — the applied input
    u_0 is filtered by it. γ = 1 is the 7a constraint; a smaller γ lets h
    shrink by at most a fraction γ per Ts, so the approach must start
    earlier and be gentler. J. Zeng, B. Zhang, K. Sreenath, "Safety-critical
    model predictive control with discrete-time control barrier function,"
    ACC 2021.

    How to write it in acados: a constraint sees ONE node (x_i, u_i), the
    DCBF couples two. x_{i+1} is therefore written as F(x_i, u_i, p_i),
    one RK4 step of length Ts — the same discretization the OCP uses (ERK,
    4 stages, 1 step), so h(F(x_i, u_i)) IS h(x_{i+1}) on a converged
    (defect-free) trajectory:

        cⱼ(x_i, u_i) = hⱼ(F(x_i, u_i)) − (1 − γ)·hⱼ(x_i) ≥ 0,   i = 0..N−1

    Node 0 now carries the constraint: x_0 is fixed, but u_0 is free, and
    the row on u_0 is the actual safety filter. No terminal row is needed —
    hⱼ(x_N) ≥ (1 − γ)·hⱼ(x_{N−1}) is the i = N−1 row.

    What is lost: cⱼ is a DIFFERENCE of convex functions of p, composed
    with the dynamics, so its linearization is no longer an inner
    approximation of the safe set — a QP iterate may violate it, only a
    converged solution satisfies it. A small γ can also make the OCP
    infeasible (the decay rate cannot be met in time); the soft slack —
    the relaxation of Zeng et al. — keeps the QP solvable.

    z₁ = 1e5 (Z_L1_DCBF): with the 7a weight 1e4 the DCBF rows at the far
    end of the horizon took slack (up to 0.3 m² at γ = 0.05) — the DCBF
    binds over many nodes against the pull of the tracking cost, so its
    multipliers are larger. With 1e5 the slack stays at solver tolerance
    (~1e-12) for γ ≥ 0.1 (7e-4 m² at γ = 0.05); the node-0 row held in
    the closed loop in every case.

acados mapping:
    con_h_expr / con_h_expr_e   7a: h(p) for all pillars (nh = n_pillars);
                                stage 0 has none (x_0 is fixed — a violated
                                h there could only make the QP infeasible)
    con_h_expr_0 / con_h_expr   7b: cⱼ(x, u) on stages 0..N−1, no terminal row
    lh = 0,  uh = ACADOS_INFTY  one-sided (HPIPM masks the infinite bound)
    idxsh, zl/Zl                slacks appended AFTER the |ω| slacks
                                (acados orders them bu, bx, g, h); stage 0
                                has no |ω| slack, so zl_0 = pillar weights
"""

from dataclasses import dataclass

import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, ACADOS_INFTY

import ocp_config_trajectory as traj

# ─────────────────────────────────────────────────────────────────
# Obstacle Design
# ─────────────────────────────────────────────────────────────────
R_DRONE = 0.25      # [m]  quadrotor as a ball: arm L = 0.175 m + propeller
MARGIN  = 0.05      # [m]  δ: inter-sample chord (≈ 2.5 mm at 2 m/s) + model slack
Z_L1      = 1e4     # linear slack weight, 7a  (exact penalty, see module docstring)
Z_L1_DCBF = 1e5     # linear slack weight, 7b  (larger multipliers, see there)
Z_L2    = 1e2       # quadratic slack weight


@dataclass(frozen=True)
class Pillar:
    """Infinite vertical cylinder: axis through (cx, cy), radius [m]."""
    cx:     float
    cy:     float
    radius: float

    def r_safe(self, r_drone: float = R_DRONE, margin: float = MARGIN) -> float:
        """Radius the point p must stay outside of: ρ + r_drone + δ."""
        return self.radius + r_drone + margin


# ─────────────────────────────────────────────────────────────────
# Constraint Function
# ─────────────────────────────────────────────────────────────────
def h_pillars(p, pillars, r_drone: float = R_DRONE, margin: float = MARGIN):
    """
    hⱼ(p) = ‖p_xy − cⱼ‖² − r_safe,ⱼ²  for every pillar (CasADi column).
    hⱼ ≥ 0  ⇔  the drone ball keeps at least δ from pillar j.
    """
    return ca.vertcat(*[(p[0] - o.cx) ** 2 + (p[1] - o.cy) ** 2
                        - o.r_safe(r_drone, margin) ** 2 for o in pillars])


def rk4_step(model, Ts: float):
    """
    F(x, u, p) = one RK4 step of model.f_expl_expr over Ts — the integrator
    of the OCP (ERK, sim_method_num_stages = 4, one step per interval).
    """
    f = ca.Function('f', [model.x, model.u, model.p], [model.f_expl_expr])
    x, u, p = model.x, model.u, model.p
    k1 = f(x,               u, p)
    k2 = f(x + Ts / 2 * k1, u, p)
    k3 = f(x + Ts / 2 * k2, u, p)
    k4 = f(x + Ts * k3,     u, p)
    return x + Ts / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def dcbf_pillars(model, Ts: float, gamma: float, pillars,
                 r_drone: float = R_DRONE, margin: float = MARGIN):
    """
    cⱼ(x, u) = hⱼ(F(x, u)) − (1 − γ)·hⱼ(x)  for every pillar (CasADi column);
    cⱼ ≥ 0 is the DCBF condition hⱼ(x_{i+1}) ≥ (1 − γ)·hⱼ(x_i).
    """
    x_next = rk4_step(model, Ts)
    return (h_pillars(x_next[0:3], pillars, r_drone, margin)
            - (1.0 - gamma) * h_pillars(model.x[0:3], pillars, r_drone, margin))


def clearance(P: np.ndarray, pillars, r_drone: float = R_DRONE) -> np.ndarray:
    """
    Metric gap between the drone ball and each pillar surface [m]:

        cⱼ(p) = ‖p_xy − cⱼ‖ − ρⱼ − r_drone      (n, n_pillars)

    cⱼ < 0 is a collision; the constraint asks for cⱼ ≥ MARGIN.
    """
    P = np.atleast_2d(P)
    return np.stack([np.hypot(P[:, 0] - o.cx, P[:, 1] - o.cy) - o.radius - r_drone
                     for o in pillars], axis=1)


def clearance_between_samples(P: np.ndarray, pillars,
                              r_drone: float = R_DRONE) -> np.ndarray:
    """
    Smallest gap on the straight segment p_k → p_{k+1} (n−1, n_pillars):
    the constraint only acts at the nodes, so this shows whether the path
    BETWEEN two samples cuts into a pillar (the ℓ²/(8 r) effect).
    """
    a, b = P[:-1, :2], P[1:, :2]
    ab   = b - a
    out  = []
    for o in pillars:
        c  = np.array([o.cx, o.cy])
        s  = np.clip(np.sum((c - a) * ab, axis=1) / np.maximum(np.sum(ab * ab, axis=1), 1e-12),
                     0.0, 1.0)
        q  = a + s[:, None] * ab                              # closest point on the segment
        out.append(np.linalg.norm(q - c, axis=1) - o.radius - r_drone)
    return np.stack(out, axis=1)


# ─────────────────────────────────────────────────────────────────
# Constraint Builder
# ─────────────────────────────────────────────────────────────────
def _append(cost, field: str, values: np.ndarray) -> None:
    """cost.<field> ← [existing slack weights; values]  (keeps the |ω| slacks first)."""
    old = getattr(cost, field)
    old = np.zeros(0) if old is None else np.atleast_1d(old)
    setattr(cost, field, np.concatenate([old, values]))


def add_pillar_constraints(ocp:      AcadosOcp,
                           pillars,
                           r_drone:  float = R_DRONE,
                           margin:   float = MARGIN,
                           z_l1:     float = Z_L1,
                           z_l2:     float = Z_L2,
                           gamma:    float = None) -> None:
    """
    gamma=None (7a): soft hⱼ(p) ≥ 0 on nodes 1..N (path + terminal).
    gamma=γ    (7b): soft DCBF cⱼ(x, u) ≥ 0 on nodes 0..N−1 (no terminal row).
    One row per pillar. Must be called after every other soft constraint of
    the OCP is set: the slacks are appended to the existing weight vectors.
    """
    n = len(pillars)
    lh, uh = np.zeros(n), np.full(n, ACADOS_INFTY)
    w1, w2 = np.full(n, z_l1), np.full(n, z_l2)

    if gamma is not None:
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f'DCBF needs 0 < gamma <= 1, got {gamma}')
        Ts = ocp.solver_options.tf / ocp.solver_options.N_horizon
        c  = dcbf_pillars(ocp.model, Ts, gamma, pillars, r_drone, margin)

        ocp.model.con_h_expr_0  = c                  # u_0: the safety filter
        ocp.constraints.lh_0    = lh
        ocp.constraints.uh_0    = uh
        ocp.constraints.idxsh_0 = np.arange(n)
        for f, w in (('zl_0', w1), ('zu_0', w1), ('Zl_0', w2), ('Zu_0', w2)):
            _append(ocp.cost, f, w)                  # stage 0 has no |ω| slack

        ocp.model.con_h_expr  = c
        ocp.constraints.lh    = lh
        ocp.constraints.uh    = uh
        ocp.constraints.idxsh = np.arange(n)
        for f, w in (('zl', w1), ('zu', w1), ('Zl', w2), ('Zu', w2)):
            _append(ocp.cost, f, w)
        return

    h = h_pillars(ocp.model.x[0:3], pillars, r_drone, margin)

    ocp.model.con_h_expr  = h
    ocp.constraints.lh    = lh
    ocp.constraints.uh    = uh
    ocp.constraints.idxsh = np.arange(n)
    for f, w in (('zl', w1), ('zu', w1), ('Zl', w2), ('Zu', w2)):
        _append(ocp.cost, f, w)

    ocp.model.con_h_expr_e  = h
    ocp.constraints.lh_e    = lh
    ocp.constraints.uh_e    = uh
    ocp.constraints.idxsh_e = np.arange(n)
    for f, w in (('zl_e', w1), ('zu_e', w1), ('Zl_e', w2), ('Zu_e', w2)):
        _append(ocp.cost, f, w)


# ─────────────────────────────────────────────────────────────────
# Solver Creator
# ─────────────────────────────────────────────────────────────────
def create_solver(pillars,
                  N:         int   = 20,
                  T_horizon: float = 1.0,
                  omega_max: float = None,
                  r_drone:   float = R_DRONE,
                  margin:    float = MARGIN,
                  z_l1:      float = None,
                  z_l2:      float = Z_L2,
                  gamma:     float = None) -> AcadosOcpSolver:
    """
    Stage 6 trajectory-tracking OCP + soft pillar constraints:
    gamma=None → distance constraint (7a), gamma=γ → DCBF (7b).
    z_l1=None → Z_L1 (7a) or Z_L1_DCBF (7b).

    The pillars are static, so they are compiled into the solver as
    constants (no extra parameters: p stays [d̂; x_r; u_r], NP = 23). The
    runtime helpers of ocp_config_trajectory (set_trajectory,
    set_initial_state, init_guess, shift_warm_start) apply unchanged.
    """
    ocp = traj.create_ocp(N, T_horizon, omega_max)
    # one name per γ: each solver is its own shared library, so several γ
    # can be built and run in one process (the sweep of simulate_obst_dcbf)
    name = 'obst_dist' if gamma is None else f'obst_dcbf_g{gamma:g}'.replace('.', 'p')
    ocp.model.name = f'quadrotor_3d_{name}'
    if z_l1 is None:
        z_l1 = Z_L1 if gamma is None else Z_L1_DCBF
    add_pillar_constraints(ocp, pillars, r_drone, margin, z_l1, z_l2, gamma)
    ocp.code_export_directory = f'c_generated_code_{name}'
    return AcadosOcpSolver(ocp, json_file=f'acados_ocp_{name}.json',
                           verbose=False)


# ─────────────────────────────────────────────────────────────────
# Runtime Helpers
# ─────────────────────────────────────────────────────────────────
def shift_warm_start(solver:  AcadosOcpSolver,
                     X_r:     np.ndarray,
                     U_r:     np.ndarray,
                     N:       int,
                     pillars,
                     r_drone: float = R_DRONE,
                     margin:  float = MARGIN) -> None:
    """
    Stage 6 shifted warm start, then every guess node whose position lies
    inside a constraint circle is pushed radially onto it.

    Why: the shift appends the reference x_r,N as the new tail node — and the
    reference runs THROUGH the pillars. Linearized at a point p̄ inside the
    circle, the constraint becomes the half-plane 2(p̄ − c)ᵀ(p − c) ≥
    ‖p̄ − c‖² + r_safe², which lies FARTHER out than the circle itself (at
    distance (‖p̄ − c‖² + r_safe²) / (2‖p̄ − c‖) ≥ r_safe). The one RTI step
    cannot get there and pays with slack. Linearized ON the circle the
    half-plane is its tangent — exact at the guess, inside the free space
    everywhere else. The radial direction keeps the side the reference is
    already on, i.e. the shorter way round.
    """
    traj.shift_warm_start(solver, X_r, U_r, N)
    for i in range(1, N + 1):
        x = solver.get(i, 'x')
        moved = False
        for o in pillars:
            dxy = x[0:2] - np.array([o.cx, o.cy])
            r, r_s = np.linalg.norm(dxy), o.r_safe(r_drone, margin)
            if r < r_s:
                n = dxy / r if r > 1e-9 else np.array([1.0, 0.0])
                x[0:2] = np.array([o.cx, o.cy]) + r_s * n
                moved = True
        if moved:
            solver.set(i, 'x', x)


def obstacle_slack(solver: AcadosOcpSolver, N: int, n_pillars: int,
                   dcbf: bool = False) -> float:
    """
    Largest pillar slack of the last solve [m²] (≈ 0 ⇔ every pillar row
    holds). 'sl' holds the lower slacks in the order [|ω| box; pillars], so
    the pillar part is the last n_pillars. Rows: nodes 1..N (7a) or
    0..N−1 (7b, dcbf=True — the terminal node has no pillar row there).
    """
    nodes = range(0, N) if dcbf else range(1, N + 1)
    return max(float(np.max(solver.get(i, 'sl')[-n_pillars:])) for i in nodes)


def predicted_positions(solver: AcadosOcpSolver, N: int) -> np.ndarray:
    """Predicted p_0..p_N of the last solve (N+1, 3)."""
    return np.array([solver.get(i, 'x')[0:3] for i in range(N + 1)])
