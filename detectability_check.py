"""
detectability_check.py — Stage 4b δ-IOSS Detectability Certificate (LMI)
========================================================================
Verifies exponential detectability (Assumption 1) of the exact system the
MHE optimizes over by constructing a quadratic i-iIOSS Lyapunov function

    U(x1, x2) = ‖x1 − x2‖²_P

through the LMI condition of Theorem 2 in

    J. D. Schiller, M. A. Müller,
    "Robust stability of moving horizon estimation for continuous-time
    systems", arXiv:2305.06614v2, 2024.

System — the augmented model shared with the EKF, taken from
quadrotor_3d_model (no local copy of the dynamics):
    ż = f(z, u, w) = [f(x, u, d); 0] + w       create_mhe_model()
    y = h(z)       = [p; q; ω]                 measurement_expr()

    z  (NZ = 19)  [px py pz, vx vy vz, qw qx qy qz, p q r, d_f(3), d_τ(3)]
    u  (NU = 4)   rotor thrusts — known input (acados model.p)
    w  (NW = 19)  process noise, additive on every derivative (model.u)
    y  (NY = 10)

    d enters f linearly with constant coefficients, so A = ∂f/∂z depends
    only on (q, ω, T_total) and the finite reduction below applies. The
    question the certificate answers: can d — never measured, no dynamics
    of its own — be reconstructed at rate κ? Only through d_f → v̇ → ṗ → y
    and d_τ → ω̇ → y.

LMI (31), with D = ∂h/∂w = 0:

    [ PA + AᵀP + κP − CᵀRC    PB ]
    [ BᵀP                     −Q ]  ⪯ 0          ∀ (x, u, w) ∈ X × U × W

    A = ∂f/∂z (q, ω, T_total),   B = ∂f/∂w = I₁₉,   C = ∂h/∂z (selection)

    ⇒ Assumption 1 with P1 = P2 = P and λ = e^{−κ}  (λ per SECOND).

Assumption 2 — why the sets look the way they do:
    f is C¹ ✓, h is affine (a selection) ✓, X and W must be CONVEX: the
    proof (App. A.3) integrates along the straight line between two states,
    so the LMI has to hold on every point of that line.

      p, v, d  unbounded            A does not depend on them   (asserted)
      q        ball ‖q‖ ≤ r_max     S³ itself is not convex; the ball is its
                                    convex hull (r_max > 1 covers small
                                    norm drift of the estimated quaternion)
      ω        box |ωᵢ| ≤ ω_max
      T_total  [T_min, T_max]       A depends on u only via Σfᵢ  (asserted)
      w        ℝ¹⁹                  A, B do not depend on w      (asserted)

Finite reduction (paper, footnote 4 — "solve at the vertices, convexity
does the rest"):
    A is affine in (q, ω) jointly and affine in T_total   (asserted
    symbolically), and the LMI is affine in A. Therefore the LMI on

        {attitude points on ‖q‖ = r_max, both hemispheres}
      × {8 vertices of the ω box} × {T_min, T_max}

    implies it on the convex hull of those points. Only q needs genuine
    gridding. The attitude set is refined by a cutting-plane loop:
        solve on an active set → check the whole attitude pool → add the
        worst violators → repeat until no violations.
    Residual gap: the hull of a finite pool is a polytope inscribed in the
    ball. A random off-pool spot check on the sphere guards it.

Downstream — what the MHE must match for Theorem 1 to apply
(equidistant sampling, T = N·Ts ⇒ Remark 5: δ̄ = 0):
    horizon   (17):  T > ln(4)/κ        (λ_max(P2, P1) = 1 since P1 = P2)
    cost      (9) :  prior  2‖χ̄ − x̂(t_i − T)‖²_P · λ^T
                     stage  (2‖w‖²_Q + ‖Δy‖²_R) · λ^{T−τ}
    prior         :  x̂(t_i − T) is the estimate produced N iterations ago
    state bounds  :  |ω| ≤ ω_max in the MHE — the estimated trajectory must
                     stay in X, constraint (8c)
    Choosing N and discretizing the cost integral on the acados grid is
    done in ocp_config_mhe (load_mhe_params, export_drone_mhe_solver); this
    script is purely CT and depends on the decay rate only through κ.

Design order (check_detectability) — purely continuous time, no Ts:
    1. choose the decay λ ∈ (0, 1)      (per second, Theorem 2: λ = e^{−κ})
           κ = −ln λ   [1/s]
    2. solve LMI (31) at this κ              →  P, Q, R
    3. minimum horizon, (17) with δ̄ = 0      →  T_min = ln4/κ
    4. horizon T = c·T_min, c = 1.5 — rounded up to whole samples in
       load_mhe_params (N = ⌈c·T_min/Ts⌉), where the acados grid is known.
       Guaranteed rate (18): ρ = 4^{1/T}·λ ≈ λ^{1−1/c}. The MHE horizon is
       independent of the MPC horizon — it fixes the CT window length.
    If LMI (31) is infeasible at this κ: raise λ toward 1 (lower κ, longer
    T_min) or tighten the envelope.

Output:
    data/mhe_params.npz with the CT quantities
    (P, Q, R, kappa, lam, T_hor_min, envelope, margins).

Ported from the MATLAB IOSS verification (YALMIP + MOSEK) to
CVXPY + MOSEK / CLARABEL / SCS.
"""

import numpy as np
import casadi as ca
import cvxpy as cp

from quadrotor_3d_model import (
    create_mhe_model, measurement_expr,
    f_hover, F_MAX, OMEGA_MAX, NU, NX,
)
from ocp_config_mhe import MHE_PARAMS_PATH, HORIZON_FACTOR, save_mhe_params

STATE_NAMES  = ['px', 'py', 'pz', 'vx', 'vy', 'vz',
                'qw', 'qx', 'qy', 'qz', 'p', 'q', 'r',
                'dfx', 'dfy', 'dfz', 'dtx', 'dty', 'dtz']
OUTPUT_NAMES = ['px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'p', 'q', 'r']

IDX_P            = slice(0, 3)
IDX_Q, IDX_OMEGA = slice(6, 10), slice(10, 13)

N_VERT = 16                     # 8 ω-box vertices × 2 thrust endpoints


# ─────────────────────────────────────────────────────────────────
# Linearization of the MHE model + structural checks
# ─────────────────────────────────────────────────────────────────
def build_jacobians(n_probe=20, seed=0):
    """
    Differentiate the MHE model symbolically and assert every structural
    property the finite reduction relies on.

    Returns
    -------
    jac_A_fun : ca.Function  (x(nx), u(4)) → A (nx×nx)
    B         : ndarray (nx×nx)  = I
    C         : ndarray (10×nx)  constant selection y = [p; q; ω]
    """
    model = create_mhe_model()
    x, w, u = model.x, model.u, model.p
    nx = x.rows()
    f = model.f_expl_expr
    h = measurement_expr(x)

    A_expr = ca.jacobian(f, x)
    B_expr = ca.jacobian(f, w)
    jac_A_fun = ca.Function('A_mhe', [x, u], [A_expr], ['x', 'u'], ['A'])

    # ── Grid validity (exact, symbolic) ──
    #    everything except (q, ω) — p, v and d — must drop out
    x_ungridded = ca.vertcat(x[0:6], x[NX:])
    assert not ca.depends_on(A_expr, x_ungridded), \
        'A depends on p, v or d — they would have to be gridded'
    assert not ca.depends_on(A_expr, w), \
        'A depends on w — W would have to be gridded'
    assert not ca.depends_on(B_expr, ca.vertcat(x, u, w)), 'B is not constant'

    # Vertex argument: A affine in z = (q, ω) jointly and affine in u
    z = x[6:13]
    assert not ca.depends_on(ca.jacobian(ca.vec(A_expr), z), z), \
        'A is not affine in (q, ω) — vertex reduction invalid (normalization term in f?)'
    assert not ca.depends_on(ca.jacobian(ca.vec(A_expr), u), u), \
        'A is not affine in the thrusts — T endpoints do not suffice'

    # ── Constant matrices ──
    B = ca.evalf(B_expr).full()
    C = ca.evalf(ca.jacobian(h, x)).full()
    D = ca.evalf(ca.jacobian(h, w)).full()
    assert np.allclose(D, 0.0), 'LMI below assumes D = 0'
    assert np.allclose(B, np.eye(nx)), 'B must be I for the additive-w model'
    ny = C.shape[0]
    assert C.shape[1] == nx and np.allclose(C @ C.T, np.eye(ny)), \
        'C must be a selection matrix (h affine, Assumption 2)'

    # ── A depends on u only through T_total = Σfᵢ ──
    rng = np.random.default_rng(seed)
    for _ in range(n_probe):
        xp = np.zeros(nx)
        qp = rng.normal(size=4)
        xp[IDX_Q]     = qp / np.linalg.norm(qp)
        xp[IDX_OMEGA] = rng.uniform(-3.0, 3.0, 3)
        split = rng.dirichlet(np.ones(NU)) * NU * f_hover
        A_split = jac_A_fun(xp, split).full()
        A_equal = jac_A_fun(xp, np.full(NU, split.sum() / NU)).full()
        assert np.allclose(A_split, A_equal, atol=1e-10), \
            'A depends on individual thrusts — grid over all 4, not T_total'

    return jac_A_fun, B, C


def eval_A_batch(jac_A_fun, quats, omegas, thrusts):
    """
    A at M points (q, ω, T_total), vectorized with CasADi map.
    quats (M, 4), omegas (M, 3), thrusts (M,)  →  A (M, nx, nx)
    Position, velocity (and d) do not enter A; each rotor gets T_total / 4.
    """
    M  = len(quats)
    nx = jac_A_fun.size1_in(0)
    X  = np.zeros((nx, M))
    X[IDX_Q]     = quats.T
    X[IDX_OMEGA] = omegas.T
    U = np.tile(np.asarray(thrusts) / NU, (NU, 1))
    out = jac_A_fun.map(M)(X, U).full()                 # nx × nx·M, hcat of blocks
    return out.reshape(nx, M, nx).transpose(1, 0, 2)


# ─────────────────────────────────────────────────────────────────
# Grids
# ─────────────────────────────────────────────────────────────────
def quat_grid_hypersp(n1, n2, n3, mode='Line'):
    """
    Unit quaternions on the upper hemisphere of S³ (qw ≥ 0), hyperspherical
    coordinates: t1 = half rotation angle, (t2, t3) = rotation axis on S².
    Returns (n1*n2*n3, 4), scalar-first, each row ‖q‖ = 1.
    """
    if mode == 'Line':
        t1 = np.linspace(0.0, np.pi / 2, n1)
        t2 = np.linspace(0.0, np.pi,     n2)
        t3 = np.linspace(0.0, 2 * np.pi, n3, endpoint=False)
    else:
        rng = np.random.default_rng()
        t1 = rng.uniform(0.0, np.pi / 2, n1)
        t2 = rng.uniform(0.0, np.pi,     n2)
        t3 = rng.uniform(0.0, 2 * np.pi, n3)
    T1, T2, T3 = np.meshgrid(t1, t2, t3, indexing='ij')
    T1, T2, T3 = T1.ravel(), T2.ravel(), T3.ravel()
    return np.column_stack([
        np.cos(T1),
        np.sin(T1) * np.cos(T2),
        np.sin(T1) * np.sin(T2) * np.cos(T3),
        np.sin(T1) * np.sin(T2) * np.sin(T3),
    ])


def attitude_pool(n1, n2, n3, r_max):
    """
    Attitude points whose convex hull approximates the ball ‖q‖ ≤ r_max.
    Both hemispheres (q and −q give different A, and the hull must contain
    the interior of the ball), duplicates removed (the hyperspherical grid
    collapses at t1 = 0 and t2 ∈ {0, π}).
    """
    Qs = quat_grid_hypersp(n1, n2, n3, 'Line')
    Qs = np.unique(np.round(Qs, 12), axis=0)
    return r_max * np.vstack([Qs, -Qs])


def vertex_points(quats, omega_max, T_range):
    """
    Expand each attitude to its 16 (ω, T) vertex combinations.
    Returns (quats, omegas, thrusts) with 16·len(quats) rows,
    grouped per attitude (rows 16k … 16k+15 belong to quats[k]).
    """
    verts = np.array(np.meshgrid(*[[-omega_max, omega_max]] * 3,
                                 indexing='ij')).reshape(3, -1).T      # (8, 3)
    n_q = len(quats)
    Qp = np.repeat(quats, N_VERT, axis=0)
    Wp = np.tile(np.repeat(verts, 2, axis=0), (n_q, 1))
    Tp = np.tile(np.asarray(T_range, dtype=float), 8 * n_q)
    return Qp, Wp, Tp


# ─────────────────────────────────────────────────────────────────
# LMI (31): numerical evaluation
# ─────────────────────────────────────────────────────────────────
def lmi_max_eig(A, P, Q, R, kappa, B, C):
    """Largest eigenvalue of the LMI (31) block for a batch A (M, nx, nx)."""
    M       = A.shape[0]
    nx, nw  = B.shape
    TL = P @ A + A.transpose(0, 2, 1) @ P + kappa * P - C.T @ R @ C
    TR = np.broadcast_to(P @ B, (M, nx, nw))
    BR = np.broadcast_to(-Q,    (M, nw, nw))
    L  = np.concatenate([np.concatenate([TL, TR], axis=2),
                         np.concatenate([TR.transpose(0, 2, 1), BR], axis=2)],
                        axis=1)
    L  = 0.5 * (L + L.transpose(0, 2, 1))
    return np.linalg.eigvalsh(L)[:, -1]


def worst_eig_per_quat(jac_A_fun, B, C, P, Q, R, kappa,
                       quats, omega_max, T_range, chunk=1000):
    """Worst LMI eigenvalue for each attitude over its 16 (ω, T) vertices."""
    worst = np.empty(len(quats))
    for s in range(0, len(quats), chunk):
        qs = quats[s:s + chunk]
        A  = eval_A_batch(jac_A_fun, *vertex_points(qs, omega_max, T_range))
        worst[s:s + len(qs)] = lmi_max_eig(A, P, Q, R, kappa, B, C) \
                                   .reshape(len(qs), N_VERT).max(axis=1)
    return worst


# ─────────────────────────────────────────────────────────────────
# LMI (31): SDP on an active attitude set
# ─────────────────────────────────────────────────────────────────
def solve_ioss_lmi(jac_A_fun, B, C, quats, omega_max, T_range, kappa,
                   q_floor, r_floor,
                   p_min       = 1e-2,
                   eps         = 1e-5,
                   solver_pref = ('MOSEK', 'CLARABEL', 'SCS')):
    """
    Impose LMI (31) at every (attitude × ω-vertex × T-endpoint) point and
    minimize trace(Q) + trace(R).

    Scaling: (31) is homogeneous in (P, Q, R); P ⪰ p_min·I fixes the scale
    and eps is the strict margin relative to that normalization.
    q_floor / r_floor: per-channel lower bounds on diag(Q), diag(R). The LMI
    only yields LOWER bounds on Q, R (raising them keeps feasibility), so
    floors cost no feasibility — use them to keep the exact kinematic
    channels (w_p, w_q) expensive in the MHE.

    Returns (status, solver_used, P, Q, R) — P, Q, R are None on failure.
    """
    nx, nw = B.shape
    ny     = C.shape[0]
    P      = cp.Variable((nx, nx), symmetric=True)
    q_diag = cp.Variable(nw)
    r_diag = cp.Variable(ny)
    Q_sdp  = cp.diag(q_diag)
    R_sdp  = cp.diag(r_diag)

    constraints = [P >> p_min * np.eye(nx),
                   q_diag >= q_floor,
                   r_diag >= r_floor]

    A_all = eval_A_batch(jac_A_fun, *vertex_points(quats, omega_max, T_range))
    CRC   = C.T @ R_sdp @ C
    PB    = P @ B
    I_blk = np.eye(nx + nw)
    for A in A_all:
        TL  = P @ A + A.T @ P + kappa * P - CRC
        LMI = cp.bmat([[TL, PB], [PB.T, -Q_sdp]])
        # symmetric by construction, but CVXPY cannot always detect it
        constraints.append(0.5 * (LMI + LMI.T) << -eps * I_blk)

    prob = cp.Problem(cp.Minimize(cp.trace(Q_sdp) + cp.trace(R_sdp)),
                      constraints)

    status    = 'no solver installed'
    installed = cp.installed_solvers()
    for solver in solver_pref:
        if solver not in installed:
            continue
        try:
            prob.solve(solver=solver, verbose=False)
        except cp.SolverError as e:
            print(f'      {solver} failed: {e}')
            continue
        status = prob.status
        if status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            if status == cp.OPTIMAL_INACCURATE:
                print(f'      {solver}: optimal_inaccurate — '
                      f'the independent eigenvalue check decides')
            return (status, solver, P.value,
                    np.diag(q_diag.value), np.diag(r_diag.value))
        print(f'      {solver}: {status}')

    return status, None, None, None, None


# ─────────────────────────────────────────────────────────────────
# Cutting-plane certification over the attitude pool
# ─────────────────────────────────────────────────────────────────
def certify(jac_A_fun, B, C, kappa, pool, omega_max, T_range,
            q_floor, r_floor,
            n_init       = 8,
            add_per_iter = 8,
            max_iter     = 15,
            seed         = 42,
            **solve_kw):
    """
    Solve on an active attitude set, check the WHOLE pool at the (ω, T)
    vertices with an independent eigenvalue computation, add the worst
    violators, repeat. The certificate is the eigenvalue check, not the
    solver status.

    Returns a dict with P, Q, R and diagnostics, or None.
    """
    rng    = np.random.default_rng(seed)
    active = [int(i) for i in rng.choice(len(pool), n_init, replace=False)]

    for it in range(1, max_iter + 1):
        status, solver, P, Q, R = solve_ioss_lmi(
            jac_A_fun, B, C, pool[active], omega_max, T_range, kappa,
            q_floor, r_floor, **solve_kw)
        if P is None:
            print(f'    iter {it:2d}: SDP failed ({status}) with '
                  f'{len(active)} active attitudes')
            return None

        worst = worst_eig_per_quat(jac_A_fun, B, C, P, Q, R, kappa,
                                   pool, omega_max, T_range)
        viol  = np.flatnonzero(worst > 0.0)
        print(f'    iter {it:2d}: {len(active):4d} active attitudes '
              f'({N_VERT * len(active)} LMIs), solver {solver}, '
              f'violators {viol.size:5d}/{len(pool)}, '
              f'max λ = {worst.max():+.3e}')

        if viol.size == 0:
            return dict(P=P, Q=Q, R=R, solver=solver, status=status,
                        n_active=len(active), iterations=it,
                        pool_max_eig=float(worst.max()),
                        active_quats=pool[active])

        in_active = set(active)
        ranked    = viol[np.argsort(worst[viol])[::-1]]
        new       = [int(i) for i in ranked if i not in in_active][:add_per_iter]
        if not new:
            print('    violations only at already-active attitudes → solver '
                  'accuracy, not grid density. Increase eps or use MOSEK/CLARABEL.')
            return None
        active += new

    print(f'    no certificate after {max_iter} iterations')
    return None


def spot_check(jac_A_fun, B, C, P, Q, R, kappa, r_max, omega_max, T_range,
               n=20000, seed=7, chunk=5000):
    """
    Off-pool random samples: attitude uniform on the sphere ‖q‖ = r_max,
    ω and T uniform in the INTERIOR of their ranges.
    - interior ω, T are implied by affinity → sanity check of that argument
    - off-pool attitudes probe the gap between the pool's hull and the ball
    Returns the worst eigenvalue (must be ≤ 0).
    """
    rng   = np.random.default_rng(seed)
    worst = -np.inf
    for s in range(0, n, chunk):
        m  = min(chunk, n - s)
        qs = rng.normal(size=(m, 4))
        qs = r_max * qs / np.linalg.norm(qs, axis=1, keepdims=True)
        om = rng.uniform(-omega_max, omega_max, (m, 3))
        Tv = rng.uniform(T_range[0], T_range[1], m)
        A  = eval_A_batch(jac_A_fun, qs, om, Tv)
        worst = max(worst, lmi_max_eig(A, P, Q, R, kappa, B, C).max())
    return float(worst)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def check_detectability():
    # ── Configuration ──────────────────────────────────────────
    #    Design order: λ → κ → LMI (P, Q, R) → T_min. The MHE horizon is NOT
    #    chosen here: load_mhe_params picks N = ⌈c·T_min/Ts⌉ afterwards.
    lam       = 0.5                     # [-]     decay per second (the design input)
    assert 0.0 < lam < 1.0, 'need 0 < λ < 1'
    kappa     = -np.log(lam)            # [1/s]   decay rate in LMI (31), λ = e^{−κ}
    omega_max = OMEGA_MAX               # [rad/s] per axis — MHE box, MPC keeps 0.9·ω_max
    T_range   = (0.0, NU * F_MAX)       # [N]     T_total, matches MPC bound 0 ≤ fᵢ ≤ F_MAX
    r_max     = 1.02                    # [-]     quaternion ball radius (norm drift margin)

    # ── Step 1: Jacobians and structural checks ──
    print('═' * 72)
    print('  Stage 4b — i-iIOSS Detectability Certificate for Continuous-time Systems')
    print('  Schiller & Müller (2024), Theorem 2 / LMI (31)')
    print('  model:  ż = [f(x, u, d); 0] + w   (create_mhe_model, z = [x; d])')
    print('          y = [p; q; ω]             (measurement_expr)')
    print('═' * 72)

    print('\n[1] Linearizing the MHE model ...')
    jac_A_fun, B, C = build_jacobians()
    nx, nw = B.shape
    ny     = C.shape[0]
    print(f'    A: {nx}×{nx}, affine in (q, ω) and in T_total, '
          f'independent of p, v, d, w  ✓')
    print(f'    B = I{nx}  ✓    C: {C.shape} selection  ✓    D = 0  ✓')

    # per-channel floors: exact kinematics (ṗ = v, q̇ = ½ q⊗ω) stay expensive
    q_floor = np.full(nw, 1e-4)
    q_floor[IDX_P] = 1.0
    q_floor[IDX_Q] = 1.0
    r_floor = np.full(ny, 1e-4)

    # Minimum MHE horizon, (17) with δ̄ = 0 (Remark 5) and P1 = P2:  T > ln4/κ
    T_hor_min = np.log(4.0) / kappa     # [s]

    # ── Step 2: Sets X × U × W and decay rate ──
    pool = attitude_pool(20, 20, 20, r_max)
    print('\n[2] Sets (Assumption 2, convex):')
    print(f'    q      : ‖q‖ ≤ {r_max}   ({len(pool)} pool attitudes, both hemispheres)')
    print(f'    ω      : |ωᵢ| ≤ {omega_max} rad/s   (8 vertices)')
    print(f'    T_total: [{T_range[0]:.2f}, {T_range[1]:.2f}] N   (2 endpoints)')
    print(f'    p, v, w: unbounded   (A independent)')
    print(f'    decay:  λ = {lam} per s   ⇒   κ = −ln λ = {kappa:.4f} 1/s')
    print(f'            T_min = ln4/κ = {T_hor_min:.3f} s   (MHE horizon must exceed it)')

    # ── Step 3: Cutting-plane certification ──
    print('\n[3] Solving LMI (31) — cutting plane over the attitude pool ...')
    cert = certify(jac_A_fun, B, C, kappa, pool, omega_max, T_range,
                   q_floor, r_floor)

    print('\n' + '═' * 72)
    if cert is None:
        print('  ✗ NO CERTIFICATE — raise λ toward 1 (lower κ, longer T_min), '
              'tighten the envelope, or try another solver.')
        print('═' * 72)
        return

    P_val, Q_val, R_val = cert['P'], cert['Q'], cert['R']

    # ── Step 4: Off-pool spot check ──
    print('\n[4] Off-pool spot check (random attitudes on the sphere, '
          'interior ω and T) ...')
    spot_max = spot_check(jac_A_fun, B, C, P_val, Q_val, R_val, kappa,
                          r_max, omega_max, T_range)
    print(f'    max λ = {spot_max:+.3e}')
    if spot_max > 0.0:
        print('  ✗ Spot check FAILED — the pool is too coarse. '
              'Refine attitude_pool() and re-run.')
        print('═' * 72)
        return

    # ── Step 5: Report and save ──
    P_eigs = np.linalg.eigvalsh(P_val)
    print('\n  ✓ i-iIOSS LYAPUNOV FUNCTION FOUND  (Assumption 1 holds on X × U × W)')
    print(f'    solver {cert["solver"]} ({cert["status"]}), '
          f'{cert["iterations"]} iterations, {cert["n_active"]} active attitudes')
    print(f'    max λ(LMI): pool {cert["pool_max_eig"]:+.3e},  spot {spot_max:+.3e}')
    print(f'\n    P: eig ∈ [{P_eigs.min():.3e}, {P_eigs.max():.3e}],  '
          f'cond = {P_eigs.max() / P_eigs.min():.1f}')

    print('\n    diag(Q) — process-noise supply rates:')
    for i, nm in enumerate(STATE_NAMES[:nw]):
        print(f'      w_{nm:5s}  {Q_val[i, i]:.4e}')
    print('\n    diag(R) — output supply rates:')
    for i, nm in enumerate(OUTPUT_NAMES):
        print(f'      {nm:5s}  {R_val[i, i]:.4e}')

    c     = HORIZON_FACTOR
    T_c   = c * T_hor_min
    print('\n    MHE requirements (Theorem 1, Remark 5):')
    print(f'      horizon   T > T_min = ln4/κ = {T_hor_min:.3f} s;  T ≈ {c}·T_min = {T_c:.3f} s '
          f'(N = ⌈T/Ts⌉ in load_mhe_params)')
    print(f'      rate      ρ = 4^(1/T)·λ ≈ λ^(1−1/c) = {4.0 ** (1.0 / T_c) * lam:.3f} per s  '
          f'(Theorem 1, (18))')
    print('      prior     2·‖χ̄ − x̂(t_i − T)‖²_P · λ^T      (x̂ from time t_i − T)')
    print('      stage     (2·‖w‖²_Q + ‖Δy‖²_R) · λ^(T−τ)   (discretized in ocp_config_mhe)')
    print(f'      bounds    |ω| ≤ {omega_max} rad/s on the estimated trajectory')

    save_mhe_params(
        MHE_PARAMS_PATH,
        P=P_val, Q=Q_val, R=R_val,
        kappa=kappa, lam=lam, T_hor_min=T_hor_min,
        omega_max=omega_max, T_range=np.asarray(T_range), r_max=r_max,
        pool_max_eig=cert['pool_max_eig'], spot_max_eig=spot_max,
        active_quats=cert['active_quats'],
    )
    print(f'\n    Saved to {MHE_PARAMS_PATH}')
    print('═' * 72)


if __name__ == '__main__':
    check_detectability()
