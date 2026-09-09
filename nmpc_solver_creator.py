"""
nmpc_solver_creator.py — acados NMPC Solver for 3D Quadrotor (Quaternion)
==========================================================================
Creates and returns the acados OCP solver.

Stage 1:    W_e = Q           (simple terminal cost)
Stage 2.1:  W_e = P_lyap      (QIH — full terminal cost + terminal set constraint)
Stage 2.2:  W_e = P_lqr       (DARE-based terminal cost, no terminal constraint)  ← CURRENT

Design rationale — why Stage 2.2 relaxes QIH:
    Stage 2.1 demonstrated that the QIH terminal set Omega_alpha is
    extremely small (alpha ≈ 0.0001) due to the quadrotor's strong
    nonlinearity. The terminal constraint was violated for ~47% of
    simulation steps — the entire transient where MPC does real work.
    By the time x_N enters Omega_alpha, the drone is already near hover.

    Dropping the terminal set constraint and using P_DARE as terminal
    cost retains the essential property (infinite tail is bounded by
    a quadratic) without the infeasibility risk from a tight constraint.
    This is the standard practical choice for drone NMPC.

DARE with quaternion:
    The full 13-state system has qw uncontrollable in the linearization
    (row/col 6 of A_c are zero). DARE requires a stabilizable pair (A_d, B_d).
    Solution: remove qw → solve DARE on 12-state reduced system → embed
    P_red back into 13×13 with a small qw weight on the diagonal.

Quaternion cost design:
    For small angles: phi ≈ 2*qx,  theta ≈ 2*qy,  psi ≈ 2*qz
    So quaternion weights ≈ 4× Euler angle weights for equivalent behavior.
    qw is only lightly penalized — it stays near 1 naturally via unit norm.

Extending in future stages:
    Stage 3 → augmented model + offset-free formulation
    Stage 4 → obstacle avoidance state constraints
"""

import numpy as np
import casadi as ca
from scipy.linalg import block_diag, solve_discrete_are
from scipy.signal import cont2discrete
from acados_template import AcadosOcp, AcadosOcpSolver

from quadrotor_3d_model import (create_model, get_hover_linearization,
                                f_hover, NX, NU)


# ─────────────────────────────────────────────────────────────────
# DARE Terminal Cost — Reduced-Order for Quaternion (Stage 2.2)
# ─────────────────────────────────────────────────────────────────
def compute_dare_terminal_cost(Q: np.ndarray,
                               R: np.ndarray,
                               Ts: float) -> np.ndarray:
    """
    Compute the LQR terminal cost matrix P_lqr via reduced-order DARE.

    The quaternion state has 13 components but only 12 DOF (||q||=1).
    At hover, qw is uncontrollable in the linearized system:
        - Row 6 of A_c is all zeros  (dqw depends on nothing at hover)
        - Column 6 of A_c is all zeros  (nothing depends on qw at hover)
    DARE requires stabilizability → we must remove qw first.

    Procedure:
        1. Linearize at hover                    → A_c (13×13), B_c (13×4)
        2. Remove qw (row/col 6)                 → A_c_red (12×12), B_c_red (12×4)
        3. Remove qw weight from Q               → Q_red (12×12)
        4. Discretize with ZOH                    → A_d_red, B_d_red
        5. Solve DARE                             → P_red (12×12)
        6. Embed back into 13×13                  → P_lqr with small qw weight

    Reduced state (12):
        x_red = [px, py, pz, vx, vy, vz, qx, qy, qz, p, q, r]
        (qw removed — index 6 in full state)

    Args:
        Q:   (13×13) full state weight matrix
        R:   (4×4)   input weight matrix
        Ts:  sample time [s]

    Returns:
        P_lqr:  (13×13) terminal cost matrix for acados W_e
    """
    # ── Step 1: continuous-time linearization at hover ──────────
    A_c, B_c = get_hover_linearization()

    # ── Step 2: remove qw (index 6) from A_c, B_c ─────────────
    #
    # Full state:    [px,py,pz, vx,vy,vz, qw, qx,qy,qz, p,q,r]
    #   indices:       0  1  2   3  4  5   6   7  8  9  10 11 12
    #
    # Reduced state: [px,py,pz, vx,vy,vz, qx,qy,qz, p,q,r]
    #   indices:       0  1  2   3  4  5   6  7  8   9 10 11
    #
    idx_keep = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]  # skip index 6 (qw)
    A_c_red = A_c[np.ix_(idx_keep, idx_keep)]   # (12×12)
    B_c_red = B_c[idx_keep, :]                   # (12×4)

    # ── Step 3: remove qw weight from Q ────────────────────────
    Q_red = Q[np.ix_(idx_keep, idx_keep)]         # (12×12)

    # ── Step 4: discretize using Zero-Order Hold ───────────────
    n_red = len(idx_keep)   # 12
    sys_d = cont2discrete(
        (A_c_red, B_c_red, np.eye(n_red), np.zeros((n_red, NU))),
        Ts, method='zoh'
    )
    A_d_red = sys_d[0]
    B_d_red = sys_d[1]

    # ── Step 5: solve DARE ─────────────────────────────────────
    P_red = solve_discrete_are(A_d_red, B_d_red, Q_red, R)

    # verify positive definiteness
    eigvals_red = np.linalg.eigvalsh(P_red)
    assert np.all(eigvals_red > 0), f'P_red not positive definite! eigvals={eigvals_red}'

    # ── Step 6: embed P_red (12×12) back into P_lqr (13×13) ───
    #
    # Place P_red entries at the correct positions in the full
    # 13×13 matrix. For qw (index 6), use a small diagonal weight
    # — qw is not controlled by DARE but needs a non-zero cost
    # to keep the Gauss-Newton Hessian well-conditioned.
    #
    P_lqr = np.zeros((NX, NX))
    P_lqr[np.ix_(idx_keep, idx_keep)] = P_red    # 12×12 block
    P_lqr[6, 6] = Q[6, 6]                        # qw weight = original Q_qw

    # ── Diagnostics ────────────────────────────────────────────
    eigvals_full = np.linalg.eigvalsh(P_lqr)
    print('─── DARE Terminal Cost (Stage 2.2, Reduced-Order) ───')
    print(f'  Sample time Ts     = {Ts:.4f} s')
    print(f'  Reduced system     = {n_red} states (qw removed)')
    print(f'  P_red eigenvalues  = {np.sort(eigvals_red)[::-1][:6]} ...')
    print(f'  P_lqr diagonal    = {np.diag(P_lqr)}')
    print(f'  Q diagonal         = {np.diag(Q)}')
    print(f'  Ratio P/Q diag     = {np.diag(P_lqr) / np.diag(Q)}')
    print(f'  P_lqr[6,6] (qw)   = {P_lqr[6,6]} (from Q, not DARE)')
    print('─────────────────────────────────────────────────────')

    return P_lqr


# ─────────────────────────────────────────────────────────────────
# Solver Creator
# ─────────────────────────────────────────────────────────────────
def create_solver(x_ref:     np.ndarray,
                  N:         int   = 20,
                  T_horizon: float = 1.0) -> AcadosOcpSolver:
    """
    Build and compile the acados NMPC solver for 3D quadrotor (quaternion).

    Args:
        x_ref:      reference state (13,)
                     [px,py,pz, vx,vy,vz, qw,qx,qy,qz, p,q,r]
        N:          prediction horizon (number of steps)
        T_horizon:  prediction horizon duration [s]

    Cost structure (NONLINEAR_LS):
        Stage:    || [x; u] - [x_ref; u_ref] ||^2_W         (17-dim)
        Terminal: || x_N - x_ref ||^2_{P_lqr}                (13-dim, DARE-based)

    No terminal set constraint — this is the relaxed Stage 2.2 design.
    See nmpc_solver_creator_qih.py (Stage 2.1) for the full QIH version.

    Constraints:
        0 <= fi <= f_max   for each motor  (box constraints on input)

    Solver:  SQP_RTI + ERK4

    Returns:
        AcadosOcpSolver ready to be called in the simulation loop
    """
    model = create_model()
    ocp   = AcadosOcp()
    ocp.model = model

    Ts    = T_horizon / N                                  # sample time
    u_ref = np.array([f_hover, f_hover, f_hover, f_hover]) # hover input

    # ── Cost ────────────────────────────────────────────────────
    ocp.cost.cost_type   = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    # cost expressions: what we penalize
    ocp.model.cost_y_expr   = ca.vertcat(model.x, model.u)  # stage:    [x; u] (17-dim)
    ocp.model.cost_y_expr_e = model.x                       # terminal: [x]    (13-dim)

    # ── Weight matrices ─────────────────────────────────────────
    Q = np.diag([
        80.,  80.,  120.,           # px, py, pz          position: high
        10.,  10.,   15.,           # vx, vy, vz          velocity: moderate
        10., 120.,  120.,  80.,     # qw, qx, qy, qz     attitude: scaled from Euler
         1.,   1.,    1.            # p,  q,  r           angular rate: low
    ])

    R = np.diag([0.1, 0.1, 0.1, 0.1])   # f1, f2, f3, f4

    # ── Terminal Cost: DARE-based P_lqr (Stage 2.2) ────────────
    P_lqr = compute_dare_terminal_cost(Q, R, Ts)

    ocp.cost.W   = block_diag(Q, R)      # stage weight    (17×17)
    ocp.cost.W_e = P_lqr                 # terminal weight (13×13) ← DARE

    # references
    ocp.cost.yref   = np.concatenate([x_ref, u_ref])   # stage    (17,)
    ocp.cost.yref_e = x_ref                            # terminal (13,)

    # ── Constraints ─────────────────────────────────────────────
    ocp.constraints.x0 = np.zeros(NX)                  # updated at runtime

    f_max = 3.0 * f_hover                              # max thrust per motor
    ocp.constraints.lbu   = np.array([0.0, 0.0, 0.0, 0.0])
    ocp.constraints.ubu   = np.array([f_max, f_max, f_max, f_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # ── Solver Options ───────────────────────────────────────────
    ocp.solver_options.N_horizon       = N
    ocp.solver_options.tf              = T_horizon
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.qp_solver       = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx  = 'GAUSS_NEWTON'
    ocp.solver_options.print_level     = 0

    return AcadosOcpSolver(ocp)
