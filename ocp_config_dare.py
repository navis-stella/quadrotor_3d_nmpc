"""
ocp_config_dare.py — Stage 2b DARE Terminal Cost NMPC Solver
============================================================
Creates and returns the acados OCP solver.

Stage 1:   W_e = Q           (simple terminal cost)
Stage 2a:  W_e = P_lyap      (QIH — full terminal cost + terminal set constraint)
Stage 2b:  W_e = P_lqr       (DARE-based terminal cost, no terminal constraint)  ← THIS FILE

Design rationale — why Stage 2b relaxes QIH:
    Stage 2a demonstrated that the QIH terminal set Omega_alpha is
    extremely small (alpha ≈ 0.0001) due to the quadrotor's strong
    nonlinearity. The terminal constraint was violated for ~47% of
    simulation steps — the entire transient where MPC does real work.
    By the time x_N enters Omega_alpha, the drone is already near hover.

    Dropping the terminal set constraint and using P_DARE as terminal
    cost retains the essential property (infinite tail is bounded by
    a quadratic) without the infeasibility risk from a tight constraint.
    This is the standard practical choice for drone NMPC.

DARE with quaternion (compute_dare_terminal_cost):
    The full 13-state system has qw uncontrollable in the linearization
    (row/col 6 of A_c are zero). DARE requires a stabilizable pair (A_d, B_d).
    Solution: remove qw → solve DARE on 12-state reduced system → embed
    P_red back into 13×13 with a small qw weight on the diagonal.

    The implementation lives in ocp_config_offsetfree.py — Stages 3–4 reuse
    this exact terminal cost, so there is one copy and the stages cannot
    drift apart.

Quaternion cost design:
    For small angles: phi ≈ 2*qx,  theta ≈ 2*qy,  psi ≈ 2*qz
    So quaternion weights ≈ 4× Euler angle weights for equivalent behavior.
    qw is only lightly penalized — it stays near 1 naturally via unit norm.
"""

import numpy as np
import casadi as ca
from scipy.linalg import block_diag
from acados_template import AcadosOcp, AcadosOcpSolver

from quadrotor_3d_model    import create_nominal_model, f_hover, F_MAX, NX, NU
from ocp_config_offsetfree import compute_dare_terminal_cost


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

    No terminal set constraint — this is the relaxed Stage 2b design.
    See ocp_config_qih.py (Stage 2a) for the full QIH version.

    Constraints:
        0 <= fi <= F_MAX   for each motor  (box constraints on input)

    Solver:  SQP_RTI + ERK4

    Returns:
        AcadosOcpSolver ready to be called in the simulation loop
    """
    model = create_nominal_model()
    ocp   = AcadosOcp()
    ocp.model = model

    Ts    = T_horizon / N                                  # sample time
    u_ref = np.full(NU, f_hover)                           # hover input

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

    # ── Terminal Cost: DARE-based P_lqr (Stage 2b) ─────────────
    P_lqr = compute_dare_terminal_cost(Q, R, Ts)

    ocp.cost.W   = block_diag(Q, R)      # stage weight    (17×17)
    ocp.cost.W_e = P_lqr                 # terminal weight (13×13) ← DARE

    # references
    ocp.cost.yref   = np.concatenate([x_ref, u_ref])   # stage    (17,)
    ocp.cost.yref_e = x_ref                            # terminal (13,)

    # ── Constraints ─────────────────────────────────────────────
    ocp.constraints.x0 = np.zeros(NX)                  # updated at runtime

    ocp.constraints.lbu   = np.zeros(NU)
    ocp.constraints.ubu   = np.full(NU, F_MAX)             # shared envelope
    ocp.constraints.idxbu = np.arange(NU)

    # ── Solver Options ───────────────────────────────────────────
    ocp.solver_options.N_horizon       = N
    ocp.solver_options.tf              = T_horizon
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.qp_solver       = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx  = 'GAUSS_NEWTON'
    ocp.solver_options.print_level     = 0

    return AcadosOcpSolver(ocp)
