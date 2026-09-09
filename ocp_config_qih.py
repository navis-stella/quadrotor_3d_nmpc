"""
ocp_config_qih.py — acados QIH-NMPC Solver for 3D Quadrotor
=============================================================
Stage 2.1:  Complete Quasi-Infinite Horizon NMPC

Implements QIH-NMPC with:
    - Terminal cost W_e = P_lyap   (from modified Lyapunov equation)
    - Terminal constraint:  (x_N - x_ref)^T P (x_N - x_ref) <= alpha
      implemented as a SOFT constraint to avoid hard infeasibility.

The terminal constraint enforces x_N ∈ Omega_alpha, the positively
invariant region where the auxiliary LQR can stabilize the system
without violating input bounds.  Softening it (via L1 + L2 slack
penalties) preserves the stability guarantee when feasible, while
gracefully degrading rather than returning no solution when a
disturbance pushes the system far from equilibrium.

Observation from simulation results:
    alpha ≈ 0.0001, while V_N peaks at ~2470 (ratio 20 million×).
    The terminal set is violated for ~47% of steps — the entire
    transient phase where MPC does all the real work.
    → Stage 2.2 drops the terminal constraint, keeping only P_DARE.

Cost structure (NONLINEAR_LS):
    Stage:     || [x; u] - [x_ref; u_ref] ||^2_W           (17-dim)
    Terminal:  || x_N - x_ref ||^2_{P_lyap}                 (13-dim)
    + soft penalty on terminal set violation

Constraints:
    Input:     0 <= f_i <= f_max           (hard, box on u)
    Terminal:  (x_N-x_ref)^T P (x_N-x_ref) <= alpha   (soft)

Extends:  Stage 1 (W_e = Q)
See also: Stage 2.2 (ocp_config_dare.py) — relaxed, P_DARE only
"""

import numpy as np
import casadi as ca
from scipy.linalg import block_diag
from acados_template import AcadosOcp, AcadosOcpSolver

from quadrotor_3d_model import create_model, f_hover, NX, NU
from compute_qih_params import compute_qih_offline_parameters


def create_solver(x_ref:     np.ndarray,
                  N:         int   = 20,
                  T_horizon: float = 1.0) -> AcadosOcpSolver:
    """
    Build and compile the acados QIH-NMPC solver.

    Args:
        x_ref:      reference state (13,)
        N:          prediction horizon steps
        T_horizon:  prediction horizon duration [s]

    Returns:
        AcadosOcpSolver ready for the simulation loop
    """
    model = create_model()
    ocp   = AcadosOcp()
    ocp.model = model

    Ts    = T_horizon / N
    u_ref = np.array([f_hover, f_hover, f_hover, f_hover])
    f_max = 3.0 * f_hover

    # ── QIH Offline Parameters ──────────────────────────────────
    Q = np.diag([
        80.,  80.,  120.,           # px, py, pz
        10.,  10.,   15.,           # vx, vy, vz
        10., 120.,  120.,  80.,     # qw, qx, qy, qz
         1.,   1.,    1.            # p,  q,  r
    ])
    R = np.diag([0.1, 0.1, 0.1, 0.1])

    P_qih, alpha_qih, kappa, K_red = \
        compute_qih_offline_parameters(Q, R, f_max)

    # ── Cost: NONLINEAR_LS ──────────────────────────────────────
    ocp.cost.cost_type   = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    ocp.model.cost_y_expr   = ca.vertcat(model.x, model.u)   # (17,)
    ocp.model.cost_y_expr_e = model.x                        # (13,)

    ocp.cost.W   = block_diag(Q, R)      # (17×17) stage
    ocp.cost.W_e = P_qih                 # (13×13) terminal — Lyapunov P

    ocp.cost.yref   = np.concatenate([x_ref, u_ref])   # (17,)
    ocp.cost.yref_e = x_ref                            # (13,)

    # ── Input Constraints (hard) ────────────────────────────────
    ocp.constraints.x0    = np.zeros(NX)               # set at runtime
    ocp.constraints.lbu   = np.array([0.0, 0.0, 0.0, 0.0])
    ocp.constraints.ubu   = np.array([f_max, f_max, f_max, f_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # ── Terminal Constraint: x_N^T P x_N <= alpha (soft) ────────
    #
    #   Nonlinear terminal constraint h_e(x_N):
    #       h_e = (x_N - x_ref)^T  P_qih  (x_N - x_ref)
    #
    #   Bounds:  0 <= h_e <= alpha
    #
    #   Note: P_qih is baked in as a numeric matrix at compile time.
    #   If x_ref changes at runtime, this constraint expression must
    #   be rebuilt (or use acados parameters).
    #
    x_dev = model.x - x_ref                   # CasADi symbolic (13,)
    P_qih_ca = ca.DM(P_qih)                   # CasADi dense matrix
    h_terminal = ca.mtimes([x_dev.T, P_qih_ca, x_dev])   # scalar (1×1)

    ocp.model.con_h_expr_e = h_terminal

    ocp.constraints.lh_e = np.array([0.0])
    ocp.constraints.uh_e = np.array([alpha_qih])

    # ── Soft Constraint on Terminal Set ─────────────────────────
    #
    #   Slack variable s >= 0 relaxes the upper bound:
    #       h_e <= alpha + s
    #
    #   Penalty added to cost:  zu * s  +  Zu * s^2
    #
    #   zu (L1): ensures exact penalty — constraint is active when
    #            zu > dual variable of the original hard constraint.
    #   Zu (L2): smooths the penalty for the QP solver.
    #
    #   Only the upper bound needs softening (lower bound h_e >= 0
    #   is always satisfied since h_e is a quadratic form).
    #
    ocp.constraints.idxsh_e = np.array([0])    # soften constraint index 0

    ocp.cost.zl_e = np.array([0.0])            # lower slack: no penalty
    ocp.cost.Zl_e = np.array([0.0])
    ocp.cost.zu_e = np.array([1e3])            # L1 penalty on upper violation
    ocp.cost.Zu_e = np.array([1e5])            # L2 penalty on upper violation

    # ── Solver Options ──────────────────────────────────────────
    ocp.solver_options.N_horizon       = N
    ocp.solver_options.tf              = T_horizon
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4       # ERK4
    ocp.solver_options.qp_solver       = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx  = 'GAUSS_NEWTON'
    ocp.solver_options.print_level     = 0

    return AcadosOcpSolver(ocp)
