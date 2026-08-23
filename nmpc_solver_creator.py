"""
nmpc_solver_creator.py — acados NMPC Solver for 3D Quadrotor
==============================================================
Creates and returns the acados OCP solver.

Stage 1:  W_e = Q   (simple terminal cost, no terminal constraint)

Extending in future stages:
    Stage 2 → W_e = P_lqr from DARE (principled terminal cost)
    Stage 3 → augmented model + offset-free formulation
    Stage 4 → obstacle avoidance state constraints
"""

import numpy as np
import casadi as ca
from scipy.linalg import block_diag
from acados_template import AcadosOcp, AcadosOcpSolver

from quadrotor_3d_model import create_model, f_hover, NX, NU


# ─────────────────────────────────────────────────────────────────
# Solver Creator
# ─────────────────────────────────────────────────────────────────
def create_solver(x_ref:     np.ndarray,
                  N:         int   = 20,
                  T_horizon: float = 1.0) -> AcadosOcpSolver:
    """
    Build and compile the acados NMPC solver for 3D quadrotor.

    Args:
        x_ref:      reference state (12,) [px,py,pz, vx,vy,vz, phi,theta,psi, p,q,r]
        N:          prediction horizon (number of steps)
        T_horizon:  prediction horizon duration [s]

    Cost structure (NONLINEAR_LS):
        Stage:    || [x; u] - [x_ref; u_ref] ||^2_W
        Terminal: || x_N - x_ref ||^2_Q

    Constraints:
        0 <= fi <= f_max   for each motor  (box constraints on input)

    Solver:  SQP_RTI + ERK4

    Returns:
        AcadosOcpSolver ready to be called in the simulation loop
    """
    model = create_model()
    ocp   = AcadosOcp()
    ocp.model = model

    u_ref = np.array([f_hover, f_hover, f_hover, f_hover])  # hover input

    # ── Cost ────────────────────────────────────────────────────
    ocp.cost.cost_type   = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    # cost expressions: what we penalize
    ocp.model.cost_y_expr   = ca.vertcat(model.x, model.u)  # stage:    [x; u]
    ocp.model.cost_y_expr_e = model.x                        # terminal: [x]

    # weight matrices
    Q = np.diag([
        80.,  80.,  120.,     # px, py, pz        position: high penalty
        10.,  10.,   15.,     # vx, vy, vz        velocity: moderate
        30.,  30.,   20.,     # phi, theta, psi   attitude: moderate-high
         1.,   1.,    1.      # p,  q,  r         angular rate: low
    ])

    R = np.diag([0.1, 0.1, 0.1, 0.1])   # f1, f2, f3, f4 — input: low

    ocp.cost.W   = block_diag(Q, R)       # stage weight (16x16)
    ocp.cost.W_e = Q                      # terminal weight — Stage 2: replace with P_lqr

    # references
    ocp.cost.yref   = np.concatenate([x_ref, u_ref])   # stage reference (16,)
    ocp.cost.yref_e = x_ref                             # terminal reference (12,)

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
