"""
ocp_config_offsetfree.py — Stage 3–4 Offset-Free NMPC Solver
============================================================
Creates the acados OCP solver using the disturbance-augmented model.
Stage 4 uses it unchanged for both estimators (EKF and MHE) — the only
Stage 4 addition is the optional soft body-rate box (add_soft_omega_box).

Key difference from Stage 2 (ocp_config_dare.py, tag stage2):
    The prediction model includes disturbance forces/torques as runtime
    parameters. At each MPC step, the estimate d̂ (EKF or MHE) is injected
    into every shooting node, so the MPC predicts the future trajectory
    accounting for the estimated disturbance.

    This is what eliminates steady-state offset:
        Standard MPC:     min || f(x,u) - ref ||   → ignores disturbance → offset
        Offset-free MPC:  min || f(x,u,d̂) - ref || → compensates d̂ → zero offset

Architecture:
    Model:     create_disturbance_model()   →  model.p = [d_fx,...,d_tz] (6 params)
    Terminal:  P_lqr from reduced-order DARE (same as Stage 2)
    Cost:      NONLINEAR_LS  ||[x; u] - [x_ref; u_ref]||²_W
    Solver:    SQP_RTI + ERK4

Runtime usage:
    # Each MPC step (z_hat from the EKF or the MHE):
    x_s, u_s = compute_ss_target(x_ref, d_hat)     # ss_target.py
    set_disturbance_param(ocp_solver, d_hat, N)
    set_reference        (ocp_solver, x_s, u_s, N)
    ocp_solver.set(0, 'lbx', x_hat)
    ocp_solver.set(0, 'ubx', x_hat)
    ocp_solver.solve()
"""

import numpy as np
import casadi as ca
from scipy.linalg import block_diag, solve_discrete_are
from scipy.signal import cont2discrete
from acados_template import AcadosOcp, AcadosOcpSolver

from quadrotor_3d_model import (
    create_disturbance_model, get_hover_linearization,
    f_hover, F_MAX, NX, NU, ND,
)


# ─────────────────────────────────────────────────────────────────
# DARE Terminal Cost — Reduced-Order for Quaternion
# ─────────────────────────────────────────────────────────────────
def compute_dare_terminal_cost(Q: np.ndarray,
                               R: np.ndarray,
                               Ts: float) -> np.ndarray:
    """
    Compute P_lqr via reduced-order DARE (same as Stage 2).

    Removes qw (index 6), solves DARE on 12 states, embeds back.
    The disturbance does NOT enter the DARE — the terminal cost
    penalizes deviation from the reference, and d̂ handles the offset
    through the prediction model, not the terminal cost.

    Returns:
        P_lqr: (13×13) terminal cost matrix
    """
    A_c, B_c = get_hover_linearization()

    idx_keep = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    A_c_red = A_c[np.ix_(idx_keep, idx_keep)]
    B_c_red = B_c[idx_keep, :]
    Q_red   = Q[np.ix_(idx_keep, idx_keep)]

    n_red = len(idx_keep)
    sys_d = cont2discrete(
        (A_c_red, B_c_red, np.eye(n_red), np.zeros((n_red, NU))),
        Ts, method='zoh'
    )
    A_d_red, B_d_red = sys_d[0], sys_d[1]

    P_red = solve_discrete_are(A_d_red, B_d_red, Q_red, R)

    eigvals_red = np.linalg.eigvalsh(P_red)
    assert np.all(eigvals_red > 0), \
        f'P_red not positive definite! eigvals={eigvals_red}'

    P_lqr = np.zeros((NX, NX))
    P_lqr[np.ix_(idx_keep, idx_keep)] = P_red
    P_lqr[6, 6] = Q[6, 6]

    print('─── DARE Terminal Cost (Offset-Free NMPC) ─────────')
    print(f'  Sample time Ts     = {Ts:.4f} s')
    print(f'  P_lqr diagonal     = {np.diag(P_lqr)}')
    print(f'  Ratio P/Q diag     = {np.diag(P_lqr) / np.diag(Q)}')
    print('───────────────────────────────────────────────────')

    return P_lqr


# ─────────────────────────────────────────────────────────────────
# Optional state constraint: body-rate box (soft)
# ─────────────────────────────────────────────────────────────────
def add_soft_omega_box(ocp: AcadosOcp, omega_max: float,
                       z_l1: float = 1e3, z_l2: float = 1e2):
    """
    |p|, |q|, |r| ≤ omega_max on nodes 1..N (node 0 is fixed by x0).

    Why: the MHE's δ-IOSS certificate only holds on |ωᵢ| ≤ ω_max
    (detectability_check.py). Without this box the MPC commands up to
    ~5.6 rad/s in the take-off transient — the true state leaves the
    certified set and the guarantee is void.

    Soft (L1 + L2 slack penalty): the box is a performance/validity limit,
    not a physical one, and a hard box could make the QP infeasible when
    an estimate starts slightly outside it. z_l1 is large enough to act as
    an exact penalty, i.e. the box holds whenever it can be held.
    """
    idx = np.array([10, 11, 12])
    lb  = np.full(3, -omega_max)
    ub  = np.full(3, +omega_max)
    ns  = len(idx)

    ocp.constraints.idxbx    = idx
    ocp.constraints.lbx      = lb
    ocp.constraints.ubx      = ub
    ocp.constraints.idxsbx   = np.arange(ns)
    ocp.cost.zl = np.full(ns, z_l1);   ocp.cost.zu = np.full(ns, z_l1)
    ocp.cost.Zl = np.full(ns, z_l2);   ocp.cost.Zu = np.full(ns, z_l2)

    ocp.constraints.idxbx_e  = idx
    ocp.constraints.lbx_e    = lb
    ocp.constraints.ubx_e    = ub
    ocp.constraints.idxsbx_e = np.arange(ns)
    ocp.cost.zl_e = np.full(ns, z_l1); ocp.cost.zu_e = np.full(ns, z_l1)
    ocp.cost.Zl_e = np.full(ns, z_l2); ocp.cost.Zu_e = np.full(ns, z_l2)


# ─────────────────────────────────────────────────────────────────
# Solver Creator
# ─────────────────────────────────────────────────────────────────
def create_solver(x_ref:     np.ndarray,
                  N:         int   = 20,
                  T_horizon: float = 1.0,
                  omega_max: float = None) -> AcadosOcpSolver:
    """
    Build the acados NMPC solver for offset-free control.

    Uses create_disturbance_model() which adds model.p = [d_fx,...,d_tz].
    The parameter p is set to the estimate d̂ (EKF or MHE) at runtime.

    Args:
        x_ref:      reference state (13,)
        N:          prediction horizon steps
        T_horizon:  prediction horizon duration [s]
        omega_max:  optional soft |ωᵢ| bound [rad/s] (add_soft_omega_box);
                    None = no rate limit (the Stage 3 formulation)

    Cost structure (NONLINEAR_LS):
        Stage:    || [x; u] - [x_ref; u_ref] ||²_W         (17-dim)
        Terminal: || x_N - x_ref ||²_{P_lqr}                (13-dim)

    Constraints:
        0 <= fi <= F_MAX   (box constraints on motor thrust)

    Parameters:
        p = [d_fx, d_fy, d_fz, d_tx, d_ty, d_tz]  (6-dim)
        Set at each shooting node k = 0,...,N via:
            ocp_solver.set(k, 'p', d_hat)

    Returns:
        AcadosOcpSolver
    """
    # ── Use disturbance model (has model.p) ────────────────────
    model = create_disturbance_model()
    ocp   = AcadosOcp()
    ocp.model = model

    Ts    = T_horizon / N
    u_ref = np.full(NU, f_hover)

    # ── Cost ────────────────────────────────────────────────────
    ocp.cost.cost_type   = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    ocp.model.cost_y_expr   = ca.vertcat(model.x, model.u)   # (17,)
    ocp.model.cost_y_expr_e = model.x                        # (13,)

    # ── Weight matrices (same as Stage 2) ───────────────────────
    Q = np.diag([
        80.,  80.,  120.,           # px, py, pz
        10.,  10.,   15.,           # vx, vy, vz
        10., 120.,  120.,  80.,     # qw, qx, qy, qz
         1.,   1.,    1.            # p,  q,  r
    ])
    R = np.diag([0.1, 0.1, 0.1, 0.1])

    P_lqr = compute_dare_terminal_cost(Q, R, Ts)

    ocp.cost.W   = block_diag(Q, R)
    ocp.cost.W_e = P_lqr

    ocp.cost.yref   = np.concatenate([x_ref, u_ref])   # (17,)
    ocp.cost.yref_e = x_ref                            # (13,)

    # ── Constraints ─────────────────────────────────────────────
    ocp.constraints.x0 = np.zeros(NX)

    ocp.constraints.lbu   = np.zeros(NU)
    ocp.constraints.ubu   = np.full(NU, F_MAX)
    ocp.constraints.idxbu = np.arange(NU)

    if omega_max is not None:
        add_soft_omega_box(ocp, omega_max)

    # ── Parameter default (zero disturbance) ────────────────────
    #
    # acados needs to know the parameter dimension at compile time.
    # Default is zero disturbance — updated at runtime from the estimator.
    #
    ocp.parameter_values = np.zeros(ND)

    # ── Solver options ──────────────────────────────────────────
    ocp.solver_options.N_horizon       = N
    ocp.solver_options.tf              = T_horizon
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.qp_solver       = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx  = 'GAUSS_NEWTON'
    ocp.solver_options.print_level     = 0

    return AcadosOcpSolver(ocp)


# ─────────────────────────────────────────────────────────────────
# Helper: inject d̂ into all shooting nodes
# ─────────────────────────────────────────────────────────────────
def set_disturbance_param(ocp_solver: AcadosOcpSolver,
                          d_hat: np.ndarray,
                          N: int):
    """
    Set the disturbance estimate d̂ at every shooting node.

    This is called once per MPC step, before solve().
    The same d̂ is used across all nodes (constant disturbance assumption).

    Args:
        ocp_solver:  the acados OCP solver
        d_hat:       (6,) estimated disturbance (EKF or MHE)
        N:           prediction horizon steps
    """
    for k in range(N + 1):   # nodes 0, 1, ..., N (including terminal)
        ocp_solver.set(k, 'p', d_hat)


# ─────────────────────────────────────────────────────────────────
# Helper: update MPC reference to equilibrium (x_s, u_s)
# ─────────────────────────────────────────────────────────────────
def set_reference(ocp_solver: AcadosOcpSolver,
                  x_s: np.ndarray,
                  u_s: np.ndarray,
                  N: int):
    """
    Update the MPC reference at all shooting nodes to the equilibrium.

    Called once per MPC step (after ss_target computes x_s, u_s).

    Without this, the MPC tracks the fixed (x_ref, u_hover) which
    may be physically inconsistent under disturbance. With this,
    the MPC tracks the achievable equilibrium — eliminating the
    attitude offset.

    Args:
        ocp_solver:  the acados OCP solver
        x_s:         (13,) equilibrium state from target calculator
        u_s:         (4,)  equilibrium input from target calculator
        N:           prediction horizon steps
    """
    yref   = np.concatenate([x_s, u_s])   # (17,)
    yref_e = x_s                          # (13,)

    for k in range(N):
        ocp_solver.set(k, 'yref', yref)
    ocp_solver.set(N, 'yref', yref_e)
