"""
ocp_config_mhe.py — Stage 4b Lyapunov MHE: OCP + δ-IOSS Certificate I/O
=======================================================================
acados OCP of the Moving Horizon Estimator for the augmented state — the
same z = [x; d] as the Stage 4a EKF, so the two estimators differ only in
how they solve the estimation problem. The runtime wrapper (buffers,
growing horizon, warm start) is state_est_mhe.py.

Conventions
───────────
Estimated state (NZ = 19):          z = [ p(3), v(3), q(4), ω(3), d(6) ]
Process noise (NW = 19):            w — additive on every derivative of z,
                                    the decision variable (acados model.u)
Known input (NU = 4):               u_rotor = [ f1, f2, f3, f4 ]
                                    → acados PARAMETER (model.p), because
                                      model.u is taken by w
Measurement (NY = 10):              y = [ p(3), q(4), ω(3) ]
Model:                              quadrotor_3d_model.create_mhe_model()

Design order (Theorem 1/2 of Schiller & Müller, 2024) — one decay symbol, λ
    λ (per second) → κ = −ln λ → LMI → P, Q, R, T_min = ln4/κ   detectability_check.py
    → N = ⌈c·T_min/Ts⌉, c = HORIZON_FACTOR = 1.5                load_mhe_params
    The MHE horizon N·Ts is independent of the MPC horizon — it sets the
    length of the CT estimation window.

Cost — the CT cost (9) of Schiller & Müller (2024) on the acados grid
    2‖χ̄ − ẑ(t−T)‖²_P · λ^T  +  ∫_{t−T}^{t} λ^{t−τ} (2‖w‖²_Q + ‖Δy‖²_R) dτ
    λ per SECOND, (P, Q, R) from the δ-IOSS certificate.

    Every node i carries its AGE  τ_i = t − t_i  [s] as a parameter (set by
    state_est_mhe from the grid), and the paper's discount λ^{t−τ} is
    evaluated at the node's own time:
    stage i = 0..N-1, one expression for all stages — the INTEGRAND:
          a_i · 2·λ^{τ_i} · ‖e_z‖²_P / Ts                          (arrival)
        +       2·λ^{τ_i} · ‖w_i‖²_Q
        + m_i ·   λ^{τ_i} · ‖e_y,i‖²_R
    acados multiplies every stage cost by its time step (default
    cost_scaling = time_steps, terminal 1), which turns the integrand into
    the Riemann sum Σ Ts·λ^{τ_i}(…) of the integral. The arrival term is a
    point cost, not an integral — the / Ts cancels acados' scaling so it
    carries exactly 2·λ^T·P. (Uniform grid; the terminal node has no cost.)
    Node-time rule: y_i is sampled at t_i, so its discount λ^{τ_i} is exact;
    for w (constant on [t_i, t_{i+1})) it is the left end of the interval.
    a_i, m_i ∈ {0, 1} are per-node flags set by the outer loop:
        full horizon (t ≥ T):  a = [1, 0, …, 0],  m = 1 everywhere
        growing horizon (t = k·Ts < T, paper: T_i = min(t_i, T)):
            the window [0, t] occupies nodes N-k..N; the prior z̄₀ sits at
            node N-k (a = 1 there, discount λ^{τ} = λ^t). Nodes 0..N-k-1 are
            padding: no data, no arrival, z₀ free, w penalized — so the
            optimum is w = 0 there and they do not bias the estimate.

State constraint (8c): the estimated trajectory must stay in the set X on
which the certificate holds → hard box |ω| ≤ ω_max on every node. Only
sensible when the TRUE ω stays inside the box too — the MPC keeps
|ω| ≤ 0.9·ω_max (ocp_config_offsetfree.add_soft_omega_box).
(‖q‖ ≤ r_max is covered in practice by the soft ‖q‖² = 1 constraint.)

Quaternion trick
────────────────
For unit q and q_ref,  q_err = q ⊗ q_ref⁻¹  points to ±[1,0,0,0] when the
two rotations are aligned (double cover of SO(3)). We use
    e_q = |q_err| − [1, 0, 0, 0]
so the cost term vanishes for both signs and stays smooth.
"""

import os
import numpy as np
from casadi import SX, vertcat, mtimes, power, fabs, sumsqr
from acados_template import AcadosOcp, AcadosOcpSolver

from quadrotor_3d_model import (
    create_mhe_model,
    measurement_expr,
    quat_error_casadi,
    NU, NY, NZ, NW,
)


# =============================================================================
#  δ-IOSS certificate  (written by detectability_check.py, read by the MHE)
# =============================================================================
#  The MHE weights come from an LMI that is slow, needs an SDP solver and must
#  pass a-posteriori verification — so it is solved once and stored. Writer
#  and reader both go through this pair so the keys are defined in one place.
#
#  Stored keys (all continuous-time — independent of the MHE sample time):
#    P            (NZ, NZ)  δ-IOSS Lyapunov weight        → MHE arrival cost
#    Q, R         (NW, NW), (NY, NY)  CT supply rates      → MHE stage cost
#    lam          decay per SECOND, λ = e^{−κ}   (the design input)
#    kappa        decay rate [1/s],  κ = −ln λ
#    T_hor_min    minimum horizon (17):  T > ln4/κ
#    omega_max, T_range, r_max    certified envelope X × U
#    pool_max_eig, spot_max_eig, active_quats                  (provenance)
#
#  The horizon and the grid quantities (N, T_hor, N_min, ρ) are derived for
#  the actual MHE sample time in load_mhe_params — changing Ts_mhe or the
#  horizon factor needs no new SDP solve.
#
#  Only certificates that PASSED verification are written.

MHE_PARAMS_PATH = 'data/mhe_params.npz'
HORIZON_FACTOR  = 1.5            # T_hor ≥ c·T_min   (c > 1 required by (17))


def save_mhe_params(path: str = MHE_PARAMS_PATH, **fields):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savez(path, **fields)


def load_mhe_params(path:           str   = MHE_PARAMS_PATH,
                    Ts:             float = 0.05,
                    horizon_factor: float = HORIZON_FACTOR) -> dict:
    """
    Load a verified δ-IOSS certificate, choose the horizon and discretize
    it for the MHE grid.

    Ts is the MHE sample time. Added keys:
        Ts              MHE sample time [s]
        N               horizon in stages, N = ⌈c·T_min/Ts⌉ (c = horizon_factor)
        T_hor           horizon time N·Ts [s]  (> T_min, independent of the MPC)
        N_min           smallest N with N·Ts > ln4/κ   (17), i.e. 4·λ^T < 1
        rho             guaranteed convergence rate per second (18),
                        ρ = 4^{1/T}·λ ≈ λ^{1−1/c}
    Q and R stay CT; acados' stage-cost scaling supplies the Ts weighting.
    """
    assert horizon_factor > 1.0, 'horizon_factor must exceed 1 — (17) is strict'

    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found — run detectability_check.py first '
            f'(it only writes the file once verification passes).')

    with np.load(path) as f:
        cert = {k: f[k] for k in f.files}
    for k in ('lam', 'kappa', 'T_hor_min',
              'omega_max', 'r_max', 'pool_max_eig', 'spot_max_eig'):
        cert[k] = float(cert[k])

    # Guards against a stale file from a different model
    assert cert['P'].shape == (NZ, NZ), f"P shape {cert['P'].shape} — stale certificate?"
    assert cert['Q'].shape == (NW, NW), f"Q shape {cert['Q'].shape}"
    assert cert['R'].shape == (NY, NY), f"R shape {cert['R'].shape}"
    assert np.isclose(cert['lam'], np.exp(-cert['kappa'])), 'λ ≠ e^{−κ}'

    # ── Horizon: N = ⌈c·T_min/Ts⌉  (the 1e-9 guards against 19.000000001) ──
    N = int(np.ceil(horizon_factor * cert['T_hor_min'] / Ts - 1e-9))

    # ── Grid quantities ──
    cert['Ts']    = float(Ts)
    cert['N']     = N
    cert['T_hor'] = N * Ts
    # strict (17): N·Ts > T_min, i.e. N_min = ⌊ln4/(κ·Ts)⌋ + 1 (the 1e-9 keeps an
    # exact multiple, e.g. T_min = 2 s at Ts = 0.05 s, from rounding down to 39)
    cert['N_min'] = int(np.floor(cert['T_hor_min'] / Ts + 1e-9)) + 1
    cert['rho']   = 4.0 ** (1.0 / cert['T_hor']) * cert['lam']

    assert N >= cert['N_min'], \
        f"N = {N} < N_min = {cert['N_min']} — horizon bound (17) violated"
    return cert


# =============================================================================
#  Per-node parameter layout
# =============================================================================
#    p = [ t_age(1), a_arr(1), m_meas(1), z_prior(NZ), y_meas(NY), u_rotor(NU) ]
#      = 3 + 19 + 10 + 4 = 36
#  export_drone_mhe_solver builds the symbols in this order, pack_mhe_param
#  fills them — keep the two next to each other.
N_PARAM = 3 + NZ + NY + NU


def pack_mhe_param(t_age:   float,
                   a_arr:   float,
                   m_meas:  float,
                   z_prior: np.ndarray,
                   y_meas:  np.ndarray,
                   u_rotor: np.ndarray) -> np.ndarray:
    """Numeric per-node parameter vector (N_PARAM,) for mhe_solver.set(i, 'p', …)."""
    return np.concatenate([[float(t_age), float(a_arr), float(m_meas)],
                           z_prior, y_meas, u_rotor])


# =============================================================================
#  MHE solver
# =============================================================================
def export_drone_mhe_solver(ts:        float,
                            lam:       float,
                            N:         int,
                            P:         np.ndarray,
                            R:         np.ndarray,
                            Q:         np.ndarray,
                            omega_max: float = None) -> AcadosOcpSolver:
    """
    Build the acados MHE OCP for the 3D quadrotor (augmented state).

    Parameters
    ----------
    ts        : MHE sampling time [s]
    lam       : decay per second λ ∈ (0, 1) — discount λ^{τ} with τ the node age [s]
    N         : horizon length (shooting intervals), N·ts = T_hor
    P         : (NZ, NZ) arrival-cost weight            (certificate, as is)
    R         : (NY, NY) CT measurement supply rate     (as is — acados scales by ts)
    Q         : (NW, NW) CT process-noise supply rate   (as is — acados scales by ts)
    omega_max : certified |ωᵢ| bound → hard box on every node, None = no box

    All arguments normally come straight from load_mhe_params(Ts=ts).
    The weighting matrices are baked into the generated C code as
    constants. Promote them to acados parameters if online tuning is needed.
    """
    # ── dynamic + observation model (from quadrotor_3d_model) ───────────
    #    z(19), u = w(19), p = rotor(4)
    model = create_mhe_model()

    ocp        = AcadosOcp()
    ocp.model  = model
    ocp.dims.N = N

    z       = model.x
    w       = model.u
    u_rotor = model.p                      # rotor thrusts (augmented below)

    y_pred = measurement_expr(z)           # 10-vector expression, y = [p; q; ω]

    # ── per-stage side parameters for the cost ──────────────────────────
    t_age   = SX.sym('t_age',   1)         # node age τ_i = t − t_i [s] → discount λ^{τ_i}
    a_arr   = SX.sym('a_arr',   1)         # 1 → this node carries the arrival cost
    m_meas  = SX.sym('m_meas',  1)         # 1 → this node carries a measurement
    z_prior = SX.sym('z_prior', NZ)        # arrival-cost reference
    y_meas  = SX.sym('y_meas',  NY)        # per-stage measurement

    # ── state-error with the quaternion trick ───────────────────────────
    #    p, v, ω, d : direct subtraction
    #    q          : |q ⊗ q_prior⁻¹| − [1,0,0,0]
    err_q = fabs(quat_error_casadi(z[6:10], z_prior[6:10])) - vertcat(1, 0, 0, 0)
    state_error = vertcat(z[0:6]  - z_prior[0:6],              # p, v
                          err_q,                                # q
                          z[10:] - z_prior[10:])                # ω, d   → 19

    # ── measurement-error with the quaternion trick ─────────────────────
    err_yq = fabs(quat_error_casadi(y_pred[3:7], y_meas[3:7])) - vertcat(1, 0, 0, 0)
    meas_error = vertcat(y_pred[0:3]  - y_meas[0:3],           # p
                         err_yq,                                # q
                         y_pred[7:10] - y_meas[7:10])           # ω      → 10

    # ── cost terms (integrand — acados multiplies each stage by ts) ─────
    #    discount λ^{t−τ} at the node's own time; the arrival is a point
    #    cost, so / ts cancels acados' stage scaling → 2·λ^T·P exactly
    discount           = power(lam, t_age)
    arrival_cost       = a_arr  * 2 * discount / ts * mtimes(state_error.T,
                                                            mtimes(P, state_error))
    process_noise_cost =          2 * discount      * mtimes(w.T, mtimes(Q, w))
    measurement_cost   = m_meas *     discount      * mtimes(meas_error.T,
                                                            mtimes(R, meas_error))

    # ── augment model.p — order must match pack_mhe_param ───────────────
    ocp.model.p = vertcat(t_age, a_arr, m_meas, z_prior, y_meas, u_rotor)
    assert ocp.model.p.rows() == N_PARAM
    ocp.parameter_values = np.zeros(N_PARAM)

    # ── external cost registration (same expression on every stage) ─────
    stage_cost = arrival_cost + process_noise_cost + measurement_cost

    ocp.cost.cost_type_0           = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_0 = stage_cost
    ocp.cost.cost_type             = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost   = stage_cost

    # ── soft ‖q‖² = 1 constraint at every shooting node ─────────────────
    #    node 0, nodes 1..N-1 and node N (its state is handed to the MPC)
    q_norm_sq = sumsqr(z[6:10])
    one, soft = np.array([1.0]), np.array([1e4])
    zero, idx = np.array([0.0]), np.array([0], dtype=int)

    ocp.model.con_h_expr_0  = q_norm_sq
    ocp.constraints.lh_0    = one;  ocp.constraints.uh_0 = one
    ocp.constraints.idxsh_0 = idx
    ocp.cost.Zl_0 = soft;  ocp.cost.Zu_0 = soft;  ocp.cost.zl_0 = zero;  ocp.cost.zu_0 = zero

    ocp.model.con_h_expr    = q_norm_sq
    ocp.constraints.lh      = one;  ocp.constraints.uh   = one
    ocp.constraints.idxsh   = idx
    ocp.cost.Zl   = soft;  ocp.cost.Zu   = soft;  ocp.cost.zl   = zero;  ocp.cost.zu   = zero

    ocp.model.con_h_expr_e  = q_norm_sq
    ocp.constraints.lh_e    = one;  ocp.constraints.uh_e = one
    ocp.constraints.idxsh_e = idx
    ocp.cost.Zl_e = soft;  ocp.cost.Zu_e = soft;  ocp.cost.zl_e = zero;  ocp.cost.zu_e = zero

    # ── (8c): estimated trajectory inside the certified set, |ωᵢ| ≤ ω_max ──
    #    Always feasible — w_ω can bend any trajectory back into the box.
    if omega_max is not None:
        idx_omega = np.arange(10, 13)
        lb_omega  = np.full(3, -omega_max)
        ub_omega  = np.full(3, +omega_max)

        ocp.constraints.idxbx_0 = idx_omega                   # no x0 — node 0 is free
        ocp.constraints.lbx_0   = lb_omega
        ocp.constraints.ubx_0   = ub_omega
        ocp.constraints.idxbx   = idx_omega
        ocp.constraints.lbx     = lb_omega
        ocp.constraints.ubx     = ub_omega
        ocp.constraints.idxbx_e = idx_omega
        ocp.constraints.lbx_e   = lb_omega
        ocp.constraints.ubx_e   = ub_omega

    # ── solver options ──────────────────────────────────────────────────
    ocp.solver_options.qp_solver           = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx      = 'EXACT'
    ocp.solver_options.integrator_type     = 'ERK'
    ocp.solver_options.N_horizon           = N
    ocp.solver_options.tf                  = N * ts
    ocp.solver_options.nlp_solver_type     = 'SQP_RTI'
    ocp.solver_options.nlp_solver_max_iter = 200
    ocp.solver_options.tol                 = 1e-10
    ocp.solver_options.qp_tol              = 1e-10

    return AcadosOcpSolver(ocp)
