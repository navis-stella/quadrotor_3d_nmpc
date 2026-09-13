"""
quadrotor_3d_model.py — Full 3D Quadrotor Model (Quaternion)
==============================================================
Contains:
  - Physical parameters
  - CasADi symbolic model              (used by acados NMPC solver)
  - Hover linearization                (used for DARE terminal cost)
  - Disturbance model (Stage 3)        (MPC + plant with disturbance parameters)
  - Augmented model (Stage 3)          (EKF: 13 states + 6 disturbances = 19)
  - AcadosSimSolver plant creator      (used as plant in simulation)
  - Numerical ODE + RK4 backup         (for future model mismatch testing)
  - Quaternion normalization utility

State (13):
    x = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, p, q, r]

    Position:            px, py, pz        [m]      world frame
    Linear velocity:     vx, vy, vz        [m/s]    world frame
    Quaternion:          qw, qx, qy, qz    [-]      scalar-first, body-to-world
    Angular velocity:    p, q, r           [rad/s]  body frame (roll, pitch, yaw rate)

    Note on naming: 'q' as pitch rate (aerospace standard) is separate from
    'qw, qx, qy, qz' quaternion components. Context makes them unambiguous.

Input (4):
    u = [f1, f2, f3, f4]   individual motor thrusts [N]

Motor layout (+ configuration, top view, z-up):

          M1 (front)
           |
    M4 ----+---- M2 (right)
    (left) |
          M3 (back)

    M1 at [ L,  0, 0]   front     spin CW  → reactive tau_z < 0
    M2 at [ 0, -L, 0]   right     spin CCW → reactive tau_z > 0
    M3 at [-L,  0, 0]   back      spin CW  → reactive tau_z < 0
    M4 at [ 0,  L, 0]   left      spin CCW → reactive tau_z > 0

Quaternion convention:
    q = [qw, qx, qy, qz]   scalar-first
    ||q|| = 1                unit norm constraint
    Hover reference: q = [1, 0, 0, 0] (identity rotation)
    q and -q represent the same rotation (double cover)

    Rotation matrix R(q) transforms body → world:
        v_world = R(q) @ v_body

Stage 3 — Offset-free NMPC:
    Disturbance states (6):
        d = [d_fx, d_fy, d_fz, d_tx, d_ty, d_tz]

        d_fx, d_fy, d_fz   [N]     constant force disturbance (world frame)
                                     covers: wind, mass error (Δm·g in z)
        d_tx, d_ty, d_tz   [N·m]   constant torque disturbance (body frame)
                                     covers: CoG offset (parasitic torques)

        Disturbance dynamics: d_dot = 0 (constant disturbance assumption)

    Architecture:
        MPC solver  → uses create_disturbance_model()
                       d enters as runtime parameter p, set from EKF estimate d̂
        Plant sim   → uses create_disturbance_plant()
                       d enters as runtime parameter p, set to true disturbance
        EKF         → uses get_augmented_dynamics_casadi()
                       augmented state z = [x(13); d(6)], NZ = 19
"""

import numpy as np
import casadi as ca
from acados_template import AcadosModel, AcadosSim, AcadosSimSolver

# ─────────────────────────────────────────────────────────────────
# Physical Parameters
# ─────────────────────────────────────────────────────────────────
m     = 1.0       # [kg]     total mass
g     = 9.81      # [m/s²]   gravity
L     = 0.175     # [m]      arm length (center to motor)
Ixx   = 0.01      # [kg·m²]  moment of inertia about x (roll)
Iyy   = 0.01      # [kg·m²]  moment of inertia about y (pitch)
Izz   = 0.02      # [kg·m²]  moment of inertia about z (yaw)
c_tau = 0.0036    # [m]      thrust-to-reactive-torque coefficient

f_hover = m * g / 4.0    # [N] hover thrust per motor ≈ 2.45 N

# State and input dimensions
NX = 13   # 3 pos + 3 vel + 4 quat + 3 angular vel
NU = 4

# Disturbance dimensions (Stage 3)
ND = 6    # 3 force (world) + 3 torque (body)
NZ = NX + ND   # 19 — augmented state dimension for EKF


# ─────────────────────────────────────────────────────────────────
# Quaternion Utility
# ─────────────────────────────────────────────────────────────────
def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion(s) to Euler angles (ZYX convention).

    Args:
        q: (4,) or (N, 4) array  [qw, qx, qy, qz]

    Returns:
        euler: (3,) or (N, 3) array  [roll, pitch, yaw] in radians
    """
    if q.ndim == 1:
        q = q.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Roll (φ)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx**2 + qy**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (θ)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (ψ)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy**2 + qz**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    euler = np.column_stack([roll, pitch, yaw])
    return euler.squeeze() if squeeze else euler


def normalize_quaternion(x: np.ndarray) -> np.ndarray:
    """
    Normalize the quaternion components in the state vector.
    Call after each integration step to prevent drift from ||q|| = 1.

    Also enforces qw > 0 convention to avoid the double-cover ambiguity
    (q and -q represent the same rotation — we pick the qw > 0 hemisphere).
    """
    qw, qx, qy, qz = x[6], x[7], x[8], x[9]
    q_norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    x[6] /= q_norm
    x[7] /= q_norm
    x[8] /= q_norm
    x[9] /= q_norm
    # enforce qw > 0 hemisphere
    if x[6] < 0:
        x[6:10] *= -1
    return x


# ─────────────────────────────────────────────────────────────────
# Helper: Core Symbolic Dynamics (shared by all model variants)
# ─────────────────────────────────────────────────────────────────
def _build_core_symbols():
    """
    Create the shared CasADi symbolic variables for the quadrotor.

    Returns a dict with all symbolic states, inputs, and intermediate
    quantities (T_total, tau_x, tau_y, tau_z) that every model variant
    needs. This avoids code duplication between create_model(),
    create_disturbance_model(), and get_augmented_dynamics_casadi().
    """
    # ── Symbolic states ────────────────────────────────────────
    px  = ca.SX.sym('px')
    py  = ca.SX.sym('py')
    pz  = ca.SX.sym('pz')
    vx  = ca.SX.sym('vx')
    vy  = ca.SX.sym('vy')
    vz  = ca.SX.sym('vz')
    qw  = ca.SX.sym('qw')
    qx  = ca.SX.sym('qx')
    qy  = ca.SX.sym('qy')
    qz  = ca.SX.sym('qz')
    p   = ca.SX.sym('p')
    q   = ca.SX.sym('q')
    r   = ca.SX.sym('r')

    x = ca.vertcat(px, py, pz, vx, vy, vz, qw, qx, qy, qz, p, q, r)

    # ── Symbolic inputs ────────────────────────────────────────
    f1 = ca.SX.sym('f1')
    f2 = ca.SX.sym('f2')
    f3 = ca.SX.sym('f3')
    f4 = ca.SX.sym('f4')
    u  = ca.vertcat(f1, f2, f3, f4)

    # ── Thrust and torques from motor forces ───────────────────
    T_total = f1 + f2 + f3 + f4
    tau_x   = L * (f4 - f2)
    tau_y   = L * (f3 - f1)
    tau_z   = c_tau * (-f1 + f2 - f3 + f4)

    return {
        # Individual symbolic scalars
        'px': px, 'py': py, 'pz': pz,
        'vx': vx, 'vy': vy, 'vz': vz,
        'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
        'p': p, 'q': q, 'r': r,
        'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4,
        # Assembled vectors
        'x': x, 'u': u,
        # Derived quantities
        'T_total': T_total,
        'tau_x': tau_x, 'tau_y': tau_y, 'tau_z': tau_z,
    }


def _build_f_expl(s, d_fx=0, d_fy=0, d_fz=0, d_tx=0, d_ty=0, d_tz=0):
    """
    Assemble the explicit ODE xdot = f(x, u, d) from the core symbols.

    The disturbance terms d_fx..d_tz default to 0 (no disturbance),
    allowing the same function to serve:
      - Nominal model:       d = 0
      - Disturbance model:   d = symbolic parameters or states

    Disturbance injection:
        d_fx, d_fy, d_fz  →  translational dynamics (world frame)
            dvx += d_fx / m
            dvy += d_fy / m
            dvz += d_fz / m

        d_tx, d_ty, d_tz  →  rotational dynamics (body frame)
            dp += d_tx / Ixx
            dq += d_ty / Iyy
            dr += d_tz / Izz
    """
    qw = s['qw']; qx = s['qx']; qy = s['qy']; qz = s['qz']
    p  = s['p'];  q  = s['q'];  r  = s['r']
    vx = s['vx']; vy = s['vy']; vz = s['vz']
    T_total = s['T_total']
    tau_x = s['tau_x']; tau_y = s['tau_y']; tau_z = s['tau_z']

    # ── Subsystem 1: Translational dynamics (world frame) ──────
    #    Only the third column of R(q) is needed for thrust direction.
    dvx = T_total / m * 2 * (qx*qz + qw*qy)            + d_fx / m
    dvy = T_total / m * 2 * (qy*qz - qw*qx)            + d_fy / m
    dvz = T_total / m * (1 - 2*(qx**2 + qy**2)) - g     + d_fz / m

    # ── Subsystem 2: Quaternion kinematics ─────────────────────
    #    q_dot = 0.5 * q ⊗ [0, p, q, r]    (no disturbance here)
    dqw = 0.5 * (-qx*p - qy*q - qz*r)
    dqx = 0.5 * ( qw*p + qy*r - qz*q)
    dqy = 0.5 * ( qw*q - qx*r + qz*p)
    dqz = 0.5 * ( qw*r + qx*q - qy*p)

    # ── Subsystem 3: Rotational dynamics (Euler's equations) ───
    dp_dt = (Iyy - Izz) / Ixx * q * r + tau_x / Ixx     + d_tx / Ixx
    dq_dt = (Izz - Ixx) / Iyy * p * r + tau_y / Iyy     + d_ty / Iyy
    dr_dt = (Ixx - Iyy) / Izz * p * q + tau_z / Izz     + d_tz / Izz

    return ca.vertcat(
        vx, vy, vz,                   # position kinematics
        dvx, dvy, dvz,                # translational dynamics
        dqw, dqx, dqy, dqz,           # quaternion kinematics
        dp_dt, dq_dt, dr_dt           # rotational dynamics
    )


# ═════════════════════════════════════════════════════════════════
# Model Variant 1: Nominal (Stages 1–2)
# ═════════════════════════════════════════════════════════════════
def create_model() -> AcadosModel:
    """
    Full 3D quadrotor nonlinear dynamics — no disturbance.
    Used by Stages 1–2 (basic NMPC, DARE, QIH).
    """
    s = _build_core_symbols()
    f_expl = _build_f_expl(s)

    model             = AcadosModel()
    model.name        = 'quadrotor_3d'
    model.x           = s['x']
    model.u           = s['u']
    model.xdot        = ca.SX.sym('xdot', NX)
    model.f_expl_expr = f_expl

    return model


# ═════════════════════════════════════════════════════════════════
# Model Variant 2: Disturbance as Runtime Parameter (Stage 3)
# ═════════════════════════════════════════════════════════════════
def create_disturbance_model() -> AcadosModel:
    """
    Quadrotor dynamics with disturbance forces/torques as runtime parameters.

    Used by:
      - MPC solver:  set p = d̂  (EKF estimate) at each shooting node
      - Plant sim:   set p = d_true (known external disturbance for testing)

    The state dimension stays NX=13 — disturbances are NOT optimized.
    The 6 disturbance parameters are set online via:
        ocp_solver.set(stage, 'p', d_hat)       # MPC
        plant_sim.set('p', d_true)               # plant

    Parameter vector:
        p = [d_fx, d_fy, d_fz, d_tx, d_ty, d_tz]

        d_fx, d_fy, d_fz  [N]    force disturbance (world frame)
        d_tx, d_ty, d_tz  [N·m]  torque disturbance (body frame)
    """
    s = _build_core_symbols()

    # ── Disturbance parameters ─────────────────────────────────
    d_fx = ca.SX.sym('d_fx')    # force disturbance x  [N]
    d_fy = ca.SX.sym('d_fy')    # force disturbance y  [N]
    d_fz = ca.SX.sym('d_fz')    # force disturbance z  [N]
    d_tx = ca.SX.sym('d_tx')    # torque disturbance x [N·m]
    d_ty = ca.SX.sym('d_ty')    # torque disturbance y [N·m]
    d_tz = ca.SX.sym('d_tz')    # torque disturbance z [N·m]
    p    = ca.vertcat(d_fx, d_fy, d_fz, d_tx, d_ty, d_tz)

    f_expl = _build_f_expl(s,
                           d_fx=d_fx, d_fy=d_fy, d_fz=d_fz,
                           d_tx=d_tx, d_ty=d_ty, d_tz=d_tz)

    model             = AcadosModel()
    model.name        = 'quadrotor_3d_disturb'
    model.x           = s['x']
    model.u           = s['u']
    model.p           = p
    model.xdot        = ca.SX.sym('xdot', NX)
    model.f_expl_expr = f_expl

    return model


# ═════════════════════════════════════════════════════════════════
# Model Variant 3: Augmented Dynamics for EKF (Stage 3)
# ═════════════════════════════════════════════════════════════════
def get_augmented_dynamics_casadi():
    """
    Augmented system dynamics as CasADi function for the EKF.

    Augmented state:
        z = [x(13); d(6)] ∈ R^19

        x = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, p, q, r]
        d = [d_fx, d_fy, d_fz, d_tx, d_ty, d_tz]

    Augmented dynamics:
        z_dot = [ f(x, u, d) ]     ← original dynamics with disturbance
                [ 0           ]     ← constant disturbance assumption

    Returns:
        f_aug:  CasADi Function  z_dot = f_aug(z, u)
                Inputs:  z (19,), u (4,)
                Outputs: z_dot (19,)

        Also returns the symbolic Jacobians for EKF:
        F_func: CasADi Function  df_aug/dz evaluated at (z, u)
                Returns (19×19) matrix

    Usage in EKF:
        z_dot_val = f_aug(z_hat, u)                    # prediction
        F_val     = F_func(z_hat, u)                   # Jacobian for covariance
    """
    s = _build_core_symbols()

    # ── Disturbance states (part of augmented state, not parameters) ──
    d_fx = ca.SX.sym('d_fx')
    d_fy = ca.SX.sym('d_fy')
    d_fz = ca.SX.sym('d_fz')
    d_tx = ca.SX.sym('d_tx')
    d_ty = ca.SX.sym('d_ty')
    d_tz = ca.SX.sym('d_tz')
    d    = ca.vertcat(d_fx, d_fy, d_fz, d_tx, d_ty, d_tz)

    # Augmented state vector
    z = ca.vertcat(s['x'], d)   # (19,)

    # ── Augmented dynamics ─────────────────────────────────────
    f_x = _build_f_expl(s,
                        d_fx=d_fx, d_fy=d_fy, d_fz=d_fz,
                        d_tx=d_tx, d_ty=d_ty, d_tz=d_tz)

    f_d = ca.SX.zeros(ND)   # d_dot = 0 (constant disturbance)

    z_dot = ca.vertcat(f_x, f_d)   # (19,)

    # ── CasADi functions ───────────────────────────────────────
    f_aug = ca.Function('f_aug', [z, s['u']], [z_dot],
                        ['z', 'u'], ['z_dot'])

    # Jacobian df_aug/dz for EKF covariance propagation
    F_sym = ca.jacobian(z_dot, z)   # (19×19) symbolic
    F_func = ca.Function('F_aug', [z, s['u']], [F_sym],
                         ['z', 'u'], ['F'])

    return f_aug, F_func


# ─────────────────────────────────────────────────────────────────
# Linearization at Hover  (for DARE terminal cost — Stage 2)
# ─────────────────────────────────────────────────────────────────
def get_hover_linearization():
    """
    Linearize 3D quadrotor dynamics at hover equilibrium.

    Equilibrium: q=[1,0,0,0], p=q=r=0, fi=mg/4

    Key couplings at hover (quaternion version):
        qy → acceleration in x    (A[3,8]  = 2g)
        qx → acceleration in -y   (A[4,7]  = -2g)

    Compare with Euler version:
        θ  → acceleration in x    (A[3,7]  = g)
        φ  → acceleration in -y   (A[4,6]  = -g)

    The factor of 2 comes from the quaternion rotation matrix:
    the derivative of R(q) w.r.t. qy at hover yields 2, not 1.

    Returns:
        A_c:  (13×13) continuous-time state Jacobian at hover
        B_c:  (13×4)  continuous-time input Jacobian at hover
    """
    A_c = np.zeros((NX, NX))

    # Position kinematics: velocity → position
    A_c[0, 3] = 1.0     # vx → dpx
    A_c[1, 4] = 1.0     # vy → dpy
    A_c[2, 5] = 1.0     # vz → dpz

    # Translational dynamics: quaternion → acceleration
    A_c[3, 8] = 2 * g     # qy → dvx
    A_c[4, 7] = -2 * g    # qx → dvy

    # Quaternion kinematics: angular velocity → quaternion derivative
    A_c[7, 10] = 0.5     # p → dqx
    A_c[8, 11] = 0.5     # q → dqy
    A_c[9, 12] = 0.5     # r → dqz

    B_c = np.zeros((NX, NU))

    # Translational: all motors contribute equally to vertical acceleration
    B_c[5, :] = 1.0 / m

    # Roll:  tau_x = L*(f4-f2)
    B_c[10, 1] = -L / Ixx
    B_c[10, 3] =  L / Ixx

    # Pitch: tau_y = L*(f3-f1)
    B_c[11, 0] = -L / Iyy
    B_c[11, 2] =  L / Iyy

    # Yaw:   tau_z = c_tau*(-f1+f2-f3+f4)
    B_c[12, 0] = -c_tau / Izz
    B_c[12, 1] =  c_tau / Izz
    B_c[12, 2] = -c_tau / Izz
    B_c[12, 3] =  c_tau / Izz

    return A_c, B_c


# ─────────────────────────────────────────────────────────────────
# AcadosSim Plant Simulators
# ─────────────────────────────────────────────────────────────────
def create_plant_simulator(T_horizon: float = 1.0,
                           N: int = 20) -> AcadosSimSolver:
    """
    Create an AcadosSimSolver for plant simulation — no disturbance.
    Used by Stages 1–2.

    One call advances the plant by one sample time Ts = T_horizon/N.
    Call normalize_quaternion(x) after each step.
    """
    model = create_model()
    sim   = AcadosSim()
    sim.model = model
    sim.solver_options.T = T_horizon / N
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    return AcadosSimSolver(sim)


def create_disturbance_plant(T_horizon: float = 1.0,
                             N: int = 20) -> AcadosSimSolver:
    """
    Create an AcadosSimSolver for plant simulation WITH disturbance.
    Used by Stage 3.

    Usage:
        plant.set('x', x_current)
        plant.set('u', u_current)
        plant.set('p', d_true)          # ← set true disturbance
        plant.solve()
        x_next = plant.get('x')
        x_next = normalize_quaternion(x_next)

    The disturbance is integrated continuously within each step,
    not applied as a discrete impulse — physically correct.
    """
    model = create_disturbance_model()
    sim   = AcadosSim()
    sim.model = model
    sim.solver_options.T = T_horizon / N
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4

    # Parameter dimensions must be set for AcadosSim
    sim.parameter_values = np.zeros(ND)

    return AcadosSimSolver(sim)


# ─────────────────────────────────────────────────────────────────
# Numerical ODE + RK4  (backup for model mismatch testing)
# ─────────────────────────────────────────────────────────────────
def f_ode(x: np.ndarray, u: np.ndarray,
          d: np.ndarray = None) -> np.ndarray:
    """
    Same dynamics as create_model() but in numerical form.
    Optional disturbance vector d = [d_fx, d_fy, d_fz, d_tx, d_ty, d_tz].
    """
    px, py, pz, vx, vy, vz, qw, qx, qy, qz, p, q, r = x
    f1, f2, f3, f4 = u

    if d is None:
        d = np.zeros(ND)
    d_fx, d_fy, d_fz, d_tx, d_ty, d_tz = d

    T_total = f1 + f2 + f3 + f4
    tau_x   = L * (f4 - f2)
    tau_y   = L * (f3 - f1)
    tau_z   = c_tau * (-f1 + f2 - f3 + f4)

    # translational dynamics (world frame)
    dvx = T_total / m * 2 * (qx*qz + qw*qy)         + d_fx / m
    dvy = T_total / m * 2 * (qy*qz - qw*qx)         + d_fy / m
    dvz = T_total / m * (1 - 2*(qx**2 + qy**2)) - g  + d_fz / m

    # quaternion kinematics
    dqw = 0.5 * (-qx*p - qy*q - qz*r)
    dqx = 0.5 * ( qw*p + qy*r - qz*q)
    dqy = 0.5 * ( qw*q - qx*r + qz*p)
    dqz = 0.5 * ( qw*r + qx*q - qy*p)

    # rotational dynamics (body frame)
    dp = (Iyy - Izz) / Ixx * q*r + tau_x / Ixx       + d_tx / Ixx
    dq = (Izz - Ixx) / Iyy * p*r + tau_y / Iyy       + d_ty / Iyy
    dr = (Ixx - Iyy) / Izz * p*q + tau_z / Izz        + d_tz / Izz

    return np.array([vx, vy, vz, dvx, dvy, dvz,
                     dqw, dqx, dqy, dqz, dp, dq, dr])


def rk4_step(x: np.ndarray, u: np.ndarray, dt: float,
             d: np.ndarray = None) -> np.ndarray:
    """
    One RK4 integration step — backup plant simulator.
    Remember to call normalize_quaternion() after this.
    """
    k1 = f_ode(x,            u, d)
    k2 = f_ode(x + dt/2*k1,  u, d)
    k3 = f_ode(x + dt/2*k2,  u, d)
    k4 = f_ode(x + dt   *k3, u, d)
    x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return normalize_quaternion(x_next)
