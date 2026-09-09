"""
quadrotor_3d_model.py — Full 3D Quadrotor Model (Quaternion)
==============================================================
Contains:
  - Physical parameters
  - CasADi symbolic model              (used by acados NMPC solver)
  - Hover linearization                (used for DARE terminal cost)
  - AcadosSimSolver plant creator      (used as plant in simulation)
  - Numerical ODE + RK4 backup         (for future model mismatch testing)
  - Quaternion utilities               (normalization, quaternion → Euler conversion)

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
    ||q|| = 1              unit norm constraint
    Hover reference: q = [1, 0, 0, 0] (identity rotation)
    q and -q represent the same rotation (double cover)

    Rotation matrix R(q) transforms body → world:
        v_world = R(q) @ v_body

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

# state and input dimensions
NX = 13   # 3 pos + 3 vel + 4 quat + 3 angular vel
NU = 4


# ─────────────────────────────────────────────────────────────────
# Quaternion Utility
# ─────────────────────────────────────────────────────────────────
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


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [qw, qx, qy, qz] to Euler angles [roll, pitch, yaw].

    Uses the ZYX intrinsic rotation convention (aerospace standard):
        roll  (φ) = rotation about body x
        pitch (θ) = rotation about body y
        yaw   (ψ) = rotation about body z

    Derivation from the rotation matrix R(q):
        φ = atan2(R21, R22) = atan2(2(qw·qx + qy·qz), 1 - 2(qx² + qy²))
        θ = asin (R20)      = asin (2(qw·qy - qz·qx))
        ψ = atan2(R10, R00) = atan2(2(qw·qz + qx·qy), 1 - 2(qy² + qz²))

    The asin is clamped to [-1, 1] to avoid NaN from floating-point overshoot
    near ±90° pitch (gimbal lock zone — not reachable in normal flight).

    Args:
        q:  (4,) or (N, 4) array of [qw, qx, qy, qz]

    Returns:
        euler:  same shape with [roll, pitch, yaw] in radians
    """
    q = np.atleast_2d(q)
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # roll (φ)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx**2 + qy**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (θ) — clamp to avoid NaN at ±90°
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # yaw (ψ)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy**2 + qz**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    result = np.column_stack([roll, pitch, yaw])
    return result.squeeze()


# ─────────────────────────────────────────────────────────────────
# CasADi Symbolic Model  (used by acados NMPC)
# ─────────────────────────────────────────────────────────────────
def create_model() -> AcadosModel:
    """
    Full 3D quadrotor nonlinear dynamics with quaternion attitude.

    Three subsystems:
        1. Translational dynamics   (world frame, driven by thrust + gravity)
        2. Quaternion kinematics    (body rates → quaternion derivative)
        3. Rotational dynamics      (body frame, Euler's equations)

    No singularity — valid for any orientation.
    """
    # ── Symbolic states ────────────────────────────────────────
    px  = ca.SX.sym('px')       # position x [m]
    py  = ca.SX.sym('py')       # position y [m]
    pz  = ca.SX.sym('pz')       # position z [m]
    vx  = ca.SX.sym('vx')       # velocity x [m/s]
    vy  = ca.SX.sym('vy')       # velocity y [m/s]
    vz  = ca.SX.sym('vz')       # velocity z [m/s]
    qw  = ca.SX.sym('qw')       # quaternion scalar part
    qx  = ca.SX.sym('qx')       # quaternion vector x
    qy  = ca.SX.sym('qy')       # quaternion vector y
    qz  = ca.SX.sym('qz')       # quaternion vector z
    p   = ca.SX.sym('p')        # roll rate  [rad/s]  body frame
    q   = ca.SX.sym('q')        # pitch rate [rad/s]  body frame
    r   = ca.SX.sym('r')        # yaw rate   [rad/s]  body frame

    x    = ca.vertcat(px, py, pz, vx, vy, vz, qw, qx, qy, qz, p, q, r)
    xdot = ca.SX.sym('xdot', NX)

    # ── Symbolic inputs ────────────────────────────────────────
    f1 = ca.SX.sym('f1')   # front motor thrust [N]
    f2 = ca.SX.sym('f2')   # right motor thrust [N]
    f3 = ca.SX.sym('f3')   # back  motor thrust [N]
    f4 = ca.SX.sym('f4')   # left  motor thrust [N]
    u  = ca.vertcat(f1, f2, f3, f4)

    # ── Thrust and torques from motor forces ───────────────────
    T_total = f1 + f2 + f3 + f4                        # total thrust [N]
    tau_x   = L * (f4 - f2)                            # roll  torque [N·m]
    tau_y   = L * (f3 - f1)                            # pitch torque [N·m]
    tau_z   = c_tau * (-f1 + f2 - f3 + f4)             # yaw   torque [N·m]

    # ── Subsystem 1: Translational dynamics (world frame) ──────
    #
    # Rotation matrix R(q): body → world.
    # Thrust in body frame: [0, 0, T_total]
    # Force in world frame: R @ [0, 0, T_total]
    #
    # Only the third column of R(q) is needed:
    #   R[:,2] = [2(qx*qz + qw*qy),
    #             2(qy*qz - qw*qx),
    #             1 - 2(qx² + qy²)]
    #
    dvx = T_total / m * 2 * (qx*qz + qw*qy)
    dvy = T_total / m * 2 * (qy*qz - qw*qx)
    dvz = T_total / m * (1 - 2*(qx**2 + qy**2)) - g

    # ── Subsystem 2: Quaternion kinematics ─────────────────────
    #
    # q_dot = 0.5 * q ⊗ [0, p, q, r]
    #
    # Expanded (scalar-first convention):
    #   dqw = 0.5 * (-qx*p - qy*q - qz*r)
    #   dqx = 0.5 * ( qw*p + qy*r - qz*q)
    #   dqy = 0.5 * ( qw*q - qx*r + qz*p)
    #   dqz = 0.5 * ( qw*r + qx*q - qy*p)
    #
    # No singularity — valid for ALL orientations.
    #
    dqw = 0.5 * (-qx*p - qy*q - qz*r)
    dqx = 0.5 * ( qw*p + qy*r - qz*q)
    dqy = 0.5 * ( qw*q - qx*r + qz*p)
    dqz = 0.5 * ( qw*r + qx*q - qy*p)

    # ── Subsystem 3: Rotational dynamics (Euler's equations) ───
    #
    # Body-frame angular momentum equations with gyroscopic coupling:
    #   I·dω = τ - ω × (I·ω)
    #
    # These are independent of attitude representation — same as
    # Euler angle version.
    #
    dp_dt = (Iyy - Izz) / Ixx * q * r + tau_x / Ixx
    dq_dt = (Izz - Ixx) / Iyy * p * r + tau_y / Iyy
    dr_dt = (Ixx - Iyy) / Izz * p * q + tau_z / Izz

    # ── Assemble explicit ODE: xdot = f(x, u) ─────────────────
    f_expl = ca.vertcat(
        vx, vy, vz,                   # position kinematics
        dvx, dvy, dvz,                # translational dynamics
        dqw, dqx, dqy, dqz,           # quaternion kinematics
        dp_dt, dq_dt, dr_dt           # rotational dynamics
    )

    # ── Build acados model ─────────────────────────────────────
    model             = AcadosModel()
    model.name        = 'quadrotor_3d'
    model.x           = x
    model.u           = u
    model.xdot        = xdot
    model.f_expl_expr = f_expl

    return model


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
    # dvx = T/m * 2*(qx*qz + qw*qy)  →  d(dvx)/d(qy) at hover = 2*T/m*qw = 2g
    # dvy = T/m * 2*(qy*qz - qw*qx)  →  d(dvy)/d(qx) at hover = -2*T/m*qw = -2g
    A_c[3, 8] = 2 * g     # qy → dvx  (tilt about y → accelerate in x)
    A_c[4, 7] = -2 * g    # qx → dvy  (tilt about x → accelerate in -y)

    # Quaternion kinematics: angular velocity → quaternion derivative
    # At hover (qw=1, qx=qy=qz=0):
    #   dqx = 0.5*qw*p = 0.5*p  →  d(dqx)/d(p) = 0.5
    #   dqy = 0.5*qw*q = 0.5*q  →  d(dqy)/d(q) = 0.5
    #   dqz = 0.5*qw*r = 0.5*r  →  d(dqz)/d(r) = 0.5
    A_c[7, 10] = 0.5     # p → dqx
    A_c[8, 11] = 0.5     # q → dqy
    A_c[9, 12] = 0.5     # r → dqz

    # Rotational dynamics: no state coupling at hover (p=q=r=0 kills cross terms)

    B_c = np.zeros((NX, NU))

    # Translational: all motors contribute equally to vertical acceleration
    B_c[5, :] = 1.0 / m    # fi → dvz

    # Rotational: motor forces → angular accelerations
    # Roll:  tau_x = L*(f4-f2) → dp = tau_x/Ixx
    B_c[10, 1] = -L / Ixx    # f2 → dp
    B_c[10, 3] =  L / Ixx    # f4 → dp

    # Pitch: tau_y = L*(f3-f1) → dq = tau_y/Iyy
    B_c[11, 0] = -L / Iyy    # f1 → dq
    B_c[11, 2] =  L / Iyy    # f3 → dq

    # Yaw:   tau_z = c_tau*(-f1+f2-f3+f4) → dr = tau_z/Izz
    B_c[12, 0] = -c_tau / Izz
    B_c[12, 1] =  c_tau / Izz
    B_c[12, 2] = -c_tau / Izz
    B_c[12, 3] =  c_tau / Izz

    return A_c, B_c


# ─────────────────────────────────────────────────────────────────
# AcadosSim Plant Simulator
# ─────────────────────────────────────────────────────────────────
def create_plant_simulator(T_horizon: float = 1.0,
                           N: int = 20) -> AcadosSimSolver:
    """
    Create an AcadosSimSolver for plant simulation.

    Uses the same model and integrator as the NMPC — no model mismatch.
    One call advances the plant by one sample time Ts = T_horizon/N.

    Important: call normalize_quaternion(x) after each simulation step
    to prevent quaternion norm drift.
    """
    model = create_model()
    sim   = AcadosSim()
    sim.model = model
    sim.solver_options.T = T_horizon / N
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    return AcadosSimSolver(sim)


# ─────────────────────────────────────────────────────────────────
# Numerical ODE + RK4  (backup for model mismatch testing)
# ─────────────────────────────────────────────────────────────────
def f_ode(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Same dynamics as create_model() but in numerical form.
    Used for future model mismatch testing (different parameters).
    """
    px, py, pz, vx, vy, vz, qw, qx, qy, qz, p, q, r = x
    f1, f2, f3, f4 = u

    T_total = f1 + f2 + f3 + f4
    tau_x   = L * (f4 - f2)
    tau_y   = L * (f3 - f1)
    tau_z   = c_tau * (-f1 + f2 - f3 + f4)

    # translational dynamics (world frame)
    dvx = T_total / m * 2 * (qx*qz + qw*qy)
    dvy = T_total / m * 2 * (qy*qz - qw*qx)
    dvz = T_total / m * (1 - 2*(qx**2 + qy**2)) - g

    # quaternion kinematics
    dqw = 0.5 * (-qx*p - qy*q - qz*r)
    dqx = 0.5 * ( qw*p + qy*r - qz*q)
    dqy = 0.5 * ( qw*q - qx*r + qz*p)
    dqz = 0.5 * ( qw*r + qx*q - qy*p)

    # rotational dynamics (body frame)
    dp = (Iyy - Izz) / Ixx * q*r + tau_x / Ixx
    dq = (Izz - Ixx) / Iyy * p*r + tau_y / Iyy
    dr = (Ixx - Iyy) / Izz * p*q + tau_z / Izz

    return np.array([vx, vy, vz, dvx, dvy, dvz,
                     dqw, dqx, dqy, dqz, dp, dq, dr])


def rk4_step(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """
    One RK4 integration step — backup plant simulator.
    Remember to call normalize_quaternion() after this.
    """
    k1 = f_ode(x,            u)
    k2 = f_ode(x + dt/2*k1,  u)
    k3 = f_ode(x + dt/2*k2,  u)
    k4 = f_ode(x + dt   *k3, u)
    x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return normalize_quaternion(x_next)