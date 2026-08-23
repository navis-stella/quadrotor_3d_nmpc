"""
quadrotor_3d_model.py — Full 3D Quadrotor Model
==================================================
Contains:
  - Physical parameters
  - CasADi symbolic model           (used by acados NMPC solver)
  - Hover linearization              (used for DARE terminal cost)
  - AcadosSimSolver plant creator    (used as plant in simulation)
  - Numerical ODE + RK4 backup       (for future model mismatch testing)

State (12):
    x = [px, py, pz, vx, vy, vz, φ, θ, ψ, p, q, r]

    Position:         px, py, pz     [m]      world frame
    Linear velocity:  vx, vy, vz     [m/s]    world frame
    Euler angles:     φ, θ, ψ        [rad]    roll, pitch, yaw (ZYX convention)
    Angular velocity: p, q, r        [rad/s]  body frame

Input (4):
    u = [f1, f2, f3, f4]   individual motor thrusts [N]

Motor layout (+ configuration, top view, z-up):

          M1 (front, CW)
           |
    M4 ----●---- M2 (right, CCW)
   (left,  |     CCW→CW: check your setup)
    CCW)   |
          M3 (back, CW → CCW: check)

    M1 at [ L,  0, 0]   front     spin CW  → reactive τ_z < 0
    M2 at [ 0, -L, 0]   right     spin CCW → reactive τ_z > 0
    M3 at [-L,  0, 0]   back      spin CW  → reactive τ_z < 0
    M4 at [ 0,  L, 0]   left      spin CCW → reactive τ_z > 0

Coordinate convention (z-up):
    x → forward,  y → left,  z → up
    φ (roll):  rotation about x — positive tilts right side down
    θ (pitch): rotation about y — positive tilts nose down
    ψ (yaw):   rotation about z — positive turns nose left

Extending this file in future stages:
    Stage 2  → get_hover_linearization() for DARE terminal cost
    Stage 3  → create_augmented_model() for offset-free NMPC
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
NX = 12
NU = 4


# ─────────────────────────────────────────────────────────────────
# CasADi Symbolic Model  (used by acados NMPC)
# ─────────────────────────────────────────────────────────────────
def create_model() -> AcadosModel:
    """
    Full 3D quadrotor nonlinear dynamics in CasADi symbolic form.

    Three subsystems:
        1. Translational dynamics   (world frame, driven by thrust + gravity)
        2. Euler angle kinematics   (body rates → Euler angle rates)
        3. Rotational dynamics      (body frame, Euler's equations)

    Note: Euler angle kinematics has gimbal lock at θ = ±π/2.
    For hover stabilization (θ ≈ 0) this is acceptable.
    For aggressive maneuvers, quaternion formulation would be needed.
    """
    # ── Symbolic states ────────────────────────────────────────
    px  = ca.SX.sym('px')     # position x [m]
    py  = ca.SX.sym('py')     # position y [m]
    pz  = ca.SX.sym('pz')     # position z [m]
    vx  = ca.SX.sym('vx')     # velocity x [m/s]
    vy  = ca.SX.sym('vy')     # velocity y [m/s]
    vz  = ca.SX.sym('vz')     # velocity z [m/s]
    phi   = ca.SX.sym('phi')    # roll  [rad]
    theta = ca.SX.sym('theta')  # pitch [rad]
    psi   = ca.SX.sym('psi')    # yaw   [rad]
    p   = ca.SX.sym('p')      # roll rate  [rad/s]  body frame
    q   = ca.SX.sym('q')      # pitch rate [rad/s]  body frame
    r   = ca.SX.sym('r')      # yaw rate   [rad/s]  body frame

    x    = ca.vertcat(px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r)
    xdot = ca.SX.sym('xdot', NX)

    # ── Symbolic inputs ────────────────────────────────────────
    f1 = ca.SX.sym('f1')   # front motor thrust [N]
    f2 = ca.SX.sym('f2')   # right motor thrust [N]
    f3 = ca.SX.sym('f3')   # back  motor thrust [N]
    f4 = ca.SX.sym('f4')   # left  motor thrust [N]
    u  = ca.vertcat(f1, f2, f3, f4)

    # ── Thrust and torques from motor forces ───────────────────
    T_total = f1 + f2 + f3 + f4                      # total thrust [N]
    tau_x   = L * (f4 - f2)                           # roll  torque [N·m]
    tau_y   = L * (f3 - f1)                            # pitch torque [N·m]
    tau_z   = c_tau * (-f1 + f2 - f3 + f4)             # yaw   torque [N·m]

    # ── Subsystem 1: Translational dynamics (world frame) ──────
    #
    # Body-to-world rotation matrix R(φ,θ,ψ) — ZYX Euler convention
    # Thrust vector in body frame: [0, 0, T_total]
    # Force in world frame: R @ [0, 0, T_total]
    # Only the third column of R matters (multiplies T_total):
    #
    #   Fx_world = T * (cos(ψ)sin(θ)cos(φ) + sin(ψ)sin(φ))
    #   Fy_world = T * (sin(ψ)sin(θ)cos(φ) - cos(ψ)sin(φ))
    #   Fz_world = T * cos(θ)cos(φ)
    #
    dvx = T_total / m * (ca.cos(psi)*ca.sin(theta)*ca.cos(phi) + ca.sin(psi)*ca.sin(phi))
    dvy = T_total / m * (ca.sin(psi)*ca.sin(theta)*ca.cos(phi) - ca.cos(psi)*ca.sin(phi))
    dvz = T_total / m * ca.cos(theta)*ca.cos(phi) - g

    # ── Subsystem 2: Euler angle kinematics ────────────────────
    #
    # Transforms body angular rates [p,q,r] → Euler angle rates [dφ,dθ,dψ]
    # Singularity at θ = ±π/2 (gimbal lock) — safe near hover
    #
    dphi   = p + ca.sin(phi)*ca.tan(theta)*q + ca.cos(phi)*ca.tan(theta)*r
    dtheta = ca.cos(phi)*q - ca.sin(phi)*r
    dpsi   = ca.sin(phi)/ca.cos(theta)*q + ca.cos(phi)/ca.cos(theta)*r

    # ── Subsystem 3: Rotational dynamics (Euler's equations) ───
    #
    # Body-frame angular momentum equations with gyroscopic coupling:
    #   I·dω = τ - ω × (I·ω)
    #
    dp_dt = (Iyy - Izz) / Ixx * q * r + tau_x / Ixx
    dq_dt = (Izz - Ixx) / Iyy * p * r + tau_y / Iyy
    dr_dt = (Ixx - Iyy) / Izz * p * q + tau_z / Izz

    # ── Assemble explicit ODE: xdot = f(x, u) ─────────────────
    f_expl = ca.vertcat(
        vx, vy, vz,               # position kinematics
        dvx, dvy, dvz,            # translational dynamics
        dphi, dtheta, dpsi,       # Euler angle kinematics
        dp_dt, dq_dt, dr_dt       # rotational dynamics
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

    Equilibrium: φ=θ=ψ=0, p=q=r=0, f1=f2=f3=f4=mg/4

    Key couplings at hover:
        pitch θ → acceleration in x   (A[3,7] = +g)
        roll  φ → acceleration in y   (A[4,6] = -g)

    This is physically intuitive:
        tilt forward (θ>0) → thrust has x-component → accelerate in x
        tilt right   (φ>0) → thrust has -y-component → accelerate in -y

    Returns:
        A_c:  (12×12) continuous-time state Jacobian at hover
        B_c:  (12×4)  continuous-time input Jacobian at hover
    """
    A_c = np.zeros((NX, NX))

    # Position kinematics: velocity → position
    A_c[0, 3] = 1.0    # vx → dpx
    A_c[1, 4] = 1.0    # vy → dpy
    A_c[2, 5] = 1.0    # vz → dpz

    # Translational dynamics: attitude → acceleration
    A_c[3, 7] = g      # θ (pitch) → dvx   (tilt forward → accelerate in x)
    A_c[4, 6] = -g     # φ (roll)  → dvy   (tilt right  → accelerate in -y)

    # Euler angle kinematics: body rates → Euler rates (at hover: identity)
    A_c[6, 9]  = 1.0   # p → dφ
    A_c[7, 10] = 1.0   # q → dθ
    A_c[8, 11] = 1.0   # r → dψ

    # Rotational dynamics: no state coupling at hover (p=q=r=0 kills cross terms)

    B_c = np.zeros((NX, NU))

    # Translational: all motors contribute equally to vertical acceleration
    B_c[5, :] = 1.0 / m    # fi → dvz = 1/m for all motors

    # Rotational: motor forces → angular accelerations via torque allocation
    # Roll:  τ_x = L*(f4-f2) → dp = τ_x/Ixx
    B_c[9, 1] = -L / Ixx    # f2 → dp (right motor, negative roll torque)
    B_c[9, 3] =  L / Ixx    # f4 → dp (left motor, positive roll torque)

    # Pitch: τ_y = L*(f3-f1) → dq = τ_y/Iyy
    B_c[10, 0] = -L / Iyy   # f1 → dq (front motor, negative pitch torque)
    B_c[10, 2] =  L / Iyy   # f3 → dq (back motor, positive pitch torque)

    # Yaw:   τ_z = c_τ*(-f1+f2-f3+f4) → dr = τ_z/Izz
    B_c[11, 0] = -c_tau / Izz   # f1 CW
    B_c[11, 1] =  c_tau / Izz   # f2 CCW
    B_c[11, 2] = -c_tau / Izz   # f3 CW
    B_c[11, 3] =  c_tau / Izz   # f4 CCW

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

    For future model mismatch testing (Stage 3), use rk4_step()
    with modified parameters instead.
    """
    model = create_model()
    sim   = AcadosSim()
    sim.model = model
    sim.solver_options.T = T_horizon / N     # one sample step
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4        # ERK4
    return AcadosSimSolver(sim)


# ─────────────────────────────────────────────────────────────────
# Numerical ODE + RK4  (backup for model mismatch testing)
# ─────────────────────────────────────────────────────────────────
def f_ode(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Same dynamics as create_model() but in numerical form.
    Used for future model mismatch testing (different parameters).
    """
    px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r = x
    f1, f2, f3, f4 = u

    T_total = f1 + f2 + f3 + f4
    tau_x   = L * (f4 - f2)
    tau_y   = L * (f3 - f1)
    tau_z   = c_tau * (-f1 + f2 - f3 + f4)

    # translational
    dvx = T_total/m * (np.cos(psi)*np.sin(theta)*np.cos(phi) + np.sin(psi)*np.sin(phi))
    dvy = T_total/m * (np.sin(psi)*np.sin(theta)*np.cos(phi) - np.cos(psi)*np.sin(phi))
    dvz = T_total/m * np.cos(theta)*np.cos(phi) - g

    # Euler kinematics
    dphi   = p + np.sin(phi)*np.tan(theta)*q + np.cos(phi)*np.tan(theta)*r
    dtheta = np.cos(phi)*q - np.sin(phi)*r
    dpsi   = np.sin(phi)/np.cos(theta)*q + np.cos(phi)/np.cos(theta)*r

    # rotational
    dp = (Iyy - Izz)/Ixx * q*r + tau_x/Ixx
    dq = (Izz - Ixx)/Iyy * p*r + tau_y/Iyy
    dr = (Ixx - Iyy)/Izz * p*q + tau_z/Izz

    return np.array([vx, vy, vz, dvx, dvy, dvz,
                     dphi, dtheta, dpsi, dp, dq, dr])


def rk4_step(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """One RK4 integration step — backup plant simulator."""
    k1 = f_ode(x,            u)
    k2 = f_ode(x + dt/2*k1,  u)
    k3 = f_ode(x + dt/2*k2,  u)
    k4 = f_ode(x + dt   *k3, u)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
