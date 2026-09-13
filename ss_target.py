"""
ss_target.py — Steady-State Target Calculator for Offset-Free NMPC
====================================================================
Computes the physically consistent equilibrium (x_s, u_s) given:
    - x_ref:  desired reference state (what the user wants)
    - d_hat:  estimated disturbance (from EKF)

Solves:
    min_{x_s, u_s}  || p_s - p_ref ||²   (track desired position)
    s.t.  f(x_s, u_s, d̂) = 0              (equilibrium condition)
          u_min ≤ u_s ≤ u_max              (input limits)

For the quadrotor, the equilibrium conditions at hover-like states
(v=0, ω=0) reduce to:

    Translational:  T_s · R(q_s) · e₃ + d_f = m·g·e₃
    Rotational:     τ(u_s) + d_τ = 0

These can be solved analytically:
    1. Required force vector  → total thrust T_s and thrust direction
    2. Thrust direction       → equilibrium quaternion q_s (with desired yaw)
    3. Torque balance         → motor mixer inversion → u_s = [f1, f2, f3, f4]

Why this matters:
    Without the target calculator, the MPC tries to achieve level hover (q=[1,0,0,0])
    AND the desired position simultaneously. Under a horizontal wind, this is
    physically impossible — the drone must tilt to hold position. The target
    calculator resolves this conflict by telling the MPC what the achievable
    equilibrium looks like.
"""

import numpy as np
from quadrotor_3d_model import m, g, L, c_tau, Ixx, Iyy, Izz, f_hover, NX, NU, ND


# ─────────────────────────────────────────────────────────────────
# Motor Mixer Matrix (thrust & torques → motor forces)
# ─────────────────────────────────────────────────────────────────
#
#   [T_total]     [  1     1     1     1  ] [f1]
#   [tau_x  ]  =  [  0    -L     0     L  ] [f2]
#   [tau_y  ]     [ -L     0     L     0  ] [f3]
#   [tau_z  ]     [-c_tau  c_tau -c_tau c_tau] [f4]
#
#   u = M_inv @ [T, tau_x, tau_y, tau_z]
#
_M = np.array([
    [ 1.0,    1.0,     1.0,    1.0   ],
    [ 0.0,   -L,       0.0,    L     ],
    [-L,      0.0,     L,      0.0   ],
    [-c_tau,  c_tau,  -c_tau,  c_tau  ],
])
_M_inv = np.linalg.inv(_M)


# ─────────────────────────────────────────────────────────────────
# Rotation Matrix → Quaternion (scalar-first)
# ─────────────────────────────────────────────────────────────────
def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3×3 rotation matrix to a unit quaternion [qw, qx, qy, qz].
    Uses Shepperd's method for numerical robustness.
    Enforces qw > 0 convention.
    """
    tr = np.trace(R)

    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz])
    q /= np.linalg.norm(q)

    # enforce qw > 0
    if q[0] < 0:
        q *= -1

    return q


# ─────────────────────────────────────────────────────────────────
# Quaternion → Yaw extraction
# ─────────────────────────────────────────────────────────────────
def _quat_to_yaw(q: np.ndarray) -> float:
    """Extract yaw angle from quaternion [qw, qx, qy, qz]."""
    qw, qx, qy, qz = q
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy**2 + qz**2)
    return np.arctan2(siny_cosp, cosy_cosp)


# ─────────────────────────────────────────────────────────────────
# Steady-State Target Calculator
# ─────────────────────────────────────────────────────────────────
def compute_ss_target(x_ref: np.ndarray,
                      d_hat: np.ndarray,
                      f_max: float = None) -> tuple:
    """
    Compute the physically consistent equilibrium (x_s, u_s) under d̂.

    The quadrotor equilibrium under constant disturbance is fully
    determined by the force and torque balance equations. The position
    target is achievable (p_s = p_ref), but attitude must tilt to
    generate the horizontal force that counters the wind.

    Analytical solution:

        1. Force balance (world frame):
           T_s · z_b = [0, 0, mg] - [d_fx, d_fy, d_fz]

           where z_b = R(q_s) · [0,0,1] is the body z-axis in world.
           This gives T_s (total thrust) and z_b (thrust direction).

        2. Quaternion from thrust direction + desired yaw:
           z_b → pitch and roll
           ψ_ref → yaw from the user's reference quaternion
           → full rotation matrix → quaternion q_s

        3. Torque balance (body frame, ω=0 → no gyroscopic terms):
           τ(u_s) = -[d_tx, d_ty, d_tz]
           Combined with T_s → 4×4 mixer inversion → u_s

    Args:
        x_ref:  (13,) desired reference state
        d_hat:  (6,)  estimated disturbance [d_fx, d_fy, d_fz, d_tx, d_ty, d_tz]
        f_max:  maximum motor thrust [N] (default: 3 × f_hover)

    Returns:
        x_s:  (13,) equilibrium state (p=p_ref, v=0, q=q_s, ω=0)
        u_s:  (4,)  equilibrium motor thrusts
    """
    if f_max is None:
        f_max = 3.0 * f_hover

    d_fx, d_fy, d_fz = d_hat[0], d_hat[1], d_hat[2]
    d_tx, d_ty, d_tz = d_hat[3], d_hat[4], d_hat[5]

    # ── Step 1: Force balance → thrust magnitude and direction ──
    #
    #   At equilibrium: T_s · z_b + d_f = m·g·e₃
    #   → T_s · z_b = [−d_fx, −d_fy, mg − d_fz]
    #
    F_req = np.array([-d_fx, -d_fy, m * g - d_fz])
    T_s   = np.linalg.norm(F_req)
    z_b   = F_req / T_s   # body z-axis in world frame (thrust direction)

    # ── Step 2: Construct rotation matrix from z_b and yaw ──────
    #
    #   R = [x_b | y_b | z_b]   (body axes as columns in world frame)
    #
    #   Given z_b (from force balance) and ψ (from x_ref):
    #     x_c = [cos(ψ), sin(ψ), 0]       candidate x direction in xy-plane
    #     y_b = z_b × x_c / ||...||        body y perpendicular to z_b
    #     x_b = y_b × z_b                  body x completes the frame
    #
    psi = _quat_to_yaw(x_ref[6:10])
    x_c = np.array([np.cos(psi), np.sin(psi), 0.0])

    y_b = np.cross(z_b, x_c)
    y_b_norm = np.linalg.norm(y_b)

    if y_b_norm < 1e-6:
        # z_b is nearly vertical → any yaw works, use standard basis
        y_b = np.array([-np.sin(psi), np.cos(psi), 0.0])
    else:
        y_b /= y_b_norm

    x_b = np.cross(y_b, z_b)
    x_b /= np.linalg.norm(x_b)   # should already be unit, but normalize

    R_s = np.column_stack([x_b, y_b, z_b])
    q_s = _rotmat_to_quat(R_s)

    # ── Step 3: Torque balance → motor forces via mixer ─────────
    #
    #   At equilibrium (ω = 0, no gyroscopic terms):
    #     τ(u_s) + d_τ = 0  →  τ(u_s) = [-d_tx, -d_ty, -d_tz]
    #
    #   Combined with total thrust:
    #     [T_s, -d_tx, -d_ty, -d_tz] = M @ u_s
    #     u_s = M_inv @ [T_s, -d_tx, -d_ty, -d_tz]
    #
    wrench_s = np.array([T_s, -d_tx, -d_ty, -d_tz])
    u_s = _M_inv @ wrench_s

    # ── Step 4: Clip motor forces to physical limits ────────────
    u_s_clipped = np.clip(u_s, 0.0, f_max)
    if not np.allclose(u_s, u_s_clipped, atol=1e-6):
        print(f'  [ss_target] WARNING: motor limits active!')
        print(f'    u_s raw    = {u_s}')
        print(f'    u_s clipped = {u_s_clipped}')
    u_s = u_s_clipped

    # ── Step 5: Assemble equilibrium state ──────────────────────
    #
    #   Position:  p_s = p_ref       (achievable — position is fully actuated)
    #   Velocity:  v_s = 0           (equilibrium)
    #   Attitude:  q_s from step 2   (tilted to balance forces)
    #   Ang. rate: ω_s = 0           (equilibrium)
    #
    x_s = np.zeros(NX)
    x_s[0:3]  = x_ref[0:3]     # position = reference
    x_s[3:6]  = 0.0             # velocity = 0
    x_s[6:10] = q_s             # quaternion = equilibrium tilt
    x_s[10:13] = 0.0            # angular rates = 0

    return x_s, u_s


# ─────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────
def print_ss_target(x_ref, x_s, u_s, d_hat):
    """Print a comparison of reference vs computed equilibrium."""
    # Extract Euler angles
    def _quat_to_euler_deg(q):
        qw, qx, qy, qz = q
        roll  = np.degrees(np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2)))
        sinp  = np.clip(2*(qw*qy - qz*qx), -1, 1)
        pitch = np.degrees(np.arcsin(sinp))
        yaw   = np.degrees(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2)))
        return roll, pitch, yaw

    r_ref = _quat_to_euler_deg(x_ref[6:10])
    r_s   = _quat_to_euler_deg(x_s[6:10])

    print('\n─── Steady-State Target Calculator ────────────────')
    print(f'  Disturbance d̂ = {d_hat}')
    print(f'')
    print(f'  {"":20s}  {"Reference":>12s}  {"Equilibrium":>12s}  {"Δ":>10s}')
    print(f'  {"position x [m]":20s}  {x_ref[0]:12.4f}  {x_s[0]:12.4f}  {x_s[0]-x_ref[0]:10.4f}')
    print(f'  {"position y [m]":20s}  {x_ref[1]:12.4f}  {x_s[1]:12.4f}  {x_s[1]-x_ref[1]:10.4f}')
    print(f'  {"position z [m]":20s}  {x_ref[2]:12.4f}  {x_s[2]:12.4f}  {x_s[2]-x_ref[2]:10.4f}')
    print(f'  {"roll  [°]":20s}  {r_ref[0]:12.4f}  {r_s[0]:12.4f}  {r_s[0]-r_ref[0]:10.4f}')
    print(f'  {"pitch [°]":20s}  {r_ref[1]:12.4f}  {r_s[1]:12.4f}  {r_s[1]-r_ref[1]:10.4f}')
    print(f'  {"yaw   [°]":20s}  {r_ref[2]:12.4f}  {r_s[2]:12.4f}  {r_s[2]-r_ref[2]:10.4f}')
    print(f'')
    print(f'  {"Motor forces [N]":20s}  {"hover":>12s}  {"equilibrium":>12s}')
    for i in range(NU):
        print(f'  {"f"+str(i+1):20s}  {f_hover:12.4f}  {u_s[i]:12.4f}')
    print(f'  {"T_total":20s}  {4*f_hover:12.4f}  {u_s.sum():12.4f}')
    print('───────────────────────────────────────────────────')
