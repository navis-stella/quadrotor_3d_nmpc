"""
ekf_aug.py — Extended Kalman Filter for Offset-Free NMPC (Stage 3)
===================================================================
Estimates the augmented state z = [x(13); d(6)] ∈ R^19 where:
    x = quadrotor state (fully measured)
    d = [d_fx, d_fy, d_fz, d_tx, d_ty, d_tz] constant disturbance

Measurement model:
    y = H @ z = x      (full state access, H = [I_13 | 0_13×6])

Since x is directly measured, the EKF innovation primarily corrects
the disturbance estimate d̂ via the cross-covariance P_xd.
This is the mechanism that gives offset-free MPC its integral action.

Algorithm (continuous-discrete EKF):
    Predict:  integrate z_dot = f_aug(z, u) with RK4 over Ts
              propagate P via F_d = I + F_c·Ts  (first-order discretization)
    Update:   K = P H^T (H P H^T + R)^{-1}
              z_hat += K (y - H z_hat)
              P = (I - K H) P (I - K H)^T + K R K^T   (Joseph form)

Tuning guide:
    Q_d (ND×ND):  disturbance process noise — controls adaptation speed
        Large Q_d  → fast adaptation, noisy d̂ estimate
        Small Q_d  → slow adaptation, smooth d̂ estimate
        Typical: Q_d = diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
                 (forces change faster than torques in practice)

    Q_x (NX×NX):  state process noise — covers model imperfections
        Typically small since the model is accurate
        Typical: Q_x = diag(small values)

    R (NX×NX):    measurement noise covariance
        Stage 3 assumes clean full-state access → small R
        Typical: R = diag([σ_pos², σ_vel², σ_quat², σ_omega²])
"""

import numpy as np
from quadrotor_3d_model import (
    NX, NU, ND, NZ,
    get_augmented_dynamics_casadi,
    normalize_quaternion,
)


class AugEKF:
    """
    Extended Kalman Filter on the augmented quadrotor system.

    State: z = [x(13); d(6)] ∈ R^19
    Measurement: y = x ∈ R^13 (full state access)

    Usage in closed-loop:
        ekf = AugEKF(Ts=0.05, Q_x=..., Q_d=..., R=...)

        for each MPC step:
            # 1. Get measurement
            y = x_measured

            # 2. EKF update (correct d̂ using measurement)
            ekf.update(y)

            # 3. Extract estimates for MPC
            x_hat = ekf.get_state()       # (13,)
            d_hat = ekf.get_disturbance() # (6,)

            # 4. Solve MPC with d_hat as parameter
            ...

            # 5. EKF predict (propagate z_hat, P forward using applied u)
            ekf.predict(u_applied)
    """

    def __init__(self, Ts: float,
                 Q_x: np.ndarray, Q_d: np.ndarray, R: np.ndarray,
                 x0: np.ndarray = None, d0: np.ndarray = None,
                 P0: np.ndarray = None):
        """
        Initialize the AugEKF.

        Args:
            Ts:   sample time [s]
            Q_x:  (NX,) or (NX,NX) state process noise
            Q_d:  (ND,) or (ND,ND) disturbance process noise
            R:    (NX,) or (NX,NX) measurement noise covariance
            x0:   (NX,) initial state estimate (default: hover)
            d0:   (ND,) initial disturbance estimate (default: zeros)
            P0:   (NZ,NZ) initial covariance (default: block-diagonal)
        """
        self.Ts = Ts

        # ── Process noise Q_aug = blkdiag(Q_x, Q_d) ───────────
        if Q_x.ndim == 1:
            Q_x = np.diag(Q_x)
        if Q_d.ndim == 1:
            Q_d = np.diag(Q_d)
        self.Q_aug = np.block([
            [Q_x,                np.zeros((NX, ND))],
            [np.zeros((ND, NX)), Q_d                ],
        ])

        # ── Measurement noise R ────────────────────────────────
        if R.ndim == 1:
            R = np.diag(R)
        self.R = R

        # ── Measurement matrix H = [I_13 | 0_13×6] ───────────
        self.H = np.hstack([np.eye(NX), np.zeros((NX, ND))])

        # ── Initial augmented state ────────────────────────────
        if x0 is None:
            x0 = np.zeros(NX)
            x0[6] = 1.0   # qw = 1 (hover quaternion)
        if d0 is None:
            d0 = np.zeros(ND)

        self.z_hat = np.concatenate([x0, d0])

        # ── Initial covariance ─────────────────────────────────
        if P0 is not None:
            self.P = P0.copy()
        else:
            # Default: confident on state, uncertain on disturbance
            P_x = 0.01 * np.eye(NX)        # small — state is measured
            P_d = 1.0  * np.eye(ND)         # large — d is unknown
            self.P = np.block([
                [P_x,                np.zeros((NX, ND))],
                [np.zeros((ND, NX)), P_d                ],
            ])

        # ── CasADi dynamics and Jacobian ───────────────────────
        self.f_aug, self.F_func = get_augmented_dynamics_casadi()

        # ── RK4 integration buffers ────────────────────────────
        # (avoid repeated allocation)
        self._k1 = np.zeros(NZ)
        self._k2 = np.zeros(NZ)
        self._k3 = np.zeros(NZ)
        self._k4 = np.zeros(NZ)

    # ─────────────────────────────────────────────────────────────
    # RK4 integration of augmented dynamics
    # ─────────────────────────────────────────────────────────────
    def _rk4_predict(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Integrate z_dot = f_aug(z, u) over one sample time Ts using RK4.

        The disturbance states d have d_dot=0, so RK4 leaves them unchanged.
        The RK4 is for the nonlinear plant states x.
        """
        dt = self.Ts
        f = self.f_aug

        k1 = np.array(f(z,              u)).flatten()
        k2 = np.array(f(z + dt/2 * k1,  u)).flatten()
        k3 = np.array(f(z + dt/2 * k2,  u)).flatten()
        k4 = np.array(f(z + dt   * k3,  u)).flatten()

        z_next = z + dt / 6.0 * (k1 + 2*k2 + 2*k3 + k4)
        return z_next

    # ─────────────────────────────────────────────────────────────
    # Prediction step
    # ─────────────────────────────────────────────────────────────
    def predict(self, u: np.ndarray):
        """
        EKF prediction step: propagate state and covariance forward.

        Called AFTER MPC solves (using the applied control input).

        State:      z_hat(k+1|k) = RK4_integrate(f_aug, z_hat(k|k), u(k))
        Covariance: P(k+1|k) = F_d P(k|k) F_d^T + Q_aug

        where F_d ≈ I + F_c·Ts  (first-order discretization of Jacobian)
        """
        # ── State prediction (RK4) ─────────────────────────────
        self.z_hat = self._rk4_predict(self.z_hat, u)

        # Normalize quaternion in the state part
        self.z_hat[:NX] = normalize_quaternion(self.z_hat[:NX].copy())

        # ── Covariance prediction ──────────────────────────────
        # Continuous Jacobian at current estimate
        F_c = np.array(self.F_func(self.z_hat, u))

        # First-order discretization: F_d ≈ I + F_c·Ts
        F_d = np.eye(NZ) + F_c * self.Ts

        self.P = F_d @ self.P @ F_d.T + self.Q_aug * self.Ts

        # Symmetrize (numerical hygiene)
        self.P = 0.5 * (self.P + self.P.T)

    # ─────────────────────────────────────────────────────────────
    # Update (correction) step
    # ─────────────────────────────────────────────────────────────
    def update(self, y: np.ndarray):
        """
        EKF update step: correct state estimate using measurement.

        Called BEFORE MPC solves (incorporating the latest measurement).

        Measurement model: y = H @ z = x  (full state access)

        Innovation:  e = y - H @ z_hat
        Kalman gain: K = P H^T (H P H^T + R)^{-1}
        Correction:  z_hat += K @ e
        Covariance:  Joseph form for numerical stability

        The key mechanism: even though x is measured directly,
        the Kalman gain K has nonzero entries for the d-states
        (rows 13-18) because P_xd ≠ 0. This is how d̂ adapts.
        """
        H = self.H
        P = self.P
        R = self.R

        # ── Innovation ─────────────────────────────────────────
        y_hat = H @ self.z_hat          # predicted measurement (= x̂)
        e = y - y_hat                    # innovation (13,)

        # Handle quaternion sign ambiguity in innovation:
        # If qw_meas and qw_hat have opposite signs, flip the measurement
        # quaternion to avoid a large spurious innovation
        if y[6] * self.z_hat[6] < 0:
            y_flipped = y.copy()
            y_flipped[6:10] *= -1
            e = y_flipped - y_hat

        # ── Kalman gain ────────────────────────────────────────
        S = H @ P @ H.T + R             # innovation covariance (13×13)
        K = P @ H.T @ np.linalg.inv(S)  # Kalman gain (19×13)

        # ── State correction ───────────────────────────────────
        self.z_hat = self.z_hat + K @ e

        # Normalize quaternion after correction
        self.z_hat[:NX] = normalize_quaternion(self.z_hat[:NX].copy())

        # ── Covariance update (Joseph form) ────────────────────
        # P = (I-KH) P (I-KH)^T + K R K^T
        # More numerically stable than P = (I-KH)P
        I_KH = np.eye(NZ) - K @ H
        self.P = I_KH @ P @ I_KH.T + K @ R @ K.T

        # Symmetrize
        self.P = 0.5 * (self.P + self.P.T)

    # ─────────────────────────────────────────────────────────────
    # Accessors
    # ─────────────────────────────────────────────────────────────
    def get_state(self) -> np.ndarray:
        """Return the estimated plant state x̂ (13,)."""
        return self.z_hat[:NX].copy()

    def get_disturbance(self) -> np.ndarray:
        """Return the estimated disturbance d̂ (6,)."""
        return self.z_hat[NX:].copy()

    def get_augmented_state(self) -> np.ndarray:
        """Return the full augmented state ẑ = [x̂; d̂] (19,)."""
        return self.z_hat.copy()

    def get_covariance(self) -> np.ndarray:
        """Return the augmented covariance P (19×19)."""
        return self.P.copy()

    def get_disturbance_covariance(self) -> np.ndarray:
        """Return the disturbance sub-block of P (6×6)."""
        return self.P[NX:, NX:].copy()


# ─────────────────────────────────────────────────────────────────
# Default tuning (good starting point for the quadrotor)
# ─────────────────────────────────────────────────────────────────
def get_default_ekf_tuning():
    """
    Return default Q_x, Q_d, R matrices for the quadrotor EKF.

    Design rationale:
        Q_x:  small — the model is accurate, process noise just
               prevents covariance collapse
        Q_d:  moderate — allows d̂ to track step disturbances
               within ~1-2 seconds (roughly 20-40 MPC steps at 50ms)
               Forces (d_f) change faster in practice than torques (d_τ),
               so Q_d_force > Q_d_torque
        R:    small — full state access with good sensors
               Position and orientation are more precisely measured
               than velocities and angular rates
    """
    # State process noise Q_x (13,)
    Q_x = np.array([
        0.001, 0.001, 0.001,           # position [m²/s]
        0.01,  0.01,  0.01,            # velocity [m²/s³]
        0.001, 0.001, 0.001, 0.001,    # quaternion [-]
        0.01,  0.01,  0.01,            # angular rates [rad²/s³]
    ])

    # Disturbance process noise Q_d (6,)
    # Controls adaptation speed:
    #   τ_adapt ≈ √(R / Q_d)  (rough rule of thumb)
    Q_d = np.array([
        0.1,  0.1,  0.1,              # force disturbance [N²/s]
        0.01, 0.01, 0.01,             # torque disturbance [N²m²/s]
    ])

    # Measurement noise R (13,)
    R = np.array([
        0.001, 0.001, 0.001,           # position [m²]
        0.01,  0.01,  0.01,            # velocity [m²/s²]
        0.001, 0.001, 0.001, 0.001,    # quaternion [-]
        0.01,  0.01,  0.01,            # angular rates [rad²/s²]
    ])

    return Q_x, Q_d, R
