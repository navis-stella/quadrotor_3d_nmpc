"""
state_est_ekf.py — Stage 4a Augmented Extended Kalman Filter
============================================================
EKF for the 3D quadrotor with augmented disturbance state.

    State:        z = [x(13); d(6)]  ∈ R^19  (augmented)
    Measurement:  y = [p; q; ω]      ∈ R^10  (position, quaternion, angular velocity)

Model sources (quadrotor_3d_model) — shared with the Stage 4b MHE:
    - Augmented dynamics  (get_augmented_dynamics), ḋ = 0
    - Measurement model   (get_measurement_function)

Estimator interface (identical for state_est_mhe.MovingHorizonEstimator):
    est = ExtendedKalmanFilter(Ts, z0, ...)
    z_hat_k = est.step(y_k, u_prev)     u_prev = u_{k-1}, None at k = 0
    → the closed loop in closed_loop_sim_config.py does not know which one it runs.

Tuning is continuous-time where it can be (Q), so changing the sample
time does not silently change how fast d̂ may move.
"""

import numpy as np

from quadrotor_3d_model import (
    get_augmented_dynamics,
    get_measurement_function,
    normalize_quaternion,
    NX, NY, NZ,
)


# ═════════════════════════════════════════════════════════════════
# Default Tuning
# ═════════════════════════════════════════════════════════════════
def default_ekf_Q(Ts: float) -> np.ndarray:
    """
    Process noise covariance per sample, Q_d = Q_c · Ts   (19×19).

    Q_c is a spectral density [unit²/s]: small on measured/well-modeled
    states, larger on the unmeasured velocity (model error accumulates) and
    on the disturbance channels (Q_c,d sets how fast d̂ is allowed to move).

    Q_c,d rationale (mocap-equivalent noise):
        Q_c,f = 0.2 N²/s    →  σ ≈ 0.45 N per √s of random-walk drift.
                               Gives ~1 s force-disturbance tracking transient;
                               the residual jitter on d̂_fx, d̂_fy under noise
                               is the observability signature discussed in
                               docs/stage4_theory.md §2. Lower it for smoother
                               estimates at the cost of slower adaptation to a
                               step change in disturbance.
        Q_c,τ = 2e-3 N²m²/s →  torque disturbances are well observable
                               through ω, so this can stay modest without
                               hurting responsiveness.
    (Identical to the original per-step tuning at 20 Hz: Q_d = Q_c · 0.05.)
    """
    Q_c = np.zeros((NZ, NZ))
    Q_c[0:3,   0:3]   = 2e-5 * np.eye(3)     # pos    (measured)
    Q_c[3:6,   3:6]   = 2e-3 * np.eye(3)     # vel    (unmeasured, model error)
    Q_c[6:10,  6:10]  = 2e-5 * np.eye(4)     # quat   (measured)
    Q_c[10:13, 10:13] = 2e-5 * np.eye(3)     # ω      (measured)
    Q_c[13:16, 13:16] = 2e-1 * np.eye(3)     # d_force  — N
    Q_c[16:19, 16:19] = 2e-3 * np.eye(3)     # d_torque — N·m
    return Q_c * Ts


def default_ekf_R(noise_std: dict = None) -> np.ndarray:
    """
    Measurement noise covariance R (10×10), matched to the sensor.

        noise_std = sensor σ dict (sensor_simulator.MOCAP_NOISE_STD)
                    → R = diag(σ²) per channel — the Kalman gain then
                      reflects the actual signal-to-noise ratio
        noise_std = None  (noise-free sensor, open-loop checks)
                    → R = 1e-6 · I — small regularization so S = HPHᵀ + R
                      stays invertible for exact measurements
    """
    if noise_std is None:
        return 1e-6 * np.eye(NY)

    R = np.zeros((NY, NY))
    R[0:3, 0:3]   = noise_std['pos']**2   * np.eye(3)    # pos    [m]
    R[3:7, 3:7]   = noise_std['quat']**2  * np.eye(4)    # quat   [-]
    R[7:10, 7:10] = noise_std['omega']**2 * np.eye(3)    # omega  [rad/s]
    return R


def default_ekf_P0() -> np.ndarray:
    """
    Initial covariance P0 (19×19).

    Measured states start with small P (will snap in one update anyway).
    Unmeasured states (v, d) start with large P — we don't know them.
    """
    P0 = np.zeros((NZ, NZ))
    P0[0:3,   0:3]   = 1e-2 * np.eye(3)     # pos:    ±0.1 m   (measured)
    P0[3:6,   3:6]   = 1.0  * np.eye(3)     # vel:    ±1 m/s  (unmeasured)
    P0[6:10,  6:10]  = 1e-2 * np.eye(4)     # quat:   small   (measured)
    P0[10:13, 10:13] = 1e-2 * np.eye(3)     # omega:  small   (measured)
    P0[13:16, 13:16] = 1.0  * np.eye(3)     # d_f:    ±1 N    (unmeasured)
    P0[16:19, 16:19] = 0.1  * np.eye(3)     # d_τ:    ±0.3 Nm (unmeasured)
    return P0


# ═════════════════════════════════════════════════════════════════
# Extended Kalman Filter
# ═════════════════════════════════════════════════════════════════
class ExtendedKalmanFilter:
    """
    Augmented EKF for the 3D quadrotor with force + torque disturbances.

    Discretization:
        State:      RK4 integration of f_aug over one sample time
        Covariance: Euler linearization  F_d = I + F_c(z, u) · Ts
        Update:     Joseph form (numerically robust to non-symmetric P)

    Quaternion handling:
        - z[6:10] renormalized after prediction and after update
        - qw > 0 hemisphere enforced (consistent with plant convention)
        - Innovation flips measurement sign if y_q and hat_q lie in
          opposite hemispheres (SO(3) double-cover fix)
    """

    name = 'EKF'

    def __init__(self,
                 Ts: float,
                 z0: np.ndarray,
                 P0: np.ndarray = None,
                 Q:  np.ndarray = None,
                 R:  np.ndarray = None):
        self.Ts = Ts

        # ── Dynamics + measurement from the model file ─────────
        self.f_aug, self.F_func = get_augmented_dynamics()
        _, self.H = get_measurement_function()

        # ── Tuning (defaults if not supplied) ──────────────────
        self.Q = Q  if Q  is not None else default_ekf_Q(Ts)
        self.R = R  if R  is not None else default_ekf_R(None)
        self.P = P0 if P0 is not None else default_ekf_P0()

        # ── State ──────────────────────────────────────────────
        self.z = normalize_quaternion(np.asarray(z0, dtype=float).copy())

        # ── Shape checks ───────────────────────────────────────
        assert self.z.shape == (NZ,),      f'z0 shape {self.z.shape}'
        assert self.P.shape == (NZ, NZ),   f'P0 shape {self.P.shape}'
        assert self.Q.shape == (NZ, NZ),   f'Q shape  {self.Q.shape}'
        assert self.R.shape == (NY, NY),   f'R shape  {self.R.shape}'
        assert self.H.shape == (NY, NZ),   f'H shape  {self.H.shape}'

    # ── Estimator interface ─────────────────────────────────────
    def step(self, y: np.ndarray, u_prev: np.ndarray = None) -> np.ndarray:
        """
        One sample: predict with u_{k-1} (skipped at k = 0), update with y_k.
        Returns the posterior ẑ_k (19,).
        """
        if u_prev is not None:
            self.predict(u_prev)
        self.update(y)
        return self.z.copy()

    def info(self) -> str:
        """One-line description for the closed-loop summary."""
        return f'EKF: RK4 mean, Euler-linearized covariance, Ts = {self.Ts} s'

    # ── State access ────────────────────────────────────────────
    @property
    def x_hat(self) -> np.ndarray:
        """Estimated physical state (13,) — for the MPC."""
        return self.z[:NX].copy()

    @property
    def d_hat(self) -> np.ndarray:
        """Estimated disturbance vector (6,) — for MPC parameter update."""
        return self.z[NX:].copy()

    # ── Predict step ────────────────────────────────────────────
    def predict(self, u: np.ndarray) -> None:
        """
        Time update:
            z ← RK4(f_aug, z, u, Ts)
            P ← F_d P F_dᵀ + Q,   F_d = I + F_c(z, u) · Ts
        """
        # Linearize BEFORE propagation (F_d propagates P from k to k+1;
        # F evaluated at z_k is the standard EKF choice)
        F_c = np.array(self.F_func(self.z, u))
        F_d = np.eye(NZ) + F_c * self.Ts

        # RK4 propagation of z
        k1 = self._f(self.z,                    u)
        k2 = self._f(self.z + 0.5*self.Ts*k1,   u)
        k3 = self._f(self.z + 0.5*self.Ts*k2,   u)
        k4 = self._f(self.z +     self.Ts*k3,   u)
        self.z = normalize_quaternion(
            self.z + self.Ts/6.0 * (k1 + 2*k2 + 2*k3 + k4))

        # Covariance propagation
        self.P = F_d @ self.P @ F_d.T + self.Q

    # ── Update step ─────────────────────────────────────────────
    def update(self, y: np.ndarray) -> None:
        """
        Measurement update (Joseph form):
            S = H P Hᵀ + R
            K = P Hᵀ S⁻¹
            z ← z + K (y - H z)
            P ← (I - K H) P (I - K H)ᵀ + K R Kᵀ
        """
        H = self.H

        # Quaternion double-cover fix: if y_q and hat_q are in opposite
        # hemispheres, flip y_q so the innovation is small.
        y_used = y.copy()
        if np.dot(y[3:7], self.z[6:10]) < 0:
            y_used[3:7] = -y[3:7]

        # Innovation
        innov = y_used - H @ self.z

        # Kalman gain via linear solve (more stable than explicit inverse)
        S = H @ self.P @ H.T + self.R
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T

        # State update
        self.z = normalize_quaternion(self.z + K @ innov)

        # Covariance update (Joseph)
        I_KH = np.eye(NZ) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    # ── Helpers ─────────────────────────────────────────────────
    def _f(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Evaluate f_aug and return a 1-D numpy array."""
        return np.array(self.f_aug(z, u)).flatten()
