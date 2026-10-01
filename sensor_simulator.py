"""
sensor_simulator.py — Stage 4 Measurement Simulator
===================================================
Simulates the sensor stream y = h(x) = [p; q; ω] for the Stage 4
estimator comparison (EKF vs MHE see exactly the same noisy stream).

Assumption (Stage 4, project-wide):
    Position p, quaternion q, and angular velocity ω are directly
    measured — representing the fused output of an idealized outer
    pose source (mocap-equivalent). Linear velocity v and disturbances
    d are NOT measured; the estimator must infer them.

Interface:
    plant produces x (13,)  →  sensor.measure(x)  →  y (10,)
    The sensor is disturbance-blind: it sees only the physical state,
    not the augmented disturbance vector d that lives in the estimator.

Noise model:
    - Additive Gaussian white noise per channel group when enabled:
        pos   ~ N(0, σ_p²  · I₃)   [m]
        quat  ~ N(0, σ_q²  · I₄)   [-]  (renormalized after injection)
        omega ~ N(0, σ_ω²  · I₃)   [rad/s]
    - Mocap-equivalent σ values (MOCAP_NOISE_STD) are used in every Stage 4
      closed-loop run, and the EKF's R is matched to the same dict.
      Passing noise_std=None gives a noise-free sensor,
      retained for open-loop EKF checks where any noise would
      confound pass/fail thresholds.
    - Quaternion noise is renormalized to preserve ||q|| = 1. This is
      a pragmatic approximation valid for small σ_q; a tangent-space
      perturbation would be more principled for large noise.

Not modeled:
    Sensor-specific biases, latency, sample-rate mismatch, dropouts,
    IMU/GPS/VIO raw-data fusion.
"""

import numpy as np

from quadrotor_3d_model import NX, NY, IDX_Y

# Mocap-equivalent noise — the single definition used by the closed loop
# (sensor) and by the EKF (matched R).
MOCAP_NOISE_STD = {
    'pos':   0.01,     # [m]      1 cm
    'quat':  0.001,    # [-]      ≈ 0.1° attitude  (φ ≈ 2·qx)
    'omega': 0.005,    # [rad/s]  ≈ 0.3°/s
}


class SensorSimulator:
    """
    Stage 4 measurement simulator: y = [p; q; ω].

    Args:
        noise_std: dict or None
            None                → noise-free (default; open-loop checks)
            {'pos': σ_p,        → σ = 0 for any missing key
             'quat': σ_q,
             'omega': σ_ω}
        seed: int or None
            RNG seed for reproducibility.
    """

    def __init__(self, noise_std: dict = None, seed: int = None):
        self.noise_free = (noise_std is None)

        if self.noise_free:
            self.sigma_pos = 0.0
            self.sigma_quat = 0.0
            self.sigma_omega = 0.0
        else:
            self.sigma_pos   = float(noise_std.get('pos',   0.0))
            self.sigma_quat  = float(noise_std.get('quat',  0.0))
            self.sigma_omega = float(noise_std.get('omega', 0.0))

        self.rng = np.random.default_rng(seed)

    # ─────────────────────────────────────────────────────────
    def measure(self, x: np.ndarray) -> np.ndarray:
        """
        Produce one measurement from the plant state.

        Args:
            x: (13,) plant state [p(3); v(3); q(4); ω(3)]
        Returns:
            y: (10,) measurement [p(3); q(4); ω(3)]
        """
        assert x.shape == (NX,), f'expected x of shape ({NX},), got {x.shape}'

        y = x[IDX_Y].astype(float)     # [p; q; ω] — same selection as h(z)

        if self.noise_free:
            return y

        # Additive Gaussian noise per channel group
        y[0:3]  += self.sigma_pos   * self.rng.standard_normal(3)
        y[3:7]  += self.sigma_quat  * self.rng.standard_normal(4)
        y[7:10] += self.sigma_omega * self.rng.standard_normal(3)

        # Renormalize quaternion after noise injection
        q_norm = np.linalg.norm(y[3:7])
        if q_norm > 1e-12:
            y[3:7] /= q_norm

        return y


# ─────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from quadrotor_3d_model import get_measurement_function, ND, NZ

    # ── 1. Noise-free measurement selects the right components ──
    sensor = SensorSimulator()
    x = np.random.randn(NX)
    y = sensor.measure(x)

    assert y.shape == (NY,)
    assert np.allclose(y[0:3],  x[0:3])
    assert np.allclose(y[3:7],  x[6:10])
    assert np.allclose(y[7:10], x[10:13])
    print('✓ Noise-free measurement selects (p, q, ω) correctly')

    # ── 2. Consistency with model-file h(z) ─────────────────────
    #    Sensor takes x (13); h takes z = [x; d] (19).
    #    Results must agree for ANY d — sensor is disturbance-blind.
    h_func, H_z = get_measurement_function()
    d_random = np.random.randn(ND)
    z = np.concatenate([x, d_random])
    y_from_h = np.array(h_func(z)).flatten()
    assert np.allclose(y_from_h, y)
    print('✓ Sensor output matches h(z) — disturbance-invariant')

    # ── 3. Noisy mode: noise applied, quaternion renormalized ──
    sensor_noisy = SensorSimulator(noise_std=MOCAP_NOISE_STD, seed=42)
    x_hover = np.zeros(NX); x_hover[6] = 1.0
    y_noisy = sensor_noisy.measure(x_hover)
    assert not np.allclose(y_noisy[0:3], x_hover[0:3])           # noise present
    assert np.isclose(np.linalg.norm(y_noisy[3:7]), 1.0, atol=1e-12)  # renorm
    print('✓ Noise injection + quaternion renormalization work')

    # ── 4. Reproducibility with seed ─────────────────────────────
    s1 = SensorSimulator(noise_std={'pos': 0.1}, seed=7)
    s2 = SensorSimulator(noise_std={'pos': 0.1}, seed=7)
    assert np.allclose(s1.measure(x_hover), s2.measure(x_hover))
    print('✓ Seeded RNG is reproducible')

    print('\nSensor simulator ready.')
