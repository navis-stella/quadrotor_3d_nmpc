"""
reference_generate.py — Stage 6 Helix Reference Trajectory
==========================================================
Time-varying reference for the trajectory-tracking NMPC: position and its
derivatives up to snap, all analytic, plus yaw. reference_flatness.py turns
a sample of it into the state/input reference (x_r, u_r).

Trajectory (fully analytic, derivatives up to snap):

  Phase 1  take-off t in [0, T_hop]               origin (at rest) -> p_start = (0,0,0.5) on the helix axis, rest-to-rest
  Phase 2  spiral   t in [T_hop, T_hop + T_ramp]  radius grows 0 -> R and phase rate theta_dot ramps 0 -> omega
  Phase 3  cruise   afterwards                    constant-radius, constant-rate helix (keeps running past `duration`)

Helix path, centred on the axis through p_start = the origin by default
(z-up by default, pass H < 0 if your frame has z pointing down):

  p(t) = [ cx + r(t) cos(theta(t)),  cy + r(t) sin(theta(t)),  pz + k * theta(t) ]
  axis     (cx, cy) = (p_start_x, p_start_y)
  climb    k = H / (2 pi n_turns)    ->  H metres of climb over n_turns revolutions

Smoothness: the hop and the phase-rate ramp both use an order-4 smoothstep
(first four derivatives vanish at both ends), so position, velocity, acceleration,
jerk AND snap are continuous everywhere (needed for the flatness-based u_r later).

Yaw is a constant (psi0); its derivatives are returned as zeros so the interface
is ready for yaw-to-centre later.

Usage:
    from reference_generate import HelixParams, HelixReference
    ref = HelixReference(HelixParams())
    r = ref.sample(1.234)                  # r.p, r.v, r.a, r.j, r.s, r.psi, ...
    hz = ref.horizon(t0, N, dt)            # arrays of shape (N+1, 3) for the OCP stages

    python reference_generate.py [--plot | --show] [--R 2 --H 5 --turns 3 --omega 1]
    (--plot saves results/stage6_trajectory/helix_reference.png)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import comb
from typing import NamedTuple

import numpy as np

# ─────────────────────────────────────────────────────────────────
# order-4 smoothstep s(tau) on [0,1]: s(0)=0, s(1)=1, derivatives 1..4 vanish at both ends
# ─────────────────────────────────────────────────────────────────
_P = np.polynomial.Polynomial
_s = [_P([0, 0, 0, 0, 0, 126, -420, 540, -315, 70])]   # s, s', s'', s''', s''''
for _ in range(4):
    _s.append(_s[-1].deriv())
_S = _s[0].integ()                                     # S(tau) = int_0^tau s, S(1) = 0.5


class RefSample(NamedTuple):
    """Reference at time(s) t. Arrays are (3,) for scalar t and (K, 3) for K times."""
    p: np.ndarray      # position
    v: np.ndarray      # velocity
    a: np.ndarray      # acceleration
    j: np.ndarray      # jerk
    s: np.ndarray      # snap
    psi: np.ndarray    # yaw
    dpsi: np.ndarray   # yaw rate
    ddpsi: np.ndarray  # yaw acceleration


@dataclass(frozen=True)
class HelixParams:
    R: float = 2.0                                   # helix radius [m]
    H: float = 5.0                                   # total climb [m] (negative -> z-down frames)
    n_turns: float = 3.0                             # total revolutions of the helix phase
    omega: float = 1.0                               # cruise phase rate [rad/s]; speed = R*omega, a = R*omega^2
    p_origin: tuple = (0.0, 0.0, 0.0)                # start of the hop (drone at rest)
    p_start: tuple = (0.0, 0.0, 0.5)                 # helix start point = end of the hop (vertical take-off)
    T_hop: float = 2.0                               # duration of the hop [s]
    T_ramp: float = 4.0                              # duration of the phase-rate ramp [s]
    psi0: float = 0.0                                # constant yaw [rad]


class HelixReference:
    def __init__(self, params: HelixParams | None = None):
        self.params = params or HelixParams()
        P = self.params
        if P.R <= 0 or P.omega <= 0 or P.n_turns <= 0 or P.T_hop <= 0 or P.T_ramp <= 0:
            raise ValueError("R, omega, n_turns, T_hop, T_ramp must be positive")
        self.k = P.H / (2.0 * np.pi * P.n_turns)
        self.center = (P.p_start[0], P.p_start[1])   # helix axis passes through the end of the take-off
        cruise = (2.0 * np.pi * P.n_turns - 0.5 * P.omega * P.T_ramp) / P.omega
        if cruise < 0:
            raise ValueError("ramp covers more than the whole helix: reduce T_ramp/omega or raise n_turns")
        self.duration = P.T_hop + P.T_ramp + cruise   # nominal end; reference keeps going afterwards

    # ------------------------------------------------------------------
    def sample(self, t) -> RefSample:
        P = self.params
        t_in = np.asarray(t, dtype=float)
        scalar = t_in.ndim == 0
        t_arr = np.atleast_1d(t_in).ravel()
        K = t_arr.size
        p, v, a, j, s = (np.zeros((K, 3)) for _ in range(5))

        # ---- phase 1: rest-to-rest hop (straight line, smoothstep timing)
        m1 = t_arr < P.T_hop
        if m1.any():
            p0 = np.asarray(P.p_origin, float)
            D = np.asarray(P.p_start, float) - p0
            tau = np.clip(t_arr[m1] / P.T_hop, 0.0, 1.0)
            T = P.T_hop
            p[m1] = p0 + D * _s[0](tau)[:, None]
            v[m1] = D * (_s[1](tau) / T)[:, None]
            a[m1] = D * (_s[2](tau) / T**2)[:, None]
            j[m1] = D * (_s[3](tau) / T**3)[:, None]
            s[m1] = D * (_s[4](tau) / T**4)[:, None]

        # ---- phases 2+3: time-warped helix h(theta(t))
        m2 = ~m1
        if m2.any():
            u = t_arr[m2] - P.T_hop
            Tr, w = P.T_ramp, P.omega
            ramp = u < Tr
            tau = np.clip(u / Tr, 0.0, 1.0)
            th = np.where(ramp, w * Tr * _S(tau), 0.5 * w * Tr + w * (u - Tr))
            thd = np.where(ramp, w * _s[0](tau), w)
            thdd = np.where(ramp, w * _s[1](tau) / Tr, 0.0)
            thd3 = np.where(ramp, w * _s[2](tau) / Tr**2, 0.0)
            thd4 = np.where(ramp, w * _s[3](tau) / Tr**3, 0.0)

            k = self.k
            cx, cy = self.center
            # radius r(t): grows 0 -> R with the same smoothstep as the phase-rate ramp, then stays R
            rad = [np.where(ramp, P.R * _s[n](tau) / Tr**n, P.R if n == 0 else 0.0) for n in range(5)]
            # xy = centre + Re/Im of r(t) * E(t), E = exp(i*theta(t));  d^n E/dt^n = Pn * E
            E = np.exp(1j * th)
            Pn = [np.ones_like(E),
                  1j * thd,
                  1j * thdd - thd**2,
                  1j * thd3 - 3 * thd * thdd - 1j * thd**3,
                  1j * thd4 - 4 * thd * thd3 - 3 * thdd**2 - 6j * thd**2 * thdd + thd**4]
            w = [sum(comb(n, m) * rad[n - m] * Pn[m] for m in range(n + 1)) * E for n in range(5)]
            thn = [th, thd, thdd, thd3, thd4]
            out_xy = []
            for n in range(5):
                zn = k * thn[n]
                if n == 0:
                    zn = P.p_start[2] + zn
                out_xy.append(np.stack([w[n].real + (cx if n == 0 else 0.0),
                                        w[n].imag + (cy if n == 0 else 0.0), zn], -1))
            p[m2], v[m2], a[m2], j[m2], s[m2] = out_xy

        psi = np.full(K, P.psi0)
        zeros = np.zeros(K)
        out = RefSample(p, v, a, j, s, psi, zeros, zeros.copy())
        if scalar:
            out = RefSample(*(x[0] for x in out))
        return out

    def horizon(self, t0: float, N: int, dt: float) -> RefSample:
        """Reference at the N+1 OCP stages t0 + k*dt, k = 0..N (arrays of shape (N+1, 3))."""
        return self.sample(t0 + dt * np.arange(N + 1))


# ─────────────────────────────────────────────────────────────────
def self_test(ref: HelixReference, g_vec=(0.0, 0.0, -9.81), verbose: bool = True) -> dict:
    P = ref.params
    Th, Tr = P.T_hop, P.T_ramp

    # 1) at rest at the origin
    r0 = ref.sample(0.0)
    assert np.allclose(r0.p, P.p_origin, atol=1e-12)
    for name in ("v", "a", "j", "s"):
        assert np.allclose(getattr(r0, name), 0.0, atol=1e-9), f"{name}(0) != 0"

    # 2) hop ends at the helix start point, at rest
    r1 = ref.sample(Th)
    assert np.allclose(r1.p, P.p_start, atol=1e-9)
    assert np.allclose(r1.v, 0.0, atol=1e-9)

    # 3) continuity of p, v, a, j, snap across both junctions
    eps = 1e-9
    for tj in (Th, Th + Tr):
        lo, hi = ref.sample(tj - eps), ref.sample(tj + eps)
        for name in ("p", "v", "a", "j", "s"):
            err = np.max(np.abs(getattr(lo, name) - getattr(hi, name)))
            assert err < 1e-6, f"{name} jumps by {err:.2e} at t={tj}"

    # 4) analytic derivative chain p -> v -> a -> j -> s against central differences
    h = 1e-5
    rng = np.random.default_rng(0)
    tt = rng.uniform(0.0, ref.duration + 5.0, 600)
    tt = tt[np.all([np.abs(tt - tj) > 1e-3 for tj in (0.0, Th, Th + Tr)], axis=0)]
    for lo_n, hi_n in (("p", "v"), ("v", "a"), ("a", "j"), ("j", "s")):
        fd = (getattr(ref.sample(tt + h), lo_n) - getattr(ref.sample(tt - h), lo_n)) / (2 * h)
        an = getattr(ref.sample(tt), hi_n)
        err = np.max(np.abs(fd - an))
        assert err < 1e-5 * (1.0 + np.max(np.abs(an))), f"d({lo_n})/dt != {hi_n}: {err:.2e}"

    # 4b) cruise phase is a circle of radius R around the axis through p_start
    rc = ref.sample(Th + Tr + np.linspace(0.0, 5.0, 50))
    rho = np.linalg.norm(rc.p[:, :2] - np.asarray(P.p_start[:2]), axis=1)
    assert np.allclose(rho, P.R, atol=1e-9), "cruise is not a circle of radius R around the axis"

    # 5) summary of what the drone has to do
    t_grid = np.linspace(0.0, ref.duration, 4000)
    r = ref.sample(t_grid)
    g = np.asarray(g_vec, float)
    f = r.a - g                                   # m * f = required thrust vector (no drag, no disturbance)
    tilt = np.degrees(np.arccos(np.clip(f @ (-g) / (np.linalg.norm(f, axis=1) * np.linalg.norm(g)), -1, 1)))
    info = dict(
        duration=ref.duration,
        v_max=float(np.max(np.linalg.norm(r.v, axis=1))),
        a_max=float(np.max(np.linalg.norm(r.a, axis=1))),
        tilt_max_deg=float(np.max(tilt)),
        thrust_per_mass_max=float(np.max(np.linalg.norm(f, axis=1))),   # [N/kg]
        z_end=float(r.p[-1, 2]),
    )
    if verbose:
        print("self-test passed (rest start, continuity up to snap, finite-difference derivatives)")
        print(f"  duration        : {info['duration']:.2f} s (hop {Th:.1f} + ramp {Tr:.1f} + cruise)")
        print(f"  peak speed      : {info['v_max']:.2f} m/s")
        print(f"  peak accel      : {info['a_max']:.2f} m/s^2")
        print(f"  peak tilt       : {info['tilt_max_deg']:.1f} deg")
        print(f"  peak thrust / m : {info['thrust_per_mass_max']:.2f} N/kg  (multiply by your mass)")
        print(f"  final height    : {info['z_end']:.2f} m")
    return info


def plot(ref: HelixReference, path: str = "results/stage6_trajectory/helix_reference.png",
         show: bool = False) -> None:
    import os
    import matplotlib
    if not show:
        matplotlib.use("Agg")          # headless: only save the PNG
    import matplotlib.pyplot as plt

    P = ref.params
    t = np.linspace(0.0, ref.duration, 3000)
    r = ref.sample(t)
    seg = [(t < P.T_hop, "tab:orange", "take-off"),
           ((t >= P.T_hop) & (t < P.T_hop + P.T_ramp), "tab:green", "spiral-out"),
           (t >= P.T_hop + P.T_ramp, "tab:blue", "cruise")]
    fig = plt.figure(figsize=(11, 4.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for m, color, label in seg:
        ax.plot(r.p[m, 0], r.p[m, 1], r.p[m, 2], color=color, label=label, lw=2)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.legend(loc="upper left")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(t, np.linalg.norm(r.v, axis=1), label="|v| [m/s]")
    ax2.plot(t, np.linalg.norm(r.a, axis=1), label="|a| [m/s$^2$]")
    ax2.plot(t, np.linalg.norm(r.j, axis=1), label="|jerk| [m/s$^3$]", alpha=0.7)
    ax2.set_xlabel("t [s]"); ax2.grid(alpha=0.3); ax2.legend()
    fig.tight_layout()
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"saved {path}")
    if show:
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--R", type=float, default=2.0)
    ap.add_argument("--H", type=float, default=5.0)
    ap.add_argument("--turns", type=float, default=3.0)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--plot", action="store_true", help="save results/stage6_trajectory/helix_reference.png")
    ap.add_argument("--show", action="store_true", help="also open an interactive window (needs a display, e.g. WSLg)")
    args = ap.parse_args()
    ref = HelixReference(HelixParams(R=args.R, H=args.H, n_turns=args.turns, omega=args.omega))
    self_test(ref)
    if args.plot or args.show:
        plot(ref, show=args.show)
