"""
reference_flatness.py — Stage 6 Differential-Flatness Map
=========================================================
(p, v, a, jerk, snap, psi, dpsi, ddpsi)  ->  (x_r, u_r), after
D. Mellinger, V. Kumar, "Minimum snap trajectory generation and control for
quadrotors," ICRA 2011, https://doi.org/10.1109/ICRA.2011.5980409
(here with the ZYX yaw convention of quat_to_euler and a disturbance term).

Turns the planner output (a RefSample from reference_generate.py) into everything the
tracking OCP needs at each stage:

    x_r = [p_r, v_r, q_r, w_r]      13   same ordering as quadrotor_3d_model (q scalar-first, w body frame)
    u_r = [f1 f2 f3 f4]_r            4   per-rotor thrusts [N], "+" layout (MIXER)

Everything is taken from quadrotor_3d_model (m, g, I, MIXER, F_MAX, OMEGA_MAX), so the
map can never disagree with the prediction model.

Model convention being inverted (z-up, world-frame force disturbance, body-frame torque
disturbance, both constant over the horizon):

    m a      = T R e3 - m g e3 + d_f                ->  f := T z_B = m (a + g e3) - d_f
    I w_dot  = tau - w x I w + d_tau                ->  tau = I alpha + w x I w - d_tau
    [T, tau] = MIXER @ u                            ->  u = MIXER^-1 [T, tau]

Steps (all derivatives analytic, up to snap):
    1. f, f_dot, f_ddot            from a, jerk, snap  (d_hat treated as constant: d_dot = 0)
    2. T, z_B and derivatives      by differentiating the normalisation f / |f|
    3. x_B, y_B and derivatives    yaw convention ZYX (same as quat_to_euler in the model):
                                   y_C = [-sin psi, cos psi, 0],  x_B = y_C x z_B / |.|,  y_B = z_B x x_B
    4. w_r, alpha_r                from  R^T R_dot = [w]x  and  R^T R_ddot = [w]x^2 + [alpha]x
    5. u_r                         from the wrench through the inverse mixer

A disturbance estimate d_hat = [d_fx d_fy d_fz d_tx d_ty d_tz] (EKF / MHE) enters steps 1 and
5, which keeps the reference consistent with the MPC prediction model f(x, u, d_hat).

Usage:
    from reference_generate import HelixReference
    from reference_flatness import flat_to_state_input, horizon_xu
    out = flat_to_state_input(ref.horizon(t, N, dt), d_hat)     # out.x_r (N+1,13), out.u_r (N+1,4)
    x_r, u_r = horizon_xu(ref, t, N, dt, d_hat)

    python reference_flatness.py   # self-test: open-loop flight of u_r through the model reproduces x_r

Singularities (not reached by the helix): free fall |f| = 0 (z_B undefined) and
z_B parallel to y_C (90 deg pitch, x_B undefined).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

import quadrotor_3d_model as qm

_I = np.array([qm.Ixx, qm.Iyy, qm.Izz])


class FlatOut(NamedTuple):
    """Arrays are (K, .) ; for a scalar-time RefSample the leading axis is dropped."""
    x_r: np.ndarray      # (13,)  [p, v, q, w]
    u_r: np.ndarray      # (4,)   per-rotor thrusts
    T: np.ndarray        # ()     collective thrust [N]
    tau: np.ndarray      # (3,)   body torque [N m]
    R: np.ndarray        # (3,3)  body-to-world rotation
    alpha: np.ndarray    # (3,)   body angular acceleration [rad/s^2]


# ─────────────────────────────────────────────────────────────────
# helpers (all on arrays of shape (K, 3))
# ─────────────────────────────────────────────────────────────────
def _dot(a, b):
    return np.sum(a * b, axis=-1, keepdims=True)


def _unit_derivs(a, ad, add):
    """n = |a|, u = a/n and their first two time derivatives."""
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    u = a / n
    nd = _dot(u, ad)
    ud = (ad - nd * u) / n
    ndd = _dot(u, add) + n * _dot(ud, ud)
    udd = (add - ndd * u - 2.0 * nd * ud) / n
    return n, nd, ndd, u, ud, udd


def _skew_vec(M):
    """Axis vector of the skew-symmetric part of M (K,3,3):  [M32-M23, M13-M31, M21-M12] / 2."""
    return 0.5 * np.stack([M[:, 2, 1] - M[:, 1, 2],
                           M[:, 0, 2] - M[:, 2, 0],
                           M[:, 1, 0] - M[:, 0, 1]], axis=-1)


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """(K,3,3) -> (K,4) scalar-first unit quaternions with qw >= 0 (Shepperd, branch-free selection)."""
    r = R
    t = np.stack([1 + r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2],
                  1 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2],
                  1 - r[:, 0, 0] + r[:, 1, 1] - r[:, 2, 2],
                  1 - r[:, 0, 0] - r[:, 1, 1] + r[:, 2, 2]], axis=-1)
    cand = np.stack([
        np.stack([t[:, 0], r[:, 2, 1] - r[:, 1, 2], r[:, 0, 2] - r[:, 2, 0], r[:, 1, 0] - r[:, 0, 1]], -1),
        np.stack([r[:, 2, 1] - r[:, 1, 2], t[:, 1], r[:, 0, 1] + r[:, 1, 0], r[:, 0, 2] + r[:, 2, 0]], -1),
        np.stack([r[:, 0, 2] - r[:, 2, 0], r[:, 0, 1] + r[:, 1, 0], t[:, 2], r[:, 1, 2] + r[:, 2, 1]], -1),
        np.stack([r[:, 1, 0] - r[:, 0, 1], r[:, 0, 2] + r[:, 2, 0], r[:, 1, 2] + r[:, 2, 1], t[:, 3]], -1),
    ], axis=1)                                                    # (K, 4 candidates, 4)
    k = np.argmax(t, axis=-1)
    q = cand[np.arange(len(k)), k]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return q * np.where(q[:, :1] < 0, -1.0, 1.0)                  # qw >= 0, like quat_normalize


def make_quat_continuous(q: np.ndarray, q_prev: np.ndarray | None = None) -> np.ndarray:
    """Flip signs along a (K,4) sequence so that consecutive quaternions have positive dot product.
    q and -q are the same rotation; a sign flip inside the horizon would look like a 2*pi error to the cost."""
    q = q.copy()
    prev = q_prev
    for i in range(len(q)):
        if prev is not None and np.dot(q[i], prev) < 0:
            q[i] = -q[i]
        prev = q[i]
    return q


# ─────────────────────────────────────────────────────────────────
def flat_to_state_input(ref, d_hat=None) -> FlatOut:
    """
    ref   : RefSample (reference_generate.py) - scalar time (arrays (3,)) or K times (arrays (K,3))
    d_hat : disturbance estimate [d_fx d_fy d_fz d_tx d_ty d_tz], shape (6,) or (K,6); default 0.
            Treated as constant in time (matches the d_dot = 0 assumption of EKF / MHE).
    """
    scalar = np.asarray(ref.p).ndim == 1
    g3 = lambda x: np.atleast_2d(np.asarray(x, float))
    a, j, s = g3(ref.a), g3(ref.j), g3(ref.s)
    K = a.shape[0]
    psi, dpsi, ddpsi = (np.broadcast_to(np.asarray(x, float), (K,)) for x in (ref.psi, ref.dpsi, ref.ddpsi))
    d = np.zeros((K, qm.ND)) if d_hat is None else np.broadcast_to(np.asarray(d_hat, float), (K, qm.ND))
    d_f, d_tau = d[:, :3], d[:, 3:]

    # 1) required thrust vector f = T z_B   (world frame) and its derivatives
    e3 = np.array([0.0, 0.0, 1.0])
    f = qm.m * (a + qm.g * e3) - d_f
    fd = qm.m * j
    fdd = qm.m * s

    # 2) T, z_B
    T, _, _, zB, zBd, zBdd = _unit_derivs(f, fd, fdd)

    # 3) yaw-dependent axes (ZYX convention: x_B is orthogonal to y_C)
    c, sn, zero = np.cos(psi), np.sin(psi), np.zeros(K)
    yC = np.stack([-sn, c, zero], -1)
    yCd = dpsi[:, None] * np.stack([-c, -sn, zero], -1)
    yCdd = ddpsi[:, None] * np.stack([-c, -sn, zero], -1) + (dpsi ** 2)[:, None] * np.stack([sn, -c, zero], -1)

    cr = np.cross(yC, zB)
    crd = np.cross(yCd, zB) + np.cross(yC, zBd)
    crdd = np.cross(yCdd, zB) + 2.0 * np.cross(yCd, zBd) + np.cross(yC, zBdd)
    _, _, _, xB, xBd, xBdd = _unit_derivs(cr, crd, crdd)

    yB = np.cross(zB, xB)
    yBd = np.cross(zBd, xB) + np.cross(zB, xBd)
    yBdd = np.cross(zBdd, xB) + 2.0 * np.cross(zBd, xBd) + np.cross(zB, xBdd)

    R = np.stack([xB, yB, zB], axis=-1)       # columns = body axes in world frame
    Rd = np.stack([xBd, yBd, zBd], axis=-1)
    Rdd = np.stack([xBdd, yBdd, zBdd], axis=-1)

    # 4) body rates / accelerations:  R^T Rd = [w]x,  R^T Rdd = [w]x^2 + [alpha]x   (skew part)
    Rt = np.swapaxes(R, 1, 2)
    w = _skew_vec(Rt @ Rd)
    alpha = _skew_vec(Rt @ Rdd)

    # 5) wrench and rotor thrusts
    tau = _I * alpha + np.cross(w, _I * w) - d_tau
    wrench = np.concatenate([T, tau], axis=-1)                    # (K,4)
    u = np.linalg.solve(qm.MIXER, wrench.T).T                     # MIXER @ u = [T, tau]

    q = make_quat_continuous(rot_to_quat(R))
    x = np.concatenate([g3(ref.p), g3(ref.v), q, w], axis=-1)

    out = FlatOut(x, u, T[:, 0], tau, R, alpha)
    return FlatOut(*(o[0] for o in out)) if scalar else out


def horizon_xu(ref_gen, t0: float, N: int, dt: float, d_hat=None):
    """Reference states (N+1,13) and inputs (N+1,4) at the OCP stages t0 + k dt. The last input is only
    needed if you track u_r on the terminal node (usually you do not)."""
    out = flat_to_state_input(ref_gen.horizon(t0, N, dt), d_hat)
    return out.x_r, out.u_r


# ─────────────────────────────────────────────────────────────────
# self-test
# ─────────────────────────────────────────────────────────────────
def _quat_angle(q1, q2):
    """Rotation angle between two quaternion sequences (inputs normalised first: RK4 does not keep |q| = 1)."""
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)
    return 2.0 * np.arccos(np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0.0, 1.0))


def _open_loop(ref_gen, d_hat, d_true, h=2e-3):
    """RK4 flight of the model with u_r(t) fed in continuously (u_r sampled at the RK stage times)."""
    import casadi as ca
    s = qm._build_core_symbols()
    dsym = ca.SX.sym("d", qm.ND)
    f = ca.Function("f", [s["x"], s["u"], dsym], [qm._build_f_expl(s, dsym)])

    n = int(np.floor(ref_gen.duration / h))
    tt = h * 0.5 * np.arange(2 * n + 1)                           # t_k, t_k + h/2, t_k + h, ...
    out = flat_to_state_input(ref_gen.sample(tt), d_hat)
    x = out.x_r[0].copy()
    X = np.zeros((n + 1, qm.NX)); X[0] = x
    d_true = np.zeros(qm.ND) if d_true is None else np.asarray(d_true, float)
    for k in range(n):
        u0, um, u1 = out.u_r[2 * k], out.u_r[2 * k + 1], out.u_r[2 * k + 2]
        k1 = f(x, u0, d_true).full().ravel()
        k2 = f(x + 0.5 * h * k1, um, d_true).full().ravel()
        k3 = f(x + 0.5 * h * k2, um, d_true).full().ravel()
        k4 = f(x + h * k3, u1, d_true).full().ravel()
        x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        X[k + 1] = x
    ref_x = out.x_r[::2]
    return tt[::2], X, ref_x, out


def self_test(ref_gen=None, verbose: bool = True) -> dict:
    from reference_generate import HelixReference
    ref_gen = ref_gen or HelixReference()

    # 1) hover: at t = 0 the reference is at rest -> level attitude, f_hover per rotor, w = 0
    o0 = flat_to_state_input(ref_gen.sample(0.0))
    assert np.allclose(o0.u_r, qm.f_hover, atol=1e-9), o0.u_r
    assert np.allclose(o0.x_r[6:10], [1, 0, 0, 0], atol=1e-9)
    assert np.allclose(o0.x_r[10:], 0.0, atol=1e-9)

    # 2) geometry: R orthonormal, R e3 * T = m (a + g e3)
    t_grid = np.linspace(0.0, ref_gen.duration, 2000)
    rs = ref_gen.sample(t_grid)
    o = flat_to_state_input(rs)
    RtR = np.swapaxes(o.R, 1, 2) @ o.R
    assert np.allclose(RtR, np.eye(3), atol=1e-12) and np.allclose(np.linalg.det(o.R), 1.0, atol=1e-12)
    f_chk = o.T[:, None] * o.R[:, :, 2]
    assert np.allclose(f_chk, qm.m * (rs.a + qm.g * np.array([0, 0, 1.0])), atol=1e-10)

    # 3) yaw in the model's own (ZYX) convention equals the commanded yaw
    eul = qm.quat_to_euler(o.x_r[:, 6:10])
    assert np.allclose(eul[:, 2], rs.psi, atol=1e-9), "yaw mismatch with quat_to_euler"

    # 4) the decisive check: fly u_r open loop through the model, with and without disturbance
    res = {}
    d_test = np.array([0.30, -0.20, 0.50, 0.010, -0.020, 0.005])  # wind + mass error + parasitic torques
    for label, d_hat, d_true in (("nominal", None, None),
                                 ("with d (d_hat = d_true)", d_test, d_test),
                                 ("with d, d_hat = 0 [control, must fail]", None, d_test)):
        t, X, Xr, _ = _open_loop(ref_gen, d_hat, d_true)
        pe = np.max(np.linalg.norm(X[:, :3] - Xr[:, :3], axis=1))
        ve = np.max(np.linalg.norm(X[:, 3:6] - Xr[:, 3:6], axis=1))
        qe = np.degrees(np.max(_quat_angle(X[:, 6:10], Xr[:, 6:10])))
        we = np.max(np.linalg.norm(X[:, 10:] - Xr[:, 10:], axis=1))
        res[label] = (pe, ve, qe, we)
        if verbose:
            print(f"  open-loop {label:40s} max |dp| {pe:9.2e} m   |dv| {ve:9.2e} m/s   "
                  f"dq {qe:9.2e} deg   |dw| {we:9.2e} rad/s")
    for key in ("nominal", "with d (d_hat = d_true)"):
        pe, ve, qe, we = res[key]
        assert max(pe, ve, we) < 1e-6 and qe < 1e-3, f"open-loop flight ({key}) does not reproduce the reference"
    assert res["with d, d_hat = 0 [control, must fail]"][0] > 1e-2, "test is not sensitive to d"

    # 5) envelope
    om = np.max(np.abs(o.x_r[:, 10:]), axis=0)
    info = dict(w_max=om, u_min=float(o.u_r.min()), u_max=float(o.u_r.max()),
                tau_max=np.max(np.abs(o.tau), axis=0), alpha_max=np.max(np.abs(o.alpha), axis=0))
    if verbose:
        print("flatness self-test passed")
        print(f"  |w_r| max        : {np.round(om, 3)} rad/s   (MPC keeps |w| <= {0.9 * qm.OMEGA_MAX:.2f})")
        print(f"  |alpha_r| max    : {np.round(info['alpha_max'], 3)} rad/s^2")
        print(f"  |tau_r| max      : {np.round(info['tau_max'], 4)} N m")
        print(f"  rotor thrust u_r : [{info['u_min']:.3f}, {info['u_max']:.3f}] N   (limits [0, {qm.F_MAX:.3f}])")
    return info


if __name__ == "__main__":
    self_test()
