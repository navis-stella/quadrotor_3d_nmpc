"""
state_est_mhe.py — Stage 4b Augmented Lyapunov Moving Horizon Estimator
=======================================================================
Runtime wrapper around the acados MHE OCP (ocp_config_mhe.py) for the
augmented state — the MHE counterpart of state_est_ekf.py.

    State:        z = [x(13); d(6)]  ∈ R^19  (augmented)
    Measurement:  y = [p; q; ω]      ∈ R^10  (position, quaternion, angular velocity)

Model sources (quadrotor_3d_model) — shared with the Stage 4a EKF:
    - Augmented dynamics  (create_mhe_model = f_aug + process noise w)
    - Measurement model   (measurement_expr)

Tuning: none by hand. P, Q, R, the decay κ and the minimum horizon T_min
come from the i-iIOSS certificate (detectability_check.py); load_mhe_params
chooses the horizon N = ⌈c·T_min/Ts⌉ (c = HORIZON_FACTOR = 1.5) —
independent of the MPC horizon. Each node gets its age τ_i = t_k − t_i [s],
and the OCP discounts it by λ^{τ_i} (one decay symbol, λ per second).

Estimator interface (identical for state_est_ekf.ExtendedKalmanFilter):
    est = MovingHorizonEstimator(Ts, z0)
    z_hat_k = est.step(y_k, u_prev)     u_prev = u_{k-1}, None at k = 0

Window (prediction form — the OCP has measurements at nodes 0..N-1 and a
free endpoint at node N):
    node i ↔ time k-N+i,   y_meas[i] = y_{k-N+i},  u_rotor[i] = u_{k-N+i}
    node N carries ẑ_k  — y_k is stored now and first used at k+1
    arrival prior = ẑ_{k-N}, the estimate delivered N steps earlier
                    (filtering prior of Theorem 1)

Start-up — growing horizon, T_i = min(t_i, T) as in the paper:
    k = 0      ẑ_0 = z̄₀  (no past data yet)
    1 ≤ k < N  window [0, t_k] on the last k stages, prior z̄₀ at node N-k
    Everything above is carried by three rolling buffers of length N
    (ẑ, y, u), so the growing and the sliding phase are the same code.
"""

from collections import deque

import numpy as np

from quadrotor_3d_model import quat_normalize, f_hover, NU, NY, NZ, NW
from ocp_config_mhe     import (
    export_drone_mhe_solver, load_mhe_params, pack_mhe_param,
    MHE_PARAMS_PATH, HORIZON_FACTOR,
)


class MovingHorizonEstimator:
    """
    Lyapunov MHE with certificate-derived weights.

    Args:
        Ts:             sample time [s]
        z0:             initial prior z̄₀ (19,)
        cert_path:      i-iIOSS certificate (detectability_check.py)
        horizon_factor: N = ⌈c·T_min/Ts⌉, c > 1
        omega_box: impose the certified |ω̂| ≤ ω_max, constraint (8c)
    """

    name = 'MHE'

    def __init__(self,
                 Ts:        float,
                 z0:        np.ndarray,
                 cert_path:      str   = MHE_PARAMS_PATH,
                 horizon_factor: float = HORIZON_FACTOR,
                 omega_box:      bool  = True):
        self.Ts   = Ts
        self.cert = load_mhe_params(cert_path, Ts=Ts, horizon_factor=horizon_factor)
        self.N    = N = self.cert['N']

        self.solver = export_drone_mhe_solver(
            ts=Ts, lam=self.cert['lam'], N=N,
            P=self.cert['P'], R=self.cert['R'], Q=self.cert['Q'],
            omega_max=self.cert['omega_max'] if omega_box else None)

        z0 = quat_normalize(np.asarray(z0, dtype=float).copy())
        assert z0.shape == (NZ,), f'z0 shape {z0.shape}'

        # ── Rolling buffers (length N) ──────────────────────────
        #    z_buf[0] is the arrival prior: ẑ_{k-N}, or z̄₀ while growing
        self.z_buf = deque([z0], maxlen=N)      # ẑ_{k-n} … ẑ_{k-1}
        self.y_buf = deque(maxlen=N)            # y_{k-n} … y_{k-1}
        self.u_buf = deque(maxlen=N)            # u_{k-n} … u_{k-1}

        # Warm start — shifted solution of the previous solve; the first
        # solve starts from z̄₀ instead of acados' zeros (q = 0 makes the
        # quaternion error degenerate)
        self.z_init = np.tile(z0, (N + 1, 1))
        self.w_init = np.zeros((N, NW))

        # Diagnostics
        self.status   = []
        self.sqp_iter = []

    # ── Estimator interface ─────────────────────────────────────
    def step(self, y: np.ndarray, u_prev: np.ndarray = None) -> np.ndarray:
        """
        One sample: store u_{k-1}, solve over the window y_{k-n..k-1},
        store y_k for the next call. Returns ẑ_k (19,).
        """
        if u_prev is not None:
            self.u_buf.append(np.asarray(u_prev, dtype=float))

        if len(self.y_buf) == 0:                 # k = 0: no past data
            z_hat = self.z_buf[0].copy()
        else:
            z_hat = self._solve()
            self.z_buf.append(z_hat)

        self.y_buf.append(np.asarray(y, dtype=float))
        return z_hat.copy()

    def info(self) -> str:
        """One-line description for the closed-loop summary."""
        c   = self.cert
        st  = np.array(self.status)
        it  = np.array(self.sqp_iter)
        return (f'MHE: N = {self.N} (N_min = {c["N_min"]}, T = {c["T_hor"]:.2f} s '
                f'> T_min = {c["T_hor_min"]:.3f} s), λ = {c["lam"]}/s, '
                f'ρ = {c["rho"]:.3f}/s, SQP iter mean {it.mean():.1f} max {it.max()}, '
                f'failed {int(np.sum(~np.isin(st, (0, 2))))}, '
                f'max_iter {int(np.sum(st == 2))}')

    # ── One OCP solve ───────────────────────────────────────────
    def _solve(self) -> np.ndarray:
        """
        The n = len(y_buf) data samples fill the LAST n stages
        (nodes N-n..N-1); the prior sits at node N-n. For n < N the leading
        N-n nodes are padding: no measurement, no arrival, hover thrust —
        z₀ is free there and the penalized w makes the optimum pass
        straight through them.
        """
        N, solver = self.N, self.solver
        prior     = self.z_buf[0]
        n         = len(self.y_buf)
        i_arr     = N - n                          # node of the arrival cost

        # Node i sits at t_k − (N−i)·Ts: its age τ_i = (N−i)·Ts sets the
        # discount λ^{τ_i}. The arrival node N−n has age n·Ts = T_i
        # (λ^T once the horizon is full); the newest sample y_{k−1} has age Ts.
        for i in range(N):
            j     = i - i_arr                      # index into the data window
            t_age = (N - i) * self.Ts
            if j >= 0:
                p_i = pack_mhe_param(t_age, float(i == i_arr), 1.0,
                                     prior, self.y_buf[j], self.u_buf[j])
            else:
                p_i = pack_mhe_param(t_age, 0.0, 0.0,
                                     prior, np.zeros(NY), np.full(NU, f_hover))
            solver.set(i, 'p', p_i)
            solver.set(i, 'x', self.z_init[i])
            solver.set(i, 'u', self.w_init[i])

        # Terminal node N — no cost, but acados still needs a parameter of
        # the declared size.
        solver.set(N, 'p', pack_mhe_param(0.0, 0.0, 0.0, prior,
                                          np.zeros(NY), np.zeros(NU)))
        solver.set(N, 'x', self.z_init[N])

        status = solver.solve()
        self.status.append(status)
        self.sqp_iter.append(solver.get_stats('sqp_iter'))
        if status not in (0, 2):
            # non-fatal: report and continue with whatever the solver returned
            print(f'  [MHE] solver status {status}')

        z_traj = np.array([solver.get(i, 'x') for i in range(N + 1)])
        w_traj = np.array([solver.get(i, 'u') for i in range(N)])

        # shift by one node for the next solve
        self.z_init = np.vstack([z_traj[1:], z_traj[-1]])
        self.w_init = np.vstack([w_traj[1:], np.zeros(NW)])
        return z_traj[N]
