"""
compute_qih_params.py — Offline QIH-NMPC Parameter Computation
================================================================
Stage 2.1:  Complete Quasi-Infinite Horizon NMPC

Computes the terminal cost P, terminal region radius alpha, stability
margin kappa, and auxiliary LQR gain K for QIH-NMPC.

Theory (Chen & Allgöwer, 1998 — continuous-time formulation):
    1. Auxiliary controller:  kappa_f(x) = K x,  K = -R^{-1} B^T P_care
    2. Stability margin:     0 < kappa < -max Re(eig(A_K))
    3. Terminal cost P:      (A_K + kappa I)^T P + P (A_K + kappa I) = -(Q + K^T R K)
    4. Terminal set:         Omega_alpha = { x : x^T P x <= alpha }
       where alpha is the largest value such that:
         (a) input constraints are satisfied under kappa_f
         (b) nonlinear residual phi(x) = f(x, Kx) - A_K x is small enough

    Key Lyapunov property of P:
        V_f(x) = x^T P x satisfies  dV_f/dt <= -kappa V_f(x)
    under the auxiliary controller inside Omega_alpha.  This exponential
    decrease is stronger than the standard DARE/CARE Lyapunov decrease
    and is what enables the formal QIH stability proof.

    Note: P_lyap ≠ P_DARE.
        DARE gives the infinite-horizon LQR cost-to-go.
        The modified Lyapunov equation (with kappa) gives a matrix that
        enforces an exponential decay rate, making P_lyap > P_care.
        This stronger requirement shrinks the terminal set.

Quaternion handling:
    qw is uncontrollable at hover — removed for CARE/Lyapunov, then
    P_red (12×12) is embedded back into P_full (13×13).

Used by:  ocp_config_qih.py  (Stage 2.1 solver)
"""

import numpy as np
import casadi as ca
from scipy.linalg import (solve_continuous_are, solve_continuous_lyapunov,
                           inv, norm, cholesky)

from quadrotor_3d_model import (get_hover_linearization, create_model,
                                f_hover, NX, NU)


# ─────────────────────────────────────────────────────────────────
# Indices for quaternion reduction
# ─────────────────────────────────────────────────────────────────
# Full state:    [px,py,pz, vx,vy,vz, qw, qx,qy,qz, p,q,r]  (13)
# Reduced state: [px,py,pz, vx,vy,vz,     qx,qy,qz, p,q,r]  (12)
IDX_KEEP = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]   # skip qw (index 6)
N_RED = 12


def compute_qih_offline_parameters(Q: np.ndarray,
                                   R: np.ndarray,
                                   f_max: float,
                                   kappa_fraction: float = 0.5,
                                   n_lipschitz_samples: int = 500,
                                   n_alpha_steps: int = 80):
    """
    Compute all offline QIH-NMPC parameters.

    Args:
        Q:       (13×13) state weight matrix
        R:       (4×4)   input weight matrix
        f_max:   maximum thrust per motor [N]
        kappa_fraction:      fraction of stability margin to use (0 < · < 1)
        n_lipschitz_samples: number of Monte Carlo samples per alpha candidate
        n_alpha_steps:       number of alpha scale factors to try

    Returns:
        P_full:  (13×13) terminal cost matrix (embedded from 12×12 Lyapunov P)
        alpha:   terminal region radius  (x^T P x <= alpha)
        kappa:   exponential decay rate
        K_red:   (4×12)  auxiliary LQR gain on the reduced state
    """

    # ── Step 1: Continuous-time linearization at hover ──────────
    A_c, B_c = get_hover_linearization()

    # ── Step 2: Remove qw → 12-state reduced system ────────────
    A_red = A_c[np.ix_(IDX_KEEP, IDX_KEEP)]     # (12×12)
    B_red = B_c[IDX_KEEP, :]                     # (12×4)
    Q_red = Q[np.ix_(IDX_KEEP, IDX_KEEP)]        # (12×12)

    # ── Step 3: CARE → auxiliary gain K ─────────────────────────
    #   CARE:  A^T P_care + P_care A - P_care B R^{-1} B^T P_care + Q = 0
    #   Gain:  K = -R^{-1} B^T P_care
    P_care = solve_continuous_are(A_red, B_red, Q_red, R)
    K_red  = -inv(R) @ B_red.T @ P_care           # (4×12)

    # ── Step 4: Verify closed-loop stability ────────────────────
    A_K = A_red + B_red @ K_red
    eigvals_AK = np.linalg.eigvals(A_K)
    max_real_eig = np.max(np.real(eigvals_AK))
    assert max_real_eig < 0, \
        f"A_K not Hurwitz! max Re(eig) = {max_real_eig:.6f}"

    print(f"  A_K eigenvalues (real parts): "
          f"{np.sort(np.real(eigvals_AK))[:5]} ...")

    # ── Step 5: Choose kappa ────────────────────────────────────
    #   Must satisfy 0 < kappa < -max Re(eig(A_K))
    #   Larger kappa → faster guaranteed decay, but smaller terminal set.
    stability_margin = abs(max_real_eig)
    kappa = kappa_fraction * stability_margin

    print(f"  Stability margin   = {stability_margin:.4f}")
    print(f"  kappa selected     = {kappa:.4f}  "
          f"({kappa_fraction:.0%} of margin)")

    # ── Step 6: Modified Lyapunov equation → terminal cost P ────
    #
    #   (A_K + kappa I)^T P + P (A_K + kappa I) = -(Q + K^T R K)
    #
    #   scipy solves:  A X + X A^H = Q_rhs
    #   so we pass A = (A_K + kappa I)^T,  Q_rhs = -(Q + K^T R K)
    #
    Q_star = Q_red + K_red.T @ R @ K_red
    A_K_kappa = A_K + kappa * np.eye(N_RED)
    P_red = solve_continuous_lyapunov(A_K_kappa.T, -Q_star)

    # Symmetrize (numerical hygiene)
    P_red = 0.5 * (P_red + P_red.T)

    eigvals_P = np.linalg.eigvalsh(P_red)
    assert np.all(eigvals_P > 0), \
        f"P_red not positive definite! min eigval = {eigvals_P.min():.6e}"

    # ── Step 7: Alpha from input constraints ────────────────────
    #
    #   Under kappa_f: u = u_hover + K x_red
    #   Physical bounds: 0 <= u_i <= f_max
    #   → deviation bounds: -f_hover <= K_i x <= f_max - f_hover
    #
    #   For x^T P x <= alpha, the maximum of |K_i x| is:
    #       max_{x^T P x <= alpha} |K_i x| = sqrt(alpha * K_i^T P^{-1} K_i)
    #
    #   Require this <= du_bound for each motor.
    #
    du_upper = f_max - f_hover           # positive headroom
    du_lower = f_hover                   # negative headroom (0 - f_hover = -f_hover)
    du_bound = min(du_upper, du_lower)   # tighter side

    P_red_inv = inv(P_red)
    alpha_input = np.inf
    for i in range(NU):
        k_i = K_red[i, :]
        gain_sq = k_i @ P_red_inv @ k_i          # scalar
        alpha_i = (du_bound ** 2) / gain_sq
        alpha_input = min(alpha_input, alpha_i)

    print(f"  Alpha (input limit) = {alpha_input:.4f}")

    # ── Step 8: Verify Lipschitz condition via sampling ─────────
    #
    #   QIH requires:  ||phi(x)|| <= L_phi ||x||  inside Omega_alpha
    #   where phi(x) = f(x, Kx + u_hover) - A_K x   (nonlinear residual)
    #
    #   Required bound:  L_phi < kappa * lambda_min(P) / ||P||_2
    #
    lambda_min_P = eigvals_P.min()
    norm_P = norm(P_red, 2)                       # spectral norm
    L_phi_max = kappa * lambda_min_P / norm_P

    print(f"  L_phi_max allowed  = {L_phi_max:.6f}")

    # CasADi function for nonlinear dynamics
    model = create_model()
    f_func = ca.Function('f_func', [model.x, model.u], [model.f_expl_expr])

    # Cholesky for sampling from the ellipsoid {x : x^T P x <= alpha}
    #   P = L L^T  →  x = L^{-T} z  with ||z|| <= sqrt(alpha)
    L_chol = cholesky(P_red, lower=True)
    L_inv_T = inv(L_chol).T                       # (12×12)

    alpha = alpha_input
    alpha_found = False

    for scale in np.linspace(1.0, 0.01, n_alpha_steps):
        test_alpha = alpha_input * scale
        sqrt_alpha = np.sqrt(test_alpha)
        max_L_phi = 0.0

        for _ in range(n_lipschitz_samples):
            # Sample uniformly inside the unit ball in R^12
            z = np.random.randn(N_RED)
            z /= norm(z)
            r = np.random.rand() ** (1.0 / N_RED)   # uniform in ball
            z *= sqrt_alpha * r

            # Transform to ellipsoid: x_red = L^{-T} z
            x_red = L_inv_T @ z

            # Reconstruct full 13-state (qw = 1 at hover)
            x_full = np.zeros(NX)
            x_full[IDX_KEEP] = x_red
            x_full[6] = 1.0                          # qw

            # Auxiliary control (clipped to physical bounds)
            u_dev = K_red @ x_red
            u_full = np.clip(u_dev + f_hover, 0.0, f_max)

            # Evaluate nonlinear dynamics
            f_eval = np.array(f_func(x_full, u_full)).flatten()
            f_red = f_eval[IDX_KEEP]

            # Nonlinear residual
            phi_x = f_red - A_K @ x_red

            x_norm = norm(x_red)
            if x_norm > 1e-8:
                L_phi_sample = norm(phi_x) / x_norm
                max_L_phi = max(max_L_phi, L_phi_sample)

        if max_L_phi < L_phi_max:
            alpha = test_alpha
            alpha_found = True
            print(f"  L_phi sampled      = {max_L_phi:.6f}  "
                  f"< {L_phi_max:.6f}  ✓  (scale={scale:.3f})")
            break

    if not alpha_found:
        print("  WARNING: Lipschitz condition not satisfied at any alpha!")
        print("  Using smallest tested alpha (conservative).")
        alpha = alpha_input * 0.01

    # ── Step 9: Embed P_red (12×12) → P_full (13×13) ───────────
    P_full = np.zeros((NX, NX))
    P_full[np.ix_(IDX_KEEP, IDX_KEEP)] = P_red
    P_full[6, 6] = Q[6, 6]               # qw: small weight, not from Lyapunov

    # ── Diagnostics ─────────────────────────────────────────────
    print("\n=== QIH-NMPC Offline Computation Complete ===")
    print(f"  kappa              = {kappa:.4f}")
    print(f"  alpha              = {alpha:.6f}")
    print(f"  P_red diag         = {np.diag(P_red)}")
    print(f"  P_care diag        = {np.diag(P_care)}")
    print(f"  Q_red diag         = {np.diag(Q_red)}")
    print(f"  Ratio P_lyap/Q     = "
          f"{np.diag(P_red) / np.diag(Q_red)}")
    print(f"  P_full[6,6] (qw)  = {P_full[6,6]} (from Q, not Lyapunov)")
    print("=" * 48)

    return P_full, alpha, kappa, K_red


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    Q = np.diag([
        80., 80., 120.,            # position
        10., 10.,  15.,            # velocity
        10., 120., 120., 80.,      # quaternion
         1.,   1.,   1.            # angular rate
    ])
    R = np.diag([0.1, 0.1, 0.1, 0.1])

    P, alpha, kappa, K = compute_qih_offline_parameters(
        Q, R, f_max=3.0 * f_hover
    )
    print(f"\nTerminal set radius alpha = {alpha:.6f}")
    print(f"For comparison, x = [0,0,0.1, ...] gives x^T P x ≈ "
          f"{0.1**2 * P[2,2]:.2f}  (10 cm offset in z alone)")
