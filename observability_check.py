"""
observability_check.py — Stage 4a Local Observability at Hover
=============================================================
Verify that the linearized augmented pair (F, H) at hover is observable.

    F = ∂f_aug/∂z |_(z_hover, u_hover)     (19 × 19)
    H = ∂h/∂z                              (10 × 19, constant)

If (F, H) is observable, both estimators can (in principle) recover the
full augmented state [x(13); d(6)] from the 10-dim measurement stream
y = [p; q; ω]. This is the local, single-point sanity check behind both:

    EKF (Stage 4a):  observability ⇒ linearized covariance update converges
    MHE (Stage 4b):  needs the stronger, non-local statement — i-iIOSS on
                     the whole operating envelope — which detectability_check.py
                     certifies via LMIs (and whose weights the MHE uses).
"""

import numpy as np

from quadrotor_3d_model import (
    get_augmented_dynamics,
    get_measurement_function,
    hover_point,
)


# ─────────────────────────────────────────────────────────────────
# State labels for diagnostic output
# ─────────────────────────────────────────────────────────────────
STATE_NAMES = [
    'px',   'py',   'pz',
    'vx',   'vy',   'vz',
    'qw',   'qx',   'qy',   'qz',
    'p',    'q',    'r',
    'd_fx', 'd_fy', 'd_fz',
    'd_tx', 'd_ty', 'd_tz',
]


# ─────────────────────────────────────────────────────────────────
# Observability matrix
# ─────────────────────────────────────────────────────────────────
def build_observability_matrix(F: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Standard Kalman observability matrix:
        O = [ H; HF; HF^2; ...; HF^{n-1} ]

    Shape: (n * NY, n) where n = F.shape[0].
    Rank(O) = n  ⇔  (F, H) is observable.
    """
    n = F.shape[0]
    ny = H.shape[0]
    O = np.zeros((n * ny, n))
    HFk = H.copy()
    for k in range(n):
        O[k*ny:(k+1)*ny, :] = HFk
        HFk = HFk @ F
    return O


# ─────────────────────────────────────────────────────────────────
# Observability check
# ─────────────────────────────────────────────────────────────────
def check_observability(verbose: bool = True) -> dict:
    """
    Test full observability of the augmented pair (F, H) at hover.

    Returns:
        result: dict with keys
            'rank'         — numerical rank of O
            'n'            — full rank target (= NZ)
            'is_observable'— bool
            'sigmas'       — singular values of O
            'unobs_basis'  — null-space basis (columns), empty if observable
    """
    z_hover, u_hover = hover_point()

    _, F_func = get_augmented_dynamics()
    _, H = get_measurement_function()

    F = np.array(F_func(z_hover, u_hover))
    O = build_observability_matrix(F, H)

    # SVD-based rank with a scale-aware tolerance
    _, sigmas, Vt = np.linalg.svd(O)
    tol = max(O.shape) * sigmas[0] * np.finfo(float).eps
    rank = int(np.sum(sigmas > tol))
    n = F.shape[0]

    unobs_basis = np.zeros((n, 0))
    if rank < n:
        null_dim = n - rank
        unobs_basis = Vt[-null_dim:].T   # columns span null(O)

    if verbose:
        _print_observability_report(F, H, O, sigmas, tol, rank, n, unobs_basis)

    return {
        'rank':          rank,
        'n':             n,
        'is_observable': rank == n,
        'sigmas':        sigmas,
        'unobs_basis':   unobs_basis,
    }


def _print_observability_report(F, H, O, sigmas, tol, rank, n, unobs_basis):
    print("═" * 68)
    print("  Stage 4a — Observability check at hover")
    print("═" * 68)
    print(f"  F shape       : {F.shape}")
    print(f"  H shape       : {H.shape}")
    print(f"  O shape       : {O.shape}   (should be {n*H.shape[0]} × {n})")
    print()

    print("  Singular values of O (largest to smallest):")
    for i, sv in enumerate(sigmas):
        marker = "   ← below tol" if sv <= tol else ""
        print(f"    σ[{i:2d}] = {sv:.4e}{marker}")

    print()
    print(f"  Rank tolerance : {tol:.4e}")
    print(f"  Rank(O)        : {rank}  /  n = {n}")
    print()

    if rank == n:
        print("  ✓ (F, H) is OBSERVABLE at hover.")
        print("    → EKF can recover z = [x; d] from y.")
    else:
        print(f"  ✗ (F, H) is NOT observable — {n - rank} unobservable direction(s).")
        print()
        print("  Unobservable subspace (columns of null(O), |v| > 0.01 shown):")
        for j in range(unobs_basis.shape[1]):
            print(f"    Direction {j+1}:")
            for i, name in enumerate(STATE_NAMES):
                v = unobs_basis[i, j]
                if abs(v) > 0.01:
                    print(f"      {name:6s}  {v:+.4f}")
    print("═" * 68)


# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    check_observability()
