# Stage 2 — Quasi-Infinite Horizon & DARE Terminal Cost: Theory

Reference: Chen & Allgöwer, 1998 (continuous-time quasi-infinite horizon NMPC).

## 2.1 Quasi-Infinite Horizon NMPC

Implementation: `compute_qih_params.py` (offline), `ocp_config_qih.py`, `simulate_qih.py`

### Goal

Relax the zero-terminal-constraint of `docs/stage1_theory.md` §1 *without* falling back to an
uncertified terminal cost. Achieve **certified** stability with a finite prediction horizon by
approximating infinite-horizon behavior near the terminal set.

### Approach

Replace the hard terminal equality constraint with:
1. A **terminal region constraint** `x̄(t+T;t) ∈ X_α^f ⊆ X` instead of `x̄(t+T;t) = 0`.
2. A **local auxiliary controller** `k^loc(x) = Kx`, used only in the offline analysis — never
   applied online — to certify that the closed loop *could* be stabilized from anywhere inside
   the terminal region.
3. A **terminal cost** `F(x) = ‖x‖²_P` that acts as a control Lyapunov function (CLF) for the
   terminal region.

### OCP formulation

```
minimize_u(·|t)  J(x(t), ū(·;t))
   = ∫[t, t+T] L(x̄(τ;t), ū(τ;t)) dτ + F(x̄(t+T;t))

subject to:
    ẋ = f(x̄, ū)
    x̄(t;t) = x(t)
    ū(τ;t) ∈ U,        ∀τ ∈ [t, t+T]
    x̄(τ;t) ∈ X,        ∀τ ∈ [t, t+T]
    x̄(t+T;t) ∈ X_α^f ⊆ X     (terminal region, replaces zero-terminal constraint)
```

**Standing assumptions:** `L(x,u)` and `F(x)` are continuous and positive definite w.r.t.
`(0,0)` — `L(0,0)=0`, `L(x,u)>0` otherwise, and likewise for `F`; `L(x,u)` is radially
unbounded. In the implementation, quadratic costs are used throughout:

```
L(x,u) = ‖x‖²_Q + ‖u‖²_R,   F(x) = ‖x‖²_P,    Q, R, P ≻ 0
```

with the plant linearized at the origin (hover): `ẋ = Ax + Bu`, `A = ∂f/∂x(0,0)`,
`B = ∂f/∂u(0,0)`. The terminal region is the sublevel set
`X_α^f = { x : ‖x‖²_P ≤ α }` for some `α > 0` still to be determined, together with the local
gain `K` and the shape matrix `P`.

### Design procedure (lecture) → implementation (`compute_qih_params.py`)

**Step 1 — Feedback gain `K` such that `A_K := A + BK` is Hurwitz.**
The lecture leaves `K` open to any stabilizing choice; the implementation uses the
**continuous algebraic Riccati equation (CARE)** to get the LQR-optimal gain, which is a
convenient and standard way to satisfy this requirement:

```
A^T P_care + P_care A − P_care B R⁻¹ Bᵀ P_care + Q = 0
K = −R⁻¹ Bᵀ P_care
```

Before this step, the model is reduced from 13 to **12 states** by removing the scalar
quaternion component `qw`: at hover, `qw` is uncontrollable (fixing the other three quaternion
components and the norm constraint pins `qw`, so its linearized dynamics contribute no usable
control direction), and keeping it in would make the linearization rank-deficient for the CARE
solve. `P_care` and `K` are computed on this reduced 12-state system; `qw` is re-inserted later
with its own (non-Lyapunov) weight from `Q`.

The Hurwitz property is checked explicitly (`max Re(eig(A_K)) < 0`) rather than assumed.

**Step 2 — Choose `κ` and solve the modified Lyapunov equation for `P`.**
Pick `κ` satisfying `0 < κ < −max Re(λ(A_K))` (implementation: a fixed fraction,
`kappa_fraction`, of the stability margin `|max Re(λ(A_K))|`), ensuring `A_K + κI` is still
Hurwitz. Then solve:

```
(A_K + κI)ᵀ P + P (A_K + κI) = −Q*,     Q* = Q + KᵀRK
```

for `P ≻ 0`. This is **not** the same equation as the CARE in Step 1 — this is the key
theoretical distinction worth keeping straight:

- `P_care` (Step 1) is the infinite-horizon LQR cost-to-go for the *linear* system.
- `P` here (Step 2, called `P_lyap` in code) comes from a **modified** Lyapunov equation that
  bakes in an explicit exponential decay rate `κ`. This makes `P_lyap ≻ P_care` — a strictly
  stronger requirement, which in turn **shrinks** the terminal set found in Steps 3–4.

The payoff for this stronger `P` is that `V_f(x) = xᵀPx` satisfies `dV_f/dt ≤ −κ V_f(x)` under
the auxiliary controller inside the terminal region — an *exponential* Lyapunov decrease, which
is what makes the quasi-infinite horizon stability proof go through (a plain DARE/CARE `P` only
gives a non-strict Lyapunov decrease and is not sufficient by itself — see §2.2 below on why the
DARE variant needs a longer horizon to compensate).

**Step 3 — Initial terminal region size `α₁` from input constraints.**
Find the largest `α₁ > 0` such that the auxiliary control law keeps the input inside its
physical bounds for every state in the candidate terminal set:

```
X_{α₁}^f = { x : xᵀPx ≤ α₁ },    Kx ∈ U   ∀x ∈ X_{α₁}^f
```

Implementation: motor thrust is a *deviation* from hover, `u = u_hover + Kx`, bounded by
`0 ≤ u_i ≤ f_max`, i.e. `−f_hover ≤ (Kx)_i ≤ f_max − f_hover`. For a fixed `α`, the worst-case
deviation on channel `i` over the ellipsoid `{x : xᵀPx ≤ α}` has closed form:

```
max_{xᵀPx ≤ α} |K_i x| = √(α · K_i P⁻¹ K_iᵀ)
```

so each motor gives an upper bound on `α`; `α₁` is the minimum (tightest) over all four motors.

**Step 4 — Refine `α` via the Lipschitz bound on the nonlinear residual.**
Define the nonlinear residual between the true plant and its linearization under the auxiliary
controller:

```
φ(x) = f(x, Kx) − A_K x
```

and its local Lipschitz-type bound over the candidate region:

```
L_φ = sup { ‖φ(x)‖ / ‖x‖  :  x ∈ X_α, x ≠ 0 }
```

The terminal-region certificate requires:

```
L_φ ≤ κ · λ_min(P) / ‖P‖₂
```

Shrink `α` (from `α₁` downward) until this holds. The lecture states this as a supremum over
the region; the implementation approximates it by **Monte Carlo sampling** — for each candidate
`α` (tried on a decreasing grid from `α₁`), it draws random points uniformly inside the
ellipsoid `{x : xᵀPx ≤ α}` (via a Cholesky transform `x = L⁻ᵀz` from the unit ball), evaluates
the true nonlinear dynamics via CasADi at each sample, and takes the largest observed
`‖φ(x)‖ / ‖x‖` as an empirical estimate of `L_φ`. The first `α` (largest to smallest) for which
the sampled `L_φ` satisfies the bound is accepted; if none does even at the smallest step, the
smallest tested `α` is used as a conservative fallback.

Note this is an **empirical**, not analytic, certificate: the sampling-based `L_φ` is a
lower bound on the true supremum, so the resulting `α` is only as trustworthy as the sample
density (`n_lipschitz_samples`) and grid resolution (`n_alpha_steps`) used.

### Putting it back together

`P` (12×12) is embedded into the full 13×13 terminal cost matrix, with `qw`'s entry taken
directly from the running-cost weight `Q[6,6]` rather than from the Lyapunov solve (since `qw`
was excluded from the reduced dynamics). The terminal set constraint `x_N ∈ Ω_α` is
**softened** in the OCP (L1+L2 slack penalties) so the solver degrades gracefully instead of
becoming infeasible under large disturbances — the certificate above is what justifies using
this particular `(P, α)` pair as the terminal ingredients in the first place, but the
implementation doesn't require hard-enforcing it exactly for every disturbance realization.

### Result

Asymptotic stability + recursive feasibility, at the cost of a more involved offline
computation and a smaller usable terminal region than a naive DARE-based terminal cost would
suggest.

---

## 2.2 DARE Terminal Cost (Stage 2.2)

Implementation: `ocp_config_dare.py`, `simulate_dare.py`

> **Note on confidence:** unlike §2.1 above, this section is reconstructed from the project
> overview rather than from the actual `ocp_config_dare.py` source or an offline
> `compute_dare_params.py` script. The high-level structure should be correct, but exact
> tuning choices (discretization step, `Q`/`R` used) may differ from what's below. Happy to
> tighten this up if you share that file.

### Motivation

Stage 2.1 buys a formal asymptotic guarantee at the price of the terminal-region machinery in
§2.1 Steps 3–4 (input-based `α₁`, Lipschitz sampling, softened terminal-set constraint in the
OCP). Stage 2.2 asks: how much of that can be dropped if we're willing to accept a weaker,
horizon-dependent guarantee instead of an unconditional one?

### Approach

Use only the **terminal cost**, `W_e = P_lqr`, computed from a **reduced-order Discrete
Algebraic Riccati Equation (DARE)** at the hover linearization — with **no terminal set
constraint** at all:

```
P_lqr = DARE(A_d, B_d, Q_red, R)   solving   
        A_dᵀ P_lqr A_d − P_lqr − A_dᵀ P_lqr B_d (R + B_dᵀ P_lqr B_d)⁻¹ B_dᵀ P_lqr A_d + Q_red = 0
```

on the discretized, `qw`-reduced hover linearization `(A_d, B_d)` (same 12-state reduction as
in §2.1, for the same rank-deficiency reason), then embedded back into the full 13×13 terminal
cost with `qw`'s entry taken from `Q`.

### Why this can still work without a terminal set

`P_lqr` is the infinite-horizon LQR cost-to-go for the linearized system, and — as in classical
LQR-as-CLF arguments — it satisfies a **non-strict** Lyapunov decrease condition under the
optimal linear feedback near the origin. Unlike §2.1's `P_lyap`, it doesn't encode an explicit
exponential decay margin `κ`, so it is a strictly weaker Lyapunov certificate on its own.

What compensates for this in practice is the prediction horizon `N`: for a **sufficiently long**
horizon, the finite-horizon optimal cost with terminal cost `P_lqr` approximates the true
infinite-horizon cost closely enough, over a large enough neighborhood of the origin, that the
value function still decreases along the closed loop — without needing to certify a specific
terminal region a priori. This is the standard "terminal-cost-only, no terminal-set" result: it
trades an unconditional guarantee (true for *any* admissible horizon, as in §2.1) for a
horizon-length-dependent one (true once `N` is long enough, without a computed bound on how
long is enough).

### Trade-off vs. Stage 2.1

| | Stage 2.1 (QIH) | Stage 2.2 (DARE) |
|---|---|---|
| Terminal cost | `P_lyap` (modified Lyapunov, exponential margin `κ`) | `P_lqr` (plain DARE) |
| Terminal set | Explicit `Ω_α`, softened in OCP | None |
| Guarantee | Asymptotic stability + recursive feasibility, any `N` | Asymptotic stability for `N` sufficiently long (no explicit bound) |
| Offline computation | CARE + modified Lyapunov + Lipschitz sampling for `α` | DARE only |
| Online complexity | Extra slack variables for terminal-set softening | None beyond terminal cost |

Stage 2.2 is the pragmatic middle ground between Stage 1 (`W_e = Q`, no certificate at all) and
Stage 2.1 (full quasi-infinite horizon certificate): it upgrades the terminal cost from an
arbitrary tracking weight to the actual LQR cost-to-go, without paying for the terminal-set
machinery.
