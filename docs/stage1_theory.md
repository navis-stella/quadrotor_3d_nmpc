# Stage 1 — Terminal-Cost-Only NMPC: Theory

Implementation: `ocp_config_basic.py`, `simulate_basic.py`
Terminal cost: `W_e = Q` (same weight matrix as the stage cost), no terminal constraint.

## 1. Starting point: Zero-Terminal-Constraint MPC

The textbook route to a *provably stable* MPC scheme starts from the zero-terminal-constraint
formulation. At each time `t`, given the current state `x(t)`, solve:

```
minimize_u(·|t)   J(x(t), ū(·;t)) = ∫[t, t+T] L(x̄(τ;t), ū(τ;t)) dτ

subject to:
    ẋ = f(x̄, ū)               (system dynamics)
    x̄(t;t) = x(t)             (initial condition)
    ū(τ;t) ∈ U,   ∀τ ∈ [t, t+T]
    x̄(τ;t) ∈ X,   ∀τ ∈ [t, t+T]
    x̄(t+T;t) = 0              (zero-terminal constraint)
```

Driving the predicted terminal state exactly to the origin removes the need to reason about
what happens *beyond* the finite horizon `T` — the cost-to-go past `t+T` is exactly zero because
the state is exactly zero. This is what makes the standard Lyapunov argument for asymptotic
stability go through: the optimal cost `J*(x(t))` is a valid Lyapunov function, and it strictly
decreases along the closed loop under the usual receding-horizon argument.

## 2. Why it's too tight

In practice, the equality constraint `x̄(t+T;t) = 0` is very restrictive for a nonlinear plant:

- It is a **hard constraint on the terminal state**, not a soft cost — the solver must find an
  admissible input trajectory that drives the full nonlinear model *exactly* to the origin
  within `T` seconds, from whatever perturbed state the plant is currently in.
- For a 13-state quadrotor with box-constrained motor thrusts, this shrinks the feasible set of
  the OCP sharply. Any short prediction horizon (needed to keep the QP/NLP solvable in real time
  under `SQP_RTI`) makes the constraint infeasible for anything but small perturbations near
  hover.
- Recovering from disturbances, re-initialization, or aggressive setpoint changes routinely
  produces infeasible QPs — which is unacceptable for a closed-loop controller running online.

This is the standard motivation in the MPC stability literature for relaxing the terminal
equality constraint into something softer: a **terminal cost** (this stage) or a **terminal
cost + terminal set** (Stage 2, quasi-infinite horizon).

## 3. Stage 1 relaxation: terminal cost only, `W_e = Q`

Stage 1 drops the terminal constraint entirely and instead penalizes the terminal state with the
same quadratic weight used in the running cost:

```
J(x(t), ū(·;t)) = ∫[t, t+T] L(x̄, ū) dτ  +  x̄(t+T;t)^T · Q · x̄(t+T;t)
```

with no constraint on `x̄(t+T;t)` beyond the usual state bounds `X`. This is exactly the
`NONLINEAR_LS` cost with `W_e = Q` used in `ocp_config_basic.py`.

### What this buys — and what it doesn't

- **No formal asymptotic stability guarantee.** A soft terminal penalty is not a certified
  control Lyapunov function for the nonlinear plant: nothing here proves the value function
  strictly decreases toward zero. `Q` was chosen for tracking performance, not for a Lyapunov
  argument.
- **Practical stability instead.** Empirically (confirmed in `simulate_basic.py`), the closed
  loop converges to a small neighborhood of the reference rather than the reference exactly — a
  residual steady-state error remains. This is the textbook symptom of terminal-cost-only NMPC:
  stability *in practice*, not *by construction*.
- **Feasibility is much easier to maintain** than under the hard zero-terminal constraint,
  since there's no equality constraint to satisfy at the horizon — only the running cost and the
  state/input bounds.

### Why this is still a reasonable baseline

The residual error is small when the horizon `T` is long relative to the plant's settling time,
because a longer horizon makes the terminal-cost approximation of the infinite-horizon
cost-to-go more accurate even without a formal certificate. Stage 1 is deliberately the
*baseline* against which Stage 2's formal guarantees (quasi-infinite horizon, DARE terminal
cost) are compared — same plant, same cost structure, only the terminal treatment differs.

## 4. Summary: what Stage 2 fixes

| | Terminal treatment | Guarantee |
|---|---|---|
| Zero-terminal-constraint MPC | Hard equality `x̄(t+T) = 0` | Asymptotic stability, but frequently infeasible |
| **Stage 1 (this doc)** | Soft terminal cost `W_e = Q` | Practical stability only, residual offset |
| Stage 2.1 — Quasi-infinite horizon | Terminal cost `P_lyap` + terminal *set* `Ω_α` | Asymptotic stability + recursive feasibility |
| Stage 2.2 — DARE terminal cost | Terminal cost `P_lqr` (no terminal set) | Asymptotic stability for sufficiently long horizon |

See `docs/stage2_theory.md` for the quasi-infinite horizon and DARE derivations.
