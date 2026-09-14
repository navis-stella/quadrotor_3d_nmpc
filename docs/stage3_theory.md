# Stage 3 — Offset-Free NMPC: Theory & Implementation

Stage 1 and Stage 2 assume the plant matches the prediction model exactly. Under that assumption, terminal-cost design (Stage 1) and terminal-cost + terminal-set design (Stage 2) are enough for stability. Stage 3 removes that assumption: the plant now carries a **constant, unmodeled disturbance** (wind, mass mismatch, CoG offset), and the question is how NMPC — which has no integral action by construction — is made to reject it exactly, not just approximately.

This document derives the three pieces that make up the fix (disturbance-augmented model, EKF, steady-state target calculator), lays out the exact per-step control-loop logic implemented in `simulate_offsetfree.py`, and cross-checks the analytical equilibrium against the closed-loop validation run in the main README.

---

## 1. Problem: Why Standard NMPC Has Offset

NMPC solves, at every step, an optimization over the **nominal** model $f(x, u)$. If the true plant is $f(x, u) + w$ for some constant disturbance $w$, the controller has no way to represent $w$ in its prediction — it only ever sees the effect of $w$ through the measured state error at the current step. There is no state in the formulation whose job is "remember and cancel a persistent error," which is precisely the definition of integral action.

Linearize near the hover reference to make this concrete. Let the true (LTI) closed loop be

$$\dot{x} = Ax + Bu + w$$

$$u = -K(x - x_{\text{ref}}) + u_{\text{ref}}$$

(the NMPC horizon behaves like this LQR-type law near the reference, since the OCP cost is quadratic and `SQP_RTI` takes one Gauss-Newton step per sample)

At steady state ($\dot{x} = 0$), and using that $(x_{\text{ref}}, u_{\text{ref}})$ is itself an equilibrium of the *nominal* system ($A x_{\text{ref}} + B u_{\text{ref}} = 0$, true at level hover with $u_{\text{ref}} = f_{\text{hover}}$):

$$(A - BK)(x^* - x_{\text{ref}}) = -w$$

$$x^* - x_{\text{ref}} = -(A - BK)^{-1} w$$

This is the textbook symptom noted going into this stage: **a residual offset proportional to the disturbance and inversely related to closed-loop gain — never zero for $w \neq 0$.**

For this quadrotor, the offset shows up asymmetrically between position and attitude:

- **Translational dynamics are a pure double integrator** ($\ddot{p} = f(\text{attitude, thrust})/m + w_f$, no term restores $p$ to any particular value). The only steady-state requirement is $\dot{v} = 0$, i.e. the *attitude and thrust* must supply exactly enough force to cancel gravity and $w_f$ — this says nothing about *where* the position converges. Position offset is therefore driven entirely by how tightly the receding-horizon law's implicit feedback gain $K$ couples position error to attitude command — it can be made small with aggressive $Q_{\text{pos}}$ weighting, but never exactly zero for $w \neq 0$.
- **Attitude has no such freedom.** Whatever tilt is needed to null $\dot{v}$ under $w_f$ is exactly the tilt the closed loop must converge to, or velocity keeps drifting. This is why the residual offset in the Stage 3 validation run is far more visible in roll/pitch than in position — the controller was already forced very close to the correct tilt by physics; it just doesn't know it's the *correct* tilt versus a tracking error to fight.

## 2. Disturbance-Augmented Plant Model

The disturbance is modeled as a 6-dimensional constant bias, split by where it physically enters the dynamics:

$$d = [d_{f_x},\; d_{f_y},\; d_{f_z},\; d_{\tau_x},\; d_{\tau_y},\; d_{\tau_z}]$$

| Component | Units | Description |
|---|---|---|
| $d_{f_x},\; d_{f_y},\; d_{f_z}$ | N | Force disturbance, **world** frame (wind, $\Delta m \cdot g$) |
| $d_{\tau_x},\; d_{\tau_y},\; d_{\tau_z}$ | N·m | Torque disturbance, **body** frame (CoG offset → parasitic torque) |

$$\dot{d} = 0 \qquad \text{(constant-disturbance assumption)}$$

Injected into the nominal 13-state dynamics:

$$\dot{v} = \frac{f_{\text{total}}}{m} \cdot R(q)_{:,2} - g \cdot e_z + \frac{d_f}{m} \qquad \text{(translational, world frame)}$$

$$\dot{q} = 0.5 \cdot q \otimes [0,\; p,\; q,\; r] \qquad \text{(unaffected by disturbance)}$$

$$I \cdot \dot{\omega} = \tau(u) - \omega \times (I \cdot \omega) + d_\tau \qquad \text{(rotational, body frame)}$$

The codebase reuses **one** dynamics builder (`_build_f_expl(s, d_fx=0, ..., d_tz=0)`, defaulting to zero) for three separate acados/CasADi objects, so all three variants are provably the same physics:

| Variant | Disturbance role | State dim | Used by |
|---|---|---|---|
| `create_model()` | $d = 0$ (nominal) | $N_x=13$ | Stages 1–2 |
| `create_disturbance_model()` / `create_disturbance_plant()` | $d$ = runtime parameter `model.p` | $N_x=13$, $p \in \mathbb{R}^6$ | MPC solver ($p = \hat{d}$) and plant simulator ($p = d_{\text{true}}$) |
| `get_augmented_dynamics_casadi()` | $d$ = extra **states** | $N_z = N_x + N_d = 19$ | EKF only |

This separation matters: the MPC and the plant both use the *same* 13-state model with $d$ as an external parameter (so the state dimension the solver optimizes over never changes), while only the EKF needs $d$ promoted to an estimated state, since that's the only place a disturbance *estimate* is produced.

## 3. State & Disturbance Estimation — Augmented EKF

**Augmented state:** $z = [x(13);\; d(6)] \in \mathbb{R}^{19}$. **Measurement model:** $y = Hz = x$, with $H = [I_{13} \mid 0_{13 \times 6}]$ — the project's measurement assumption is that $p, v, q, \omega$ are all directly observed (standing in for a fused IMU+GPS/VIO front end that Stage 3 doesn't model). Since $x$ is fully measured, only $d$ is genuinely being *estimated*.

**Predict** (continuous-discrete EKF, called once per step with the applied input):

$$\hat{z}_{k+1|k} = \text{RK4}\bigl(f_{\text{aug}}(\hat{z}_{k|k},\, u_k),\; T_s\bigr)$$

(nonlinear plant states integrated; disturbance states pass through unchanged since $\dot{d} = 0$)

$$F_c = \left.\frac{\partial f_{\text{aug}}}{\partial z}\right|_{(\hat{z},\, u)} \qquad \text{(19×19 continuous Jacobian, CasADi AD)}$$

$$F_d \approx I + F_c \cdot T_s \qquad \text{(first-order discretization)}$$

$$P_{k+1|k} = F_d\, P_{k|k}\, F_d^\top + Q_{\text{aug}} \cdot T_s$$

**Update** (called before the OCP is solved, using the new measurement):

Innovation (with quaternion sign-flip guard: if $y_{q_w}$ and $\hat{z}_{q_w}$ disagree in sign, flip $y$'s quaternion — same rotation, avoids a spurious large innovation from the $q / {-q}$ double cover):

$$e = y - H\hat{z}$$

$$S = H P H^\top + R$$

$$K = P H^\top S^{-1} \qquad \text{(19×13 Kalman gain)}$$

$$\hat{z} \leftarrow \hat{z} + K\, e$$

$$P \leftarrow (I - KH)\, P\, (I - KH)^\top + K R K^\top \qquad \text{(Joseph form, numerically stable)}$$

**Why does $\hat{d}$ update at all, if $x$ is fully measured?** The innovation $e$ only ever multiplies rows 1–13 of $K$ directly into the $x$-block of $\hat{z}$. The correction to $\hat{d}$ (rows 14–19 of $K$) exists only because of the **cross-covariance** $P_{xd}$ built up during the predict step: the Jacobian $F_c$ has nonzero entries coupling $d$ into $\dot{x}$ (e.g. $\partial\dot{v}/\partial d_f = I/m$), so uncertainty in $d$ shows up as *predicted* uncertainty in $x$. When the measurement then corrects $x$ beyond what the model alone anticipated, the same Kalman gain — through $P_{xd}$ — pulls $\hat{d}$ in the direction that would have explained that error. This is exactly the mechanism that gives an EKF (or any observer built this way) integral-action-like behavior: a persistent, systematic prediction error gets attributed to the disturbance state until it explains the error away.

**Default tuning** (from `get_default_ekf_tuning()`):

| Block | Values | Rationale |
|---|---|---|
| $Q_x$ (state process noise) | small (`1e-3`–`1e-2`) | model is accurate; just prevents covariance collapse |
| $Q_d$ (disturbance process noise) | `[0.1,0.1,0.1, 0.01,0.01,0.01]` | forces assumed to vary faster than torques in practice; larger $Q_d$ → faster $\hat{d}$ adaptation but noisier estimate |
| $R$ (measurement noise) | small, position/attitude tighter than velocity/rate | reflects the "clean full-state access" assumption for this stage |

## 4. Steady-State Target Calculator

**Why it's needed.** Without it, the MPC reference stays at $(x_{\text{ref}}, u_{\text{hover}})$ — position at the target, attitude level, thrust at hover. Under a lateral disturbance, *both cannot hold simultaneously*: staying level means drifting off position; holding position means tilting. Feeding the raw $\hat{d}$ into the prediction model alone doesn't resolve this tension — the OCP would still be penalizing deviation from an unreachable reference. `ss_target.py` computes the actual reachable equilibrium $(x_s, u_s)$ under $\hat{d}$, so the OCP's cost is zero at a point the plant can actually sit at.

**Derivation.** At equilibrium, $v = 0$, $\omega = 0$, so all rate terms and gyroscopic couplings vanish, leaving a static force/torque balance:

**(1) Force balance (world frame) → thrust magnitude & direction.**

$$T_s \cdot z_b + d_f = m g\, e_3$$

$$T_s \cdot z_b = [0,\; 0,\; mg] - d_f \;=:\; F_{\text{req}}$$

$$T_s = \|F_{\text{req}}\| \qquad \text{(required total thrust)}$$

$$z_b = \frac{F_{\text{req}}}{T_s} \qquad \text{(required body z-axis, expressed in world frame)}$$

**(2) Attitude from $z_b$ + desired yaw.** $z_b$ alone fixes two degrees of freedom of the rotation (which way "up" points in the body frame); yaw $\psi$ (taken from the reference quaternion) fixes the third. A body frame consistent with both is built directly (Gram–Schmidt against a yaw-aligned candidate $x$-axis):

$$x_c = [\cos\psi,\; \sin\psi,\; 0] \qquad \text{(candidate x-axis in the xy-plane, at reference yaw)}$$

$$y_b = \frac{z_b \times x_c}{\|z_b \times x_c\|} \qquad \text{(falls back to } [-\sin\psi,\; \cos\psi,\; 0] \text{ if } z_b \text{ is nearly vertical)}$$

$$x_b = y_b \times z_b \qquad \text{(completes a right-handed, orthonormal frame)}$$

$$R_s = [x_b \mid y_b \mid z_b] \qquad \text{(columns — body axes expressed in world frame)}$$

$$q_s = \text{rotmat\_to\_quat}(R_s) \qquad \text{(Shepperd's method; } q_w > 0 \text{ hemisphere enforced)}$$

**(3) Torque balance (body frame, $\omega = 0$ → no gyroscopic term) → motor mixer inversion.**

$$\tau(u_s) + d_\tau = 0 \quad \Longrightarrow \quad \tau(u_s) = -d_\tau$$

$$\begin{bmatrix} T_s \\ -d_{\tau_x} \\ -d_{\tau_y} \\ -d_{\tau_z} \end{bmatrix} = \underbrace{\begin{bmatrix} 1 & 1 & 1 & 1 \\ 0 & -L & 0 & L \\ -L & 0 & L & 0 \\ -c_\tau & c_\tau & -c_\tau & c_\tau \end{bmatrix}}_{M} \begin{bmatrix} f_1 \\ f_2 \\ f_3 \\ f_4 \end{bmatrix}$$

$$u_s = M^{-1} [T_s,\; -d_{\tau_x},\; -d_{\tau_y},\; -d_{\tau_z}]^\top, \quad \text{then clipped to } [0,\; f_{\max}]$$

**(4) Assemble the equilibrium state.** Position is free at equilibrium (per §1, the translational dynamics impose no constraint on *where* — only on $\dot{v}=0$), so the achievable position target is simply the reference position:

$$x_s = [p_{\text{ref}};\; 0_3;\; q_s;\; 0_3], \qquad u_s \text{ as computed above}$$

This is the whole reason offset-free tracking is exact rather than approximate here: the equilibrium the plant *can* reach is computed in closed form from $\hat{d}$, rather than approached asymptotically by penalizing the gap to an unreachable point.

## 5. OCP Modification

Two runtime updates are made every control step, on top of the unchanged Stage 2 OCP structure:

```python
for k in 0..N:   ocp_solver.set(k, 'p', d̂)                 # disturbance parameter
                                                             # (same d̂ at every node —
                                                             #  ḋ=0 assumption)
for k in 0..N-1: ocp_solver.set(k, 'yref',   [x_s; u_s])   # stage reference
ocp_solver.set(N, 'yref_e', x_s)                            # terminal reference
```

The **terminal cost is untouched**: $W_e = P_{\text{lqr}}$, the same reduced-order DARE solution from Stage 2, computed once offline at the nominal hover linearization ($d$ does not enter the DARE). This is a deliberate separation of concerns: the terminal cost's job is to certify stability of the *tracking error* dynamics near equilibrium; the *offset* is corrected entirely through the prediction model ($p = \hat{d}$) and the moving reference $(x_s, u_s)$ — not by re-deriving a disturbance-dependent terminal weight.

| Quantity | Stage 2 | Stage 3 |
|---|---|---|
| State dim | $N_x = 13$ | $N_x = 13$ (unchanged — $d$ is a parameter, not a state) |
| Disturbance dim | — | $N_d = 6$ |
| EKF augmented dim | — | $N_z = N_x + N_d = 19$ |
| `model.p` | none | $d \in \mathbb{R}^6$, set from $\hat{d}$ (MPC) or $d_{\text{true}}$ (plant) |
| Reference | fixed $(x_{\text{ref}}, u_{\text{hover}})$ | moving $(x_s, u_s)$, recomputed from $\hat{d}$ every step |
| Terminal cost | $P_{\text{lqr}}$ (DARE) | $P_{\text{lqr}}$ (DARE) — identical |

## 6. Closed-Loop Implementation Logic

Per control step $k$, in the order `simulate_offsetfree.py` executes them:

1. **Measure** $y = x_k$ (full-state feedback per the project's measurement assumption).
2. **EKF update($y$)** — always runs, whether or not offset-free correction is active, so $\hat{d}$ is already converged by the time it's needed. → yields $\hat{x}$, $\hat{d}$.
3. **Activation switch.**
   - $t \geq T_{\text{ACTIVATE}}$: compute $(x_s, u_s) = \text{ss\_target}(x_{\text{ref}}, \hat{d})$; inject $\hat{d}$ as the OCP parameter at every node; set the reference to $(x_s, u_s)$ — **offset-free ON**.
   - $t < T_{\text{ACTIVATE}}$: inject $d = 0$; keep the reference at $(x_{\text{ref}}, u_{\text{hover}})$ — **standard MPC**, the Stage 2 baseline, for direct comparison inside the same run.
4. **Solve the OCP** with $x_0$ fixed to the measured state ($\text{lbx} = \text{ubx} = x_k$), `SQP_RTI` (one Gauss-Newton/QP iteration per call — real-time iteration).
5. **Apply** $u_k$ = the first control action from the solution.
6. **EKF predict($u_k$)** — propagate $\hat{z}, P$ forward using the *applied* input (must happen after the input is known, before the next update).
7. **Step the true plant** forward one sample with the *true* disturbance $d_{\text{true}}(t)$ (unknown to the controller/EKF) via the disturbance-parameterized `AcadosSimSolver`; normalize the quaternion.

```mermaid
flowchart TD
    A[Measure y = x_k] --> B["EKF update(y)\nx̂, d̂"]
    B --> C{t ≥ T_ACTIVATE ?}
    C -- "yes: OFFSET-FREE" --> D["ss_target(x_ref, d̂) → x_s, u_s\nset OCP param p = d̂ (all nodes)\nset yref = (x_s, u_s)"]
    C -- "no: STANDARD MPC" --> E["set OCP param p = 0\nset yref = (x_ref, u_hover)"]
    D --> F[Solve OCP: SQP_RTI\nx0 = x_k]
    E --> F
    F --> G[Apply u_k]
    G --> H["EKF predict(u_k)\nRK4 + covariance propagation"]
    G --> I["Plant step (true d_true(t))\nAcadosSimSolver, ERK4"]
    I --> J[Normalize quaternion]
    H --> K[k ← k+1]
    J --> K
    K --> A
```

## 7. Validation Scenario & Cross-Check

`simulate_offsetfree.py` runs both regimes in one 10 s simulation, hovering at $x_{\text{ref}} = (2, 1, 3)$ m:

| Quantity | Value |
|---|---|
| $d_{f_x}$ (wind, world x) | $0.5$ N |
| $d_{f_y}$ (wind, world y) | $-0.3$ N |
| $d_{f_z}$ (mass error, world z) | $-0.981 \times 3 = -2.943$ N ($\approx 30\%$ mass error) |
| $d_\tau$ (torque) | $0$ (no CoG offset in this run) |
| $T_{\text{ACTIVATE}}$ | $4.0$ s |
| Disturbance onset | $t = 0$ (present throughout — isolates *estimation/activation*, not *detection*, delay) |

**Analytical equilibrium** (§4, evaluated at these numbers):

$$F_{\text{req}} = [-0.5,\; -(-0.3),\; 9.81 - (-2.943)] = [-0.5,\; 0.3,\; 12.753] \;\text{N}$$

$$T_s = \|F_{\text{req}}\| \approx 12.77 \;\text{N} \qquad (\text{vs. } mg = 9.81 \;\text{N})$$

$$z_b = \frac{F_{\text{req}}}{T_s} \approx [-0.0392,\; 0.0235,\; 0.9990]$$

$$\text{tilt from vertical} = \arcsin\bigl(\|[z_{b_x},\; z_{b_y}]\|\bigr) \approx 2.6°$$

This splits, at $\psi_{\text{ref}} = 0$, into roughly pitch $\approx -2.2°$ to $-2.6°$, roll $\approx -1.3°$ to $-1.5°$ (matches the previously logged run: $\theta_{\text{eq}} \approx -2.65°$).

No torque disturbance implies $\tau(u_s) = 0$, giving a uniform mixer split:

$$u_s = M^{-1}[T_s,\; 0,\; 0,\; 0]^\top = \frac{T_s}{4}\,[1,\;1,\;1,\;1] \approx 3.19 \;\text{N per motor} \quad (\text{vs. } f_{\text{hover}} \approx 2.45 \;\text{N})$$

This matches the closed-loop result in `results/stage3_offsetfree/`:

![EKF disturbance estimation](../results/stage3_offsetfree/disturbance.png)

$\hat{d}$ converges to $[0.5,\; -0.3,\; -2.943,\; 0,\; 0,\; 0]$ within ~2 s — well before $T_{\text{ACTIVATE}}$ — confirming that the *estimator* isn't the bottleneck on activation delay; only the *correction mechanism* is gated by $T_{\text{ACTIVATE}}$ in this scenario. Torque-disturbance estimates show a transient (correlated with the aggressive early attitude maneuver, note the `1e-5` scale) and correctly decay to ~0.

![Input trajectory](../results/stage3_offsetfree/inputs.png)

After $t = 4$ s, total thrust settles at $\approx 12.8$ N and each motor at $\approx 3.19$ N — matching the analytical $T_s$ and $u_s$ above to within numerical tolerance.

![State trajectory](../results/stage3_offsetfree/states.png)

Position is already close to $x_{\text{ref}}$ under standard MPC (§1's $(A-BK)^{-1}w$ offset is small here because position is weighted heavily, $Q_{\text{pos}} = \text{diag}(80,80,120)$, relative to attitude). The offset is unambiguous in roll/pitch: the state settles near the analytically-predicted tilt *before* $T_{\text{ACTIVATE}}$ too (because that tilt is what physically nulls $\dot{v}$, regardless of whether the controller "intends" it) — what changes at $t = 4$ s is that the reference itself moves onto the `eq. target` (green dashed) instead of remaining at level hover, so the small step seen at the activation boundary is the reference catching up to where the plant already needed to be, and velocity/angular rates settle exactly to zero afterward rather than hovering near it.

## 8. Looking Ahead — Stage 4

The measurement assumption fixed here ($p, q, \omega$ directly observed; only $v$ and $d$ estimated) is deliberately the same one Stage 4 will use to compare **DOB vs. EKF vs. MHE** as disturbance/state estimators under an identical scenario — this EKF is the first data point in that comparison, not a final choice. Keeping the measurement model, disturbance scenario, and (where applicable) OCP structure identical across estimators is what makes that later comparison free of confounding variables.

## 9. Key Takeaways

- A soft cost against an unreachable reference is not the same as tracking the reachable one — the target calculator, not the disturbance parameter alone, is what makes the offset *exactly* zero rather than merely smaller.
- Full state measurement does not make disturbance estimation redundant: the disturbance is not directly measured, and the EKF's cross-covariance is what lets a fully-measured state still carry information about an unmeasured one.
- Consistent with the maglev-project lesson carried into this one: an EKF (not a linearized KF) is used because the operating point shifts with the disturbance-dependent equilibrium, not just with the reference.
