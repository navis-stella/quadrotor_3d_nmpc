# 3D Quadrotor NMPC

Nonlinear Model Predictive Control for a 3D quadrotor, implemented in Python using the [acados](https://docs.acados.org/) solver and [CasADi](https://web.casadi.org/) symbolic framework. The project progresses through MPC stability frameworks and extensions — from basic terminal cost to quasi-infinite horizon, offset-free tracking, and beyond — applying each method to the same quadrotor plant for direct comparison.

## Quadrotor Model

Quaternion orientation (scalar-first, body→world), `+` motor layout.

| Property | Value |
|----------|-------|
| States `x` (13) | `[px py pz, vx vy vz, qw qx qy qz, p q r]` |
| Inputs `u` (4) | `[f1 f2 f3 f4]` — individual motor thrusts [N] |
| Frames | position & linear velocity in world frame; angular rates `p q r` in body frame |
| Hover | `q = [1,0,0,0]`, `f_hover = m·g / 4 ≈ 2.45 N` |
| Torques | `τx = L(f4−f2)`, `τy = L(f3−f1)`, `τz = c_τ(−f1+f2−f3+f4)` |
| Integration | ERK4 (acados) |

Motor thrust is used as the direct control input — valid for simulation given the time-scale separation between the MPC loop and motor dynamics. Quaternion cost weights are set ≈ 4× the equivalent Euler-angle weights (small-angle: `φ ≈ 2·qx`); `qw` is only lightly weighted. The plant state is passed through `normalize_quaternion(x)` after every step to prevent norm drift.

> Note on naming: `q` in `[p q r]` is the body pitch rate — not to be confused with the quaternion components `qw, qx, qy, qz`.

## Stage Roadmap

Each stage implements a distinct MPC method or extension on top of the same quadrotor model. Working stages are tagged in git.

| Tag | Stage | Method |
|-----|-------|--------|
| `stage1` | 1 | **Basic NMPC** — terminal cost `W_e = Q`, no terminal constraint. Practical stability with residual error. |
| `stage2` | 2.1 | **Quasi-Infinite Horizon** — `W_e = P_lyap` from modified Lyapunov equation + soft terminal set `x_N ∈ Ω_α`. Asymptotic stability + recursive feasibility. |
| `stage2` | 2.2 | **DARE terminal cost** — `W_e = P_lqr` from reduced-order DARE (qw removed), no terminal constraint. |
| *upcoming* | 3 | Offset-free NMPC — augmented disturbance model + EKF. |
| *upcoming* | 4 | Obstacle-avoidance state constraints. |
| *upcoming* | 5 | Time-varying reference trajectory. |
| *upcoming* | 6 | Noisy measurements — EKF/MHE + nominal MPC. |
| *upcoming* | 7 | Stochastic MPC. |
| *upcoming* | 8 | Robust MPC. |

## File Structure

```
quadrotor_3d_nmpc/
├── quadrotor_3d_model.py       # Shared: CasADi dynamics, hover linearization,
│                               #         AcadosSimSolver plant, quaternion utilities
├── plot_utils.py               # Shared: plot_states, plot_inputs,
│                               #         plot_3d_trajectory, plot_terminal_constraint
│
├── ocp_config_basic.py         # Stage 1   — OCP setup (W_e = Q)
├── simulate_basic.py           # Stage 1   — closed-loop simulation
│
├── compute_qih_params.py       # Stage 2.1 — CARE, Lyapunov eq., terminal α, Lipschitz check
├── ocp_config_qih.py           # Stage 2.1 — OCP setup (P_lyap + soft terminal set)
├── simulate_qih.py             # Stage 2.1 — simulation + terminal constraint diagnostics
│
├── ocp_config_dare.py          # Stage 2.2 — OCP setup (W_e = P_lqr, reduced-order)
├── simulate_dare.py            # Stage 2.2 — closed-loop simulation
│
├── run.sh                      # WSL bridge launcher (Windows Git Bash → WSL2)
├── CLAUDE.md                   # Project guide for Claude Code
├── results/                    # Saved plots per stage (gitignored)
│   ├── stage1_basic/
│   ├── stage2.1_qih/
│   └── stage2.2_dare/
└── c_generated_code/           # acados auto-generated (gitignored)
```

All simulation scripts import from `quadrotor_3d_model.py` and `plot_utils.py`. OCP configuration is separated from the simulation loop for clarity and side-by-side method comparison.

## Toolchain

- **acados** (Python interface, `acados_template`) — OCP solver, `SQP_RTI`, `NONLINEAR_LS` cost
- **CasADi** 3.7.2 — symbolic dynamics and automatic differentiation
- **NumPy** 2.x, **SciPy** — offline computations (CARE / DARE, Lyapunov equations)
- **WSL2 / Ubuntu** — acados C libraries and Python venv (`~/acados_env`)
- **VS Code** on Windows for editing; execution routed through the WSL bridge

## Running

Project files live on the Windows E: drive; acados runs in WSL2. The wrapper handles the hop:

```bash
# From Windows Git Bash, in the project directory
./run.sh simulate_basic.py
./run.sh simulate_qih.py
./run.sh simulate_dare.py

# Quick environment check
./run.sh -c "import acados_template; print('ok')"
```

Or, working directly inside WSL:

```bash
source ~/acados_env/bin/activate
cd /mnt/e/WorkSpace/GitHub/quadrotor_3d_nmpc
python simulate_dare.py
```

First solver build compiles C code (~30 s); `c_generated_code/` is regenerated on subsequent runs and is gitignored. Plots are saved under `results/<stage>/`.

## MPC Methods Overview

**Basic NMPC (Stage 1).** Standard NMPC with terminal cost equal to the stage cost weight matrix (`W_e = Q`). Produces practical stability — the system converges to a neighborhood of the reference but a soft terminal penalty cannot formally guarantee asymptotic convergence. Establishes the baseline and demonstrates why terminal cost design matters.

**Quasi-Infinite Horizon (Stage 2.1).** Uses `W_e = P_lyap` computed offline from the continuous algebraic Riccati equation and a modified Lyapunov equation, together with a terminal set constraint `x_N ∈ Ω_α`. Ω_α is the largest positively invariant ellipsoidal region where the auxiliary LQR controller respects input bounds. This combination provides both asymptotic stability and recursive feasibility guarantees. The terminal constraint is softened (L1+L2 slack penalties) so the solver degrades gracefully rather than becoming infeasible under large disturbances.

**DARE terminal cost (Stage 2.2).** Terminal cost `W_e = P_lqr` from a reduced-order Discrete Algebraic Riccati Equation at the hover linearization (with `qw` removed to keep the linearization full-rank). P_lqr encodes the infinite-horizon LQR cost-to-go and acts as a Control Lyapunov Function, giving an asymptotic stability guarantee for sufficiently long horizons — without the added complexity of a terminal set constraint.

**Offset-free NMPC (Stage 3, upcoming).** Augments the model with integrating disturbance states estimated by an EKF. Eliminates the constant steady-state offset that appears when the plant differs from the MPC model (e.g., mass mismatch, persistent wind). Standard MPC lacks integral action — this is the principled fix.
