# 3D Quadrotor NMPC

Nonlinear model predictive control for a 3D quadrotor in Python, using the
[acados](https://docs.acados.org/) solver and the [CasADi](https://web.casadi.org/) symbolic
framework. The project works through MPC theory stage by stage, from terminal ingredients and
stability to offset-free control, state estimation and tracking of piece-wise constant
references. Every stage runs on the same quadrotor
model and changes one ingredient at a time, so the effect of each method can be compared
directly.

Each stage has its own document with the theory, the implementation, the scenario and the
results with plots. This README gives the overview.

## Quadrotor model

| Property | Value |
|----------|-------|
| States (13) | position, velocity (world frame), unit quaternion (scalar-first), body rates |
| Inputs (4) | individual motor thrusts, $0 \le f_i \le 3 f_{\text{hover}} \approx 7.35$ N |
| Layout | `+` configuration, hover thrust $f_{\text{hover}} = mg/4 \approx 2.45$ N |
| Disturbance (Stages 3–4) | constant force (world frame) and torque (body frame), $d \in \mathbb{R}^6$ |
| Tracking model (Stage 5) | state augmented by an artificial reference position, $`\tilde x = [x;\, p_a] \in \mathbb{R}^{16}`$, $\dot p_a = 0$ |
| Integration | ERK4 (acados) |

Equations of motion, disturbance model and quaternion conventions →
[docs/quadrotor_model.md](docs/quadrotor_model.md)

## Stages

| Tag | Stage | Method | Document |
|-----|-------|--------|----------|
| `stage1` | 1 | **Basic NMPC** — terminal cost $W_e = Q$, no terminal constraint | [stage1_basic_nmpc.md](docs/stage1_basic_nmpc.md) |
| `stage2` | 2a | **Quasi-infinite horizon** — terminal cost $P_{\text{lyap}}$ + soft terminal set $\Omega_\alpha$ | [stage2_terminal_cost.md](docs/stage2_terminal_cost.md) |
| `stage2` | 2b | **DARE terminal cost** — $P_{\text{lqr}}$ from the reduced-order DARE, no terminal set | [stage2_terminal_cost.md](docs/stage2_terminal_cost.md) |
| `stage3` | 3 | **Offset-free NMPC** — disturbance model + EKF + analytical steady-state target calculator | [stage3_offset_free.md](docs/stage3_offset_free.md) |
| `stage4` | 4a | **Partial measurement, augmented EKF** — only $`y = [p;\, q;\, \omega]`$ measured, with noise | [stage4a_ekf.md](docs/stage4a_ekf.md) |
| `stage4` | 4b | **Lyapunov MHE** — weights and horizon from an LMI detectability certificate; comparison with the EKF | [stage4b_mhe.md](docs/stage4b_mhe.md) |
| `stage5` | 5 | **Tracking NMPC with artificial reference** — piece-wise constant references, recursively feasible for any reference, converges to the closest admissible steady state (nominal plant, full state feedback) | [stage5_tracking.md](docs/stage5_tracking.md) |
| *upcoming* | 6 | Time-varying reference trajectory tracking | |
| *upcoming* | 7 | Stochastic MPC | |
| *upcoming* | 8 | Robust MPC | |
| *upcoming* | 9 | Obstacle avoidance with state constraints | |

## Key results

| Stage | Result |
|-------|--------|
| 1 | Hover-to-hover step converges in about 2 s with no visible offset, but $W_e = Q$ gives no stability certificate. |
| 2a | The certified terminal set is tiny ($\alpha \approx 10^{-4}$); the trajectory enters it only after 2.3 s, so the guarantee covers just the final approach. |
| 2b | Same nominal trajectory as Stage 1, now with the LQR cost-to-go as terminal cost; used as the terminal cost from Stage 3 on. |
| 3 | Under wind and a 30% mass error, standard NMPC settles 58 / −36 / −79 mm off target; offset-free NMPC removes the offset to below 0.05 mm. |
| 4a | With only $`[p;\, q;\, \omega]`$ measured, the EKF keeps tracking unbiased; velocity estimation error 53 mm/s, force-disturbance error 0.14 N, 0.7 ms per step. |
| 4b | The certified MHE ($T = 3$ s $> T_{\min} = 2$ s) estimates velocity $5\times$ and the disturbance $18\times$ more accurately than the EKF, at 14 ms per step (budget 50 ms). |
| 5 | A reference sequence with 6 m jumps, the last one outside the room ($x = 12$ m), is flown on the nominal plant. The artificial reference $p_a^\ast$ runs ahead of the quadrotor and stops at the boundary of the steady-state set; the position settles at the closest admissible point $y_r^\ast = (8, 6, 2)$ m. The flight is aggressive (tilt up to ≈ 65°, body rates up to ≈ 6.6 rad/s, motors saturated at the jumps). |

## File structure

```
quadrotor_3d_nmpc/
├── quadrotor_3d_model.py       # Shared: CasADi dynamics (nominal, disturbance-parameterized,
│                               #         augmented [x;d], MHE model), hover linearization,
│                               #         AcadosSimSolver plant, quaternion utilities,
│                               #         measurement model h(z), tracking model
│                               #         [x; p_a] and steady-state map g(p_a)
├── plot_utils.py               # Shared: all figures
│
├── ocp_config_basic.py         # Stage 1   — W_e = Q
├── simulate_basic.py           # Stage 1   — entry point
├── compute_qih_params.py       # Stage 2a  — offline CARE / Lyapunov P, α, Lipschitz check
├── ocp_config_qih.py           # Stage 2a  — P_lyap + soft terminal set
├── simulate_qih.py             # Stage 2a  — entry point (+ terminal-set diagnostic)
├── ocp_config_dare.py          # Stage 2b  — P_lqr terminal cost (DARE from ocp_config_offsetfree)
├── simulate_dare.py            # Stage 2b  — entry point
├── ekf_aug.py                  # Stage 3   — augmented EKF, full-state measurement
├── simulate_offsetfree.py      # Stage 3   — entry point (offset-free switched on at t = 4 s)
│
├── ss_target.py                # Stage 3–4 — analytical steady-state target calculator
├── ocp_config_offsetfree.py    # Stage 3–4 — offset-free NMPC (d̂ as runtime parameter,
│                               #             DARE terminal cost, soft |ω| box)
│
├── sensor_simulator.py         # Stage 4   — y = [p; q; ω] + Gaussian noise
├── observability_check.py      # Stage 4a  — observability of (F, H) at hover
├── state_est_ekf.py            # Stage 4a  — augmented EKF
├── detectability_check.py      # Stage 4b  — LMI detectability certificate → data/mhe_params.npz
├── ocp_config_mhe.py           # Stage 4b  — acados MHE OCP + certificate I/O
├── state_est_mhe.py            # Stage 4b  — MHE runtime wrapper (buffers, prior, warm start)
├── closed_loop_sim_config.py   # Stage 4   — the one closed loop both estimators run in
├── simulate_ekf.py             # Stage 4a  — entry point: closed loop with the EKF
├── simulate_mhe.py             # Stage 4b  — entry point: closed loop with the MHE
├── simulate_compare.py         # Stage 4   — EKF vs MHE from the saved runs
├── data/mhe_params.npz         # Stage 4b  — verified certificate (P, Q, R, λ, T_min, envelope)
│
├── ocp_config_tracking.py      # Stage 5   — tracking NMPC: artificial reference p_a as extra state,
│                               #             terminal equality x_N = x_a, offset cost V_o
├── simulate_tracking.py        # Stage 5   — entry point (reference jumps at t = 0, 5, 10 s)
│
├── run.sh                      # WSL bridge launcher (Windows Git Bash → WSL2)
├── CLAUDE.md                   # Project guide for Claude Code
├── docs/
│   ├── quadrotor_model.md
│   ├── stage1_basic_nmpc.md
│   ├── stage2_terminal_cost.md
│   ├── stage3_offset_free.md
│   ├── stage4a_ekf.md
│   ├── stage4b_mhe.md
│   └── stage5_tracking.md
├── results/                    # Figures per stage (tracked)
│   ├── stage1_basic/
│   ├── stage2a_qih/
│   ├── stage2b_dare/
│   ├── stage3_offsetfree/
│   ├── stage4a_ekf/             # figures + sim_data.npz
│   ├── stage4b_mhe/             # figures + sim_data.npz
│   ├── stage4_compare/
│   └── stage5_tracking/
└── c_generated_code/           # acados generated code (gitignored)
```

The `main` branch holds every stage, all running on the same shared files
(`quadrotor_3d_model.py`, `plot_utils.py`, and from Stage 2b on `ocp_config_offsetfree.py`).
Shared files only grow backward-compatibly: new stages add functions or optional keyword
arguments, so earlier stages keep running unchanged. The tags `stage1` … `stage5` are snapshots
of the code as it was when each stage was finished. OCP configuration is kept separate from the
simulation loop, so methods can be swapped and compared side by side.

## Toolchain

- **acados** (Python interface, `acados_template`) — OCP solver for the NMPC and the MHE
- **CasADi** 3.7.2 — symbolic dynamics and automatic differentiation
- **NumPy** 2.x, **SciPy** — offline computations (CARE, DARE, Lyapunov equations)
- **CVXPY** + **MOSEK** — SDP for the Stage 4b certificate (only `detectability_check.py`)
- **WSL2 / Ubuntu** — acados C libraries and Python venv (`~/acados_env`)
- **VS Code** on Windows for editing; execution routed through the WSL bridge

## Running

The project files live on the Windows drive; acados runs in WSL2. `run.sh` handles the hop:

```bash
# From Windows Git Bash, in the project directory (main branch)
./run.sh simulate_basic.py            # Stage 1               → results/stage1_basic/
./run.sh simulate_qih.py              # Stage 2a              → results/stage2a_qih/
./run.sh simulate_dare.py             # Stage 2b              → results/stage2b_dare/
./run.sh simulate_offsetfree.py       # Stage 3               → results/stage3_offsetfree/
./run.sh simulate_ekf.py              # Stage 4a closed loop  → results/stage4a_ekf/
./run.sh simulate_mhe.py              # Stage 4b closed loop  → results/stage4b_mhe/
./run.sh simulate_compare.py          # EKF vs MHE            → results/stage4_compare/
./run.sh simulate_tracking.py         # Stage 5               → results/stage5_tracking/

./run.sh observability_check.py       # observability at hover
./run.sh detectability_check.py       # re-derive the MHE certificate (needs cvxpy + MOSEK)

# The code as it was at the end of a stage: check out its tag, e.g.
git checkout stage3 && ./run.sh simulate_offsetfree.py   # back with: git checkout main

# Quick environment check
./run.sh -c "import acados_template; print('ok')"
```

Or directly inside WSL:

```bash
source ~/acados_env/bin/activate
cd /mnt/e/WorkSpace/GitHub/quadrotor_3d_nmpc
python simulate_ekf.py
```

The first solver build compiles C code (about 30 s); `c_generated_code/` is regenerated on later
runs and is gitignored.
