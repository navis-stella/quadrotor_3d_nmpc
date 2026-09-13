# quadrotor_3d_nmpc — project guide

3D quadrotor nonlinear MPC practice using **acados** (Python interface) + CasADi.

## Environment (important)

- Project files live on the Windows E: drive; this Claude Code session runs on **Windows** (Git Bash).
- **acados** (C libs + `acados_template`) is installed in **WSL / Ubuntu**, not on Windows:
  - repo: `~/acados`, env vars `ACADOS_SOURCE_DIR=~/acados`, `LD_LIBRARY_PATH=~/acados/lib`
  - venv: `~/acados_env` (casadi 3.7.2, numpy 2.x, scipy) — auto-activated by `~/.bashrc`
  - project path inside WSL: `/mnt/e/WorkSpace/GitHub/quadrotor_3d_nmpc`
- **Run Python through the wrapper**, which hops into WSL and sets everything up:
  ```bash
  ./run.sh simulate_dare.py
  ./run.sh -c "import acados_template; print('ok')"
  ```
  `run.sh` (Windows) → `wsl.exe -e bash /home/celestialnavis/acados_run.sh` (WSL-side launcher).
- Editing files: do it directly on the E: path — same bytes WSL sees via `/mnt/e`.
- First solver build compiles C code (~30 s); `c_generated_code/` is regenerated and gitignored.

## Files

| File | Role |
|---|---|
| `quadrotor_3d_model.py` | CasADi symbolic model (`create_model`), hover linearization (`get_hover_linearization`), disturbance model (`create_disturbance_model`), augmented dynamics (`get_augmented_dynamics_casadi`), `AcadosSimSolver` plant (`create_plant_simulator`, `create_disturbance_plant`), numeric ODE + RK4 backup, quaternion utilities (`normalize_quaternion`, `quat_to_euler`). Physical params + `NX=13`, `NU=4`, `ND=6`, `NZ=19`. |
| `plot_utils.py` | Shared plotting functions used by all simulation scripts: `plot_states`, `plot_inputs`, `plot_3d_trajectory`, `plot_terminal_constraint`, `plot_disturbance`. |
| `ss_target.py` | Steady-state target calculator for offset-free NMPC: analytical equilibrium (x_s, u_s) from force/torque balance under d̂. Motor mixer inversion. `compute_ss_target()`, `print_ss_target()`. |
| `ekf_aug.py` | **Stage 3** — Augmented EKF (`AugEKF`): estimates `z = [x(13); d(6)]`, disturbance adaptation via cross-covariance. Default tuning in `get_default_ekf_tuning()`. |
| `ocp_config_offsetfree.py` | **Stage 3** — Offset-free NMPC solver: disturbance model with `p = d̂`, DARE terminal cost. Helpers: `set_disturbance_param()`, `set_reference()`. |
| `simulate_offsetfree.py` | **Stage 3** — Closed-loop offset-free sim with EKF + steady-state target + disturbance scenario. Results saved to `results/stage3_offsetfree/`. |

## Model conventions

- **State (13):** `[px py pz, vx vy vz, qw qx qy qz, p q r]`
  - position & linear velocity in **world** frame
  - quaternion **scalar-first**, **body→world**, unit norm; hover `q = [1,0,0,0]`
  - `p q r` = body-frame roll/pitch/yaw rates. Note: `q` (pitch rate) ≠ `qw/qx/qy/qz`.
- **Input (4):** `[f1 f2 f3 f4]` individual motor thrusts [N]. `+` layout: M1 front, M2 right, M3 back, M4 left. `f_hover = m g / 4 ≈ 2.45 N`.
- **Torques:** `tau_x = L(f4−f2)`, `tau_y = L(f3−f1)`, `tau_z = c_tau(−f1+f2−f3+f4)`.
- Quaternion cost weights ≈ 4× the equivalent Euler-angle weights (small-angle `phi ≈ 2 qx`); `qw` only lightly weighted.
- After every plant step call `normalize_quaternion(x)` to stop norm drift (also enforces `qw > 0`).

## Stage roadmap

1. **Stage 1:** terminal cost `W_e = Q`, no terminal constraint.
2. **Stage 2.1:** QIH-NMPC — `W_e = P_lyap` + terminal set constraint (soft).
3. **Stage 2.2:** DARE-based terminal cost `W_e = P_lqr`, no terminal constraint.
4. **Stage 3:** augmented model + offset-free NMPC.
5. **Stage 4:** obstacle-avoidance state constraints.
6. **Stage 5:** time-varying reference trajectory.
7. **Stage 6:** noisy measurements — EKF/MHE + nominal MPC.
8. **Stage 7:** stochastic MPC.
9. **Stage 8:** robust MPC.

## Conventions

- Keep the existing house style: box-drawing section headers, aligned assignments, docstrings that explain the *why*.
- Don't commit `c_generated_code/`, `results/`, `*.json` (already gitignored).
