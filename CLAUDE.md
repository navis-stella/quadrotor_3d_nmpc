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
  ./run.sh simulation_main.py
  ./run.sh -c "import acados_template; print('ok')"
  ```
  `run.sh` (Windows) → `wsl.exe -e bash /home/celestialnavis/acados_run.sh` (WSL-side launcher).
- Editing files: do it directly on the E: path — same bytes WSL sees via `/mnt/e`.
- First solver build compiles C code (~30 s); `c_generated_code/` is regenerated and gitignored.

## Files

| File | Role |
|---|---|
| `quadrotor_3d_model.py` | CasADi symbolic model (`create_model`), hover linearization for DARE (`get_hover_linearization`), `AcadosSimSolver` plant (`create_plant_simulator`), numeric ODE + RK4 backup, `normalize_quaternion`. Physical params + `NX=13`, `NU=4`. |
| `nmpc_solver_creator.py` | Builds the acados OCP solver (`create_solver`): NONLINEAR_LS cost, box thrust constraints, SQP_RTI + ERK + PARTIAL_CONDENSING_HPIPM. |
| `simulation_main.py` | Closed-loop sim + matplotlib plots. Updated for the 13-state quaternion model. Calls `normalize_quaternion()` after each plant step. |

## Model conventions

- **State (13):** `[px py pz, vx vy vz, qw qx qy qz, p q r]`
  - position & linear velocity in **world** frame
  - quaternion **scalar-first**, **body→world**, unit norm; hover `q = [1,0,0,0]`
  - `p q r` = body-frame roll/pitch/yaw rates. Note: `q` (pitch rate) ≠ `qw/qx/qy/qz`.
- **Input (4):** `[f1 f2 f3 f4]` individual motor thrusts [N]. `+` layout: M1 front, M2 right, M3 back, M4 left. `f_hover = m g / 4 ≈ 2.45 N`.
- **Torques:** `tau_x = L(f4−f2)`, `tau_y = L(f3−f1)`, `tau_z = c_tau(−f1+f2−f3+f4)`.
- Quaternion cost weights ≈ 4× the equivalent Euler-angle weights (small-angle `phi ≈ 2 qx`); `qw` only lightly weighted.
- After every plant step call `normalize_quaternion(x)` to stop norm drift (also enforces `qw > 0`).

## Stage roadmap (from code comments)

1. **Stage 1 (current):** terminal cost `W_e = Q`, no terminal constraint.
2. **Stage 2:** `W_e = P` from DARE using `get_hover_linearization()`.
3. **Stage 3:** augmented model + offset-free NMPC; EKF in the loop.
4. **Stage 4:** obstacle-avoidance state constraints.
5. **Stage 5:** time-varying reference trajectory.

## Conventions

- Keep the existing house style: box-drawing section headers, aligned assignments, docstrings that explain the *why*.
- Don't commit `c_generated_code/`, `*.png`, `*.json` (already gitignored).
