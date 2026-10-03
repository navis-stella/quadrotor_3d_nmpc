# quadrotor_3d_nmpc — project guide

3D quadrotor nonlinear MPC practice using **acados** (Python interface) + CasADi.

## Environment (important)

- Project files live on the Windows E: drive; this Claude Code session runs on **Windows** (Git Bash).
- **acados** (C libs + `acados_template`) is installed in **WSL / Ubuntu**, not on Windows:
  - repo: `~/acados`, env vars `ACADOS_SOURCE_DIR=~/acados`, `LD_LIBRARY_PATH=~/acados/lib`
  - venv: `~/acados_env` (casadi 3.7.2, numpy 2.x, scipy, cvxpy + MOSEK) — auto-activated by `~/.bashrc`
  - project path inside WSL: `/mnt/e/WorkSpace/GitHub/quadrotor_3d_nmpc`
- **Run Python through the wrapper**, which hops into WSL and sets everything up:
  ```bash
  ./run.sh simulate_ekf.py
  ./run.sh -c "import acados_template; print('ok')"
  ```
  `run.sh` (Windows) → `wsl.exe -e bash /home/celestialnavis/acados_run.sh` (WSL-side launcher).
- Editing files: do it directly on the E: path — same bytes WSL sees via `/mnt/e`.
- First solver build compiles C code (~30 s); `c_generated_code/` is regenerated and gitignored.
- acados is built with OpenMP: set `OMP_NUM_THREADS=1` for these small problems (`closed_loop_sim_config.py` does) — otherwise thread overhead dominates the solve times.

## Files

The main branch holds **every stage**, all on the same shared files (`quadrotor_3d_model.py`, `plot_utils.py`, `ocp_config_offsetfree.py`, `ss_target.py`). Shared files must stay backward compatible: add functions or optional kwargs (default `None` = old behavior), never change an existing signature or result. After touching a shared file, re-run the earlier stages. Tags `stage1`…`stage4` are historical snapshots.

| File | Role |
|---|---|
| `quadrotor_3d_model.py` | Single source of the physics. Params, dims (`NX=13`, `NU=4`, `ND=6`, `NZ=19`, `NY=10`, `NW=NZ`), `IDX_Y`, `MIXER`, envelope `F_MAX`, `OMEGA_MAX`. Quaternion utils (`quat_to_euler`, `quat_normalize`, CasADi `quat_error_casadi`). Models: `create_disturbance_model` (MPC + plant, `p = d`), `get_augmented_dynamics` (EKF), `create_mhe_model` (MHE: same augmented ODE + process noise `w` as `model.u`, rotor thrusts as `model.p`). `create_nominal_model` (d = 0: Stages 1–2 OCPs, QIH check). Measurement: `measurement_expr`, `get_measurement_function`, `project_measurement` (common prior z̄₀). `hover_point`, `get_hover_linearization` (CasADi Jacobian), `create_disturbance_plant(Ts)`. |
| `ocp_config_basic.py` | **Stage 1** — NMPC on `create_nominal_model()`, `W_e = Q`, input box `F_MAX`. |
| `simulate_basic.py` | **Stage 1** — entry point; plant = `create_disturbance_plant(Ts)` with p = 0. Output `results/stage1_basic/`. |
| `compute_qih_params.py` | **Stage 2a** — offline CARE gain, modified-Lyapunov `P`, κ, α (input limit + seeded Monte Carlo Lipschitz check → reproducible α). |
| `ocp_config_qih.py` | **Stage 2a** — QIH-NMPC: `W_e = P_lyap` + soft terminal set `h_e ≤ α`. Module-level `Q`, `R`; `get_qih_params()` (cached, shared with the plot). |
| `simulate_qih.py` | **Stage 2a** — entry point + `plot_terminal_constraint`. Output `results/stage2a_qih/`. |
| `ocp_config_dare.py` | **Stage 2b** — `W_e = P_lqr`; imports `compute_dare_terminal_cost` from `ocp_config_offsetfree` (one implementation). |
| `simulate_dare.py` | **Stage 2b** — entry point. Output `results/stage2b_dare/`. |
| `ekf_aug.py` | **Stage 3** — `AugEKF` on z = [x; d] with full-state measurement y = x; `get_default_ekf_tuning`. Kept separate from `state_est_ekf` so Stage 3 results stay reproducible. |
| `simulate_offsetfree.py` | **Stage 3** — offset-free NMPC under full state feedback, activated at `T_ACTIVATE = 4 s`; figures via `plot_utils(..., X_s=, t_switch=)`. Output `results/stage3_offsetfree/`. |
| `sensor_simulator.py` | **Stage 4** — `SensorSimulator`: `y = [p; q; ω]` with per-channel Gaussian noise; `MOCAP_NOISE_STD` is the one noise definition (sensor and EKF R). Noise-free mode (`noise_std=None`) for open-loop checks. |
| `ocp_config_offsetfree.py` | **Stage 3, reused in 4** — Offset-free NMPC: disturbance model with `p = d̂`, DARE terminal cost (`compute_dare_terminal_cost`). Helpers `set_disturbance_param`, `set_reference`, `add_soft_omega_box` (`create_solver(..., omega_max=)`). |
| `ss_target.py` | **Stage 3, reused in 4** — Steady-state target (x_s, u_s) under d̂: force/torque balance, inverse `MIXER`. `compute_ss_target`, `print_ss_target`. |
| `state_est_ekf.py` | **Stage 4a** — `ExtendedKalmanFilter` on z = [x; d]: RK4 mean, Euler-linearized covariance, Joseph update, double-cover fix. Tuning: `default_ekf_Q(Ts)` (continuous-time density × Ts), `default_ekf_R(noise_std)`, `default_ekf_P0`. |
| `ocp_config_mhe.py` | **Stage 4b** — MHE OCP on z = [x; d]: discounted arrival + process-noise + measurement cost (quaternion double-cover trick), per-node flags for the growing horizon, soft ‖q‖=1, hard |ω| ≤ ω_max (8c). `export_drone_mhe_solver`, param layout `pack_mhe_param` / `N_PARAM` (36). Stage costs are written as the integrand — acados' default `cost_scaling` (= time steps) makes them the Riemann sum; the arrival (point) cost is divided by `Ts` to cancel that scaling. Certificate I/O `save_mhe_params` / `load_mhe_params(Ts=, horizon_factor=)`: N = ⌈c·T_min/Ts⌉ (`HORIZON_FACTOR` c = 1.5), N_min, ρ; path `MHE_PARAMS_PATH`. One decay symbol, λ per second: each node carries its age τ_i = t − t_i [s] (`t_age`) and is discounted by λ^{τ_i}. |
| `state_est_mhe.py` | **Stage 4b** — `MovingHorizonEstimator`: runtime wrapper (rolling ẑ/y/u buffers of length N, filtering prior ẑ_{k−N}, growing horizon, shifted warm start). Weights, λ and N from the certificate — no hand tuning. The MHE horizon is independent of the MPC horizon. |
| `detectability_check.py` | **Stage 4b** — δ-IOSS (i-iIOSS) certificate for the MHE: CT-LMI `[PA+AᵀP+κP−CᵀRC, PB; BᵀP, −Q] ⪯ 0` on the augmented model (B = I₁₉), vertex reduction over (ω box, T_total) + cutting-plane over attitudes + off-pool spot check. Design order (purely CT, no Ts): λ (decay per second, input) → κ = −ln λ → LMI → P, Q, R, T_min = ln4/κ; the horizon N is chosen afterwards in `load_mhe_params`. Writes `data/mhe_params.npz`. Needs `cvxpy` (MOSEK). |
| `observability_check.py` | **Stage 4** — local observability of the linearized augmented pair (F, H) at hover (SVD of the observability matrix). |
| `closed_loop_sim_config.py` | **Stage 4** — the ONE closed loop both estimators run in: sensor → `est.step(y_k, u_{k−1})` (EKF or MHE) → `ss_target` + offset-free NMPC → disturbance plant. Multi-rate capable (MPC every r samples, ZOH); currently Ts = Ts_mpc = 0.05 s. MPC soft \|ω\| ≤ 0.9·`OMEGA_MAX` keeps the plant in the certified envelope. Wall-clock timing of estimator and MPC. `X_REF`, `X0`, `get_disturbance`, `simulate`, `save_result`/`load_result` (`sim_data.npz`), `metrics`, `print_summary`, `run(kind)` (CLI shared by the two entry scripts). |
| `simulate_ekf.py` | **Stage 4a** — entry point: `closed_loop_sim_config.run('ekf')`. Output `results/stage4a_ekf/` (figures + `sim_data.npz`). |
| `simulate_mhe.py` | **Stage 4b** — entry point: `closed_loop_sim_config.run('mhe')`; needs `data/mhe_params.npz`. Output `results/stage4b_mhe/` (figures + `sim_data.npz`). |
| `simulate_compare.py` | **Stage 4** — loads both `sim_data.npz` (no simulation) → `results/stage4_compare/ekf_vs_mhe.png` + metrics table. |
| `plot_utils.py` | All figures: `plot_states`, `plot_inputs`, `plot_3d_trajectory`, `plot_terminal_constraint` (2a), `plot_disturbance`, `plot_velocity_estimation`, `plot_solve_times`, `plot_estimator_comparison`. Each ensures its save-path directory exists. |

## Model conventions

- **State (13):** `[px py pz, vx vy vz, qw qx qy qz, p q r]`
  - position & linear velocity in **world** frame
  - quaternion **scalar-first**, **body→world**, unit norm; hover `q = [1,0,0,0]`
  - `p q r` = body-frame roll/pitch/yaw rates. Note: `q` (pitch rate) ≠ `qw/qx/qy/qz`.
- **Augmented state (19):** `z = [x; d]`, `d = [d_fx d_fy d_fz` (world, N) `d_tx d_ty d_tz` (body, N·m)`]`, ḋ = 0.
- **Input (4):** `[f1 f2 f3 f4]` individual motor thrusts [N]. `+` layout: M1 front, M2 right, M3 back, M4 left. `f_hover = m g / 4 ≈ 2.45 N`, `0 ≤ fᵢ ≤ F_MAX = 3 f_hover`.
- **Torques:** `[T, tau_x, tau_y, tau_z] = MIXER @ u` — `tau_x = L(f4−f2)`, `tau_y = L(f3−f1)`, `tau_z = c_tau(−f1+f2−f3+f4)`.
- Quaternion cost weights ≈ 4× the equivalent Euler-angle weights (small-angle `phi ≈ 2 qx`); `qw` only lightly weighted.
- After every plant step call `quat_normalize(x)` to stop norm drift (also enforces `qw > 0`).

## Stage roadmap

1. **Stage 1:** terminal cost `W_e = Q`, no terminal constraint. (tag `stage1`)
2. **Stage 2a:** QIH-NMPC — `W_e = P_lyap` + terminal set constraint (soft). (tag `stage2`)
3. **Stage 2b:** DARE-based terminal cost `W_e = P_lqr`, no terminal constraint. (tag `stage2`)
4. **Stage 3:** augmented model + offset-free NMPC (full state feedback). (tag `stage3`)
5. **Stage 4:** partial, noisy measurement `y = [p; q; ω]` → estimate z = [x; d]; same offset-free NMPC.
   - **4a:** augmented EKF.
   - **4b:** augmented Lyapunov MHE with δ-IOSS certificate (λ → LMI → T_min → N).
6. **Stage 5:** obstacle-avoidance state constraints.
7. **Stage 6:** time-varying reference trajectory.
8. **Stage 7:** stochastic MPC.
9. **Stage 8:** robust MPC.

## Conventions

- Keep the existing house style: box-drawing section headers, aligned assignments, docstrings that explain the *why*.
- File headers: `name.py — <Stage> Title` with `<Stage>` = `All Stages`, `Stage 1`, `Stage 2a` (QIH), `Stage 2b` (DARE), `Stage 3`, `Stage 3–4`, `Stage 4`, `Stage 4a` (EKF) or `Stage 4b` (MHE), a `===` underline of the same length, then what/why.
- Don't commit `c_generated_code/`, `*.json` (gitignored). `results/<stage>/` figures are tracked on purpose (README).
