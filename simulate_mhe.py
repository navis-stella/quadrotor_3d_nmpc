"""
simulate_mhe.py — Stage 4b Closed-Loop Offset-Free NMPC with Augmented Lyapunov MHE
===================================================================================
Runs the shared Stage 4 closed loop (closed_loop_sim_config.py) with
the augmented Lyapunov MHE (state_est_mhe.py) as the estimator:

    sensor y = [p; q; ω] → MHE ẑ = [x̂; d̂] → ss_target + offset-free NMPC → plant

The MHE weights, decay and horizon come from data/mhe_params.npz — run
detectability_check.py first. Everything except the estimator is identical
to simulate_ekf.py — see the closed_loop_sim_config.py header for the
scenario, rates and timing.

Usage:
    ./run.sh simulate_mhe.py                  # Stage 3 disturbance scenario
Output: results/stage4b_mhe/  figures + sim_data.npz (→ simulate_compare.py)
"""

from closed_loop_sim_config import run

if __name__ == '__main__':
    run('mhe')
