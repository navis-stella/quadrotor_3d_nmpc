"""
simulate_ekf.py — Stage 4a Closed-Loop Offset-Free NMPC with Augmented EKF
==========================================================================
Runs the shared Stage 4 closed loop (closed_loop_sim_config.py) with
the augmented EKF (state_est_ekf.py) as the estimator:

    sensor y = [p; q; ω] → EKF ẑ = [x̂; d̂] → ss_target + offset-free NMPC → plant

Everything except the estimator is identical to simulate_mhe.py — see the
closed_loop_sim_config.py header for the scenario, rates and timing.

Usage:
    ./run.sh simulate_ekf.py                  # Stage 3 disturbance scenario
Output: results/stage4a_ekf/  figures + sim_data.npz (→ simulate_compare.py)
"""

from closed_loop_sim_config import run

if __name__ == '__main__':
    run('ekf')
