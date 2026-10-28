"""
simulate_obst_dist.py — Stage 7a Obstacle Avoidance with a Distance Constraint
==============================================================================
Entry point: the Stage 6 helix through three pillars (obstacle_sim_config),
each pillar a soft squared-distance constraint on the predicted positions:

    hⱼ(p_i) = ‖p_xy,i − cⱼ‖² − r_safe² ≥ 0        i = 1..N

    exact penalty (Z_L1), warm start projected onto the free space
    (ocp_config_obstacle, gamma=None).

The constraint only says WHERE the quadrotor may be: it acts once a
predicted node reaches the circle, i.e. about one horizon before a pillar,
and the closed loop then rides the circle at the margin δ. Stage 7b
(simulate_obst_dcbf.py) replaces it by a DCBF that limits how fast the
quadrotor may approach.

Run:  python simulate_obst_dist.py   → results/stage7a_obst_dist/
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')    # before the solver libraries load

import obstacle_sim_config as osc

RESULTS_DIR = os.path.join('results', 'stage7a_obst_dist')


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    res = osc.simulate()
    osc.summarize(res, 'Stage 7a — Obstacle avoidance, distance constraint (nominal, full state)')
    osc.make_figures(res, RESULTS_DIR, 'Stage 7a — Obstacle avoidance')
