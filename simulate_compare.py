"""
simulate_compare.py — Stage 4 EKF vs MHE Comparison
===================================================
Loads the saved runs of simulate_ekf.py and simulate_mhe.py — the same
closed loop (closed_loop_sim_config.py) with only the estimator swapped —
and puts them side by side. No simulation happens here, so the comparison can be
redrawn without re-running the solvers.

    tracking error ‖p − p_ref‖, velocity estimation error ‖v̂ − v‖,
    force-disturbance estimation error ‖d̂_f − d_f‖, estimator computation time
    + a table of the steady-state metrics and timing statistics

Usage:
    ./run.sh simulate_ekf.py && ./run.sh simulate_mhe.py
    ./run.sh simulate_compare.py
Output: results/stage4_compare/ekf_vs_mhe.png
"""

import numpy as np

from closed_loop_sim_config import ESTIMATORS, results_dir, load_result, metrics
from plot_utils  import plot_estimator_comparison


def print_comparison(results: dict):
    """Side-by-side table of the metrics of closed_loop_sim_config.print_summary."""
    ms = {name: metrics(r) for name, r in results.items()}
    rows = [('‖Δp‖ steady state [mm]',   lambda m: 1e3 * np.linalg.norm(m['pos_err'])),
            ('‖Δv̂‖ steady state [mm/s]', lambda m: 1e3 * np.linalg.norm(m['v_err'])),
            ('‖Δd̂_f‖ steady state [N]',  lambda m: np.linalg.norm(m['d_err'][:3])),
            ('max |ω| true [rad/s]',      lambda m: m['omega_max']),
            ('estimator mean [ms]',       lambda m: m['T_est'].mean()),
            ('estimator p99 [ms]',        lambda m: np.percentile(m['T_est'], 99)),
            ('MPC mean [ms]',             lambda m: m['T_mpc'].mean())]

    print('\n' + '=' * 72)
    print('STAGE 4 — ESTIMATOR COMPARISON (identical closed loop)')
    print('=' * 72)
    print(f'  {"":30s}' + ''.join(f'{n:>12s}' for n in ms))
    for label, f in rows:
        print(f'  {label:30s}' + ''.join(f'{f(m):12.4f}' for m in ms.values()))
    for r in results.values():
        print(f'  {r["info"]}')
    print('=' * 72)


if __name__ == '__main__':
    results = {}
    for kind in ESTIMATORS:
        r = load_result(results_dir(kind))
        results[r['name']] = r

    x_refs = [r['x_ref'] for r in results.values()]
    assert all(np.allclose(x_refs[0], x) for x in x_refs), 'runs use different references'

    plot_estimator_comparison(results, x_refs[0],
                              title='Stage 4 — EKF vs MHE on the identical closed loop',
                              save_path='results/stage4_compare/ekf_vs_mhe.png')
    print_comparison(results)
