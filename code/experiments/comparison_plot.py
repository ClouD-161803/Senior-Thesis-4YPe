"""
Generate a Stellato/Sambharya-style conformal quantile bounds comparison plot.

Runs ISTA or FISTA with inline metric computation (memory-efficient) for
K=10000 iterations on N=500 MNIST images, then plots quantile bounds vs
iteration on a log-scale x-axis.

Usage:
    cd code
    conda run -n conformal python experiments/comparison_plot.py --solver ista
    conda run -n conformal python experiments/comparison_plot.py --solver fista
"""

import sys
import json
import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data import ImageConfig, BlurConfig, NoiseConfig, DataPipeline, MNISTSource
from utils.conformal import conformal_upper_quantile

# ---- Configuration ----
N_SAMPLES = 500
K_MAX = 10000
QUANTILES = [0.3, 0.8, 0.9]
SEED = 42
RHO = 1e-4
STEP_SIZE_FACTOR = 0.99
CALIBRATION_RATIO = 0.5
OUTPUT_DIR = Path('./results')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Conformal quantile bounds comparison plot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--solver', type=str, choices=['ista', 'fista'],
                        default='ista', help='Solver algorithm')
    return parser.parse_args()


def setup_data():
    """Setup data pipeline, load and degrade images."""
    image_cfg = ImageConfig(shape=(28, 28), dtype=jnp.float32)
    blur_cfg = BlurConfig(kernel_size=8, std_dev=1.6)
    noise_cfg = NoiseConfig(std_dev=1e-3, enabled=True)
    source = MNISTSource(image_cfg, seed=SEED)
    pipeline = DataPipeline(image_cfg, blur_cfg, noise_cfg, source, seed=SEED)

    clean = pipeline.load_clean_images(N_SAMPLES)
    degraded, _ = pipeline.apply_degradation(clean, seed=SEED)
    blur_op = pipeline.get_blur_operator()

    return clean, degraded, blur_op


def run_solver_metrics_only(degraded, clean, blur_op, solver_name):
    """
    Run solver for K_MAX iterations, computing NMSE(dB) at each step.

    Memory-efficient: the lax.scan body outputs a scalar metric per step
    instead of the full (H, W) iterate, so total storage is (N, K+1)
    floats (~20 MB) instead of (N, K, H, W) (~16 GB).

    Returns:
        metrics: array of shape (N_SAMPLES, K_MAX+1).
    """
    L = blur_op.get_lipschitz_constant()
    step_size = STEP_SIZE_FACTOR / L
    forward_op = blur_op.apply
    adjoint_op = blur_op.apply_adjoint

    def nmse_db(z, gt):
        mse = jnp.mean((z - gt) ** 2)
        norm = jnp.mean(gt ** 2)
        return 10.0 * jnp.log10(mse / norm)

    def proximal_step(z, grad):
        u = z - step_size * grad
        threshold = RHO * step_size
        z_thresh = jnp.sign(u) * jnp.maximum(jnp.abs(u) - threshold, 0)
        return jnp.clip(z_thresh, 0, 1)

    def ista_single(measurement, ground_truth):
        z0 = jnp.zeros_like(measurement)
        score_0 = nmse_db(z0, ground_truth)

        def ista_step(z, _):
            grad = adjoint_op(forward_op(z) - measurement)
            z_new = proximal_step(z, grad)
            score = nmse_db(z_new, ground_truth)
            return z_new, score

        _, scores = jax.lax.scan(ista_step, z0, None, length=K_MAX)
        return jnp.concatenate([score_0[None], scores])

    def fista_single(measurement, ground_truth):
        z0 = jnp.zeros_like(measurement)
        score_0 = nmse_db(z0, ground_truth)

        def fista_step(carry, _):
            z, z_prev, t = carry
            grad = adjoint_op(forward_op(z) - measurement)
            z_new = proximal_step(z, grad)
            t_new = (1 + jnp.sqrt(1 + 4 * t ** 2)) / 2
            z_accel = z_new + ((t - 1) / t_new) * (z_new - z_prev)
            score = nmse_db(z_new, ground_truth)
            return (z_accel, z_new, t_new), score

        _, scores = jax.lax.scan(fista_step, (z0, z0, 1.0), None, length=K_MAX)
        return jnp.concatenate([score_0[None], scores])

    solve_single = ista_single if solver_name == 'ista' else fista_single

    print(f"Compiling & running {solver_name.upper()} (metrics-only) "
          f"for K={K_MAX}, N={N_SAMPLES}...")
    t0 = time.time()
    batch_fn = jax.vmap(solve_single)
    metrics = np.array(batch_fn(degraded, clean))
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s. Metrics shape: {metrics.shape}")
    return metrics


def compute_bounds(metrics):
    """Compute empirical and conformal quantile bounds at each iteration."""
    np.random.seed(SEED)
    indices = np.random.permutation(N_SAMPLES)
    n_cal = int(np.ceil(N_SAMPLES * CALIBRATION_RATIO))
    cal_idx, test_idx = indices[:n_cal], indices[n_cal:]

    cal_metrics = metrics[cal_idx]
    test_metrics = metrics[test_idx]
    K_plus_1 = metrics.shape[1]

    print(f"Cal: {len(cal_idx)} | Test: {len(test_idx)}")

    results = {}
    for q in QUANTILES:
        emp = np.zeros(K_plus_1)
        conf = np.zeros(K_plus_1)
        cov = np.zeros(K_plus_1)
        for k in range(K_plus_1):
            emp[k] = float(np.quantile(test_metrics[:, k], q))
            conf[k] = conformal_upper_quantile(cal_metrics[:, k], q)
            cov[k] = float(np.mean(test_metrics[:, k] <= conf[k]))
        results[q] = {'empirical': emp, 'conformal': conf, 'coverage': cov}
        print(f"  q={q}: final conformal = {conf[-1]:.2f} dB, "
              f"final coverage = {cov[-1]:.3f}")

    return results


def validate_against_k1000(results, solver_name):
    """Load K=1000 JSON and check the overlapping range matches."""
    k1000_json = OUTPUT_DIR / f'quantile_bounds_{solver_name}_nmse_N500_K1000.json'
    if not k1000_json.exists():
        print(f"K=1000 JSON not found at {k1000_json}, skipping validation.")
        return

    with open(k1000_json) as f:
        k1000 = json.load(f)

    print(f"\nValidating against K=1000 {solver_name.upper()} run:")
    for q in QUANTILES:
        key = str(q)
        if key not in k1000['quantile_results']:
            continue
        ref_conf = np.array(k1000['quantile_results'][key]['conformal_q'])
        our_conf = results[q]['conformal'][:len(ref_conf)]
        max_diff = np.max(np.abs(ref_conf - our_conf))
        print(f"  q={q}: max |diff| in conformal bound (iters 0..1000) = {max_diff:.6f}")


def make_plot(results, solver_name):
    """Generate Stellato-style comparison plot with log-scale x-axis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.titlesize': 24,
    })

    iters = np.arange(1, K_MAX + 1)  # skip k=0 for log scale

    n_q = len(QUANTILES)
    fig, axes = plt.subplots(1, n_q, figsize=(7 * n_q, 6), sharey=True, squeeze=False)
    axes = axes[0]

    for i, q in enumerate(QUANTILES):
        ax = axes[i]
        r = results[q]
        ax.plot(iters, r['empirical'][1:],
                label='Empirical (test)', linewidth=2)
        ax.plot(iters, r['conformal'][1:],
                label='Conformal bound (cal)', linewidth=2, linestyle='--')
        ax.set_xscale('log')
        ax.set_title(f'{int(100 * q)}th quantile bound')
        ax.set_xlabel('Iteration $k$')
        if i == 0:
            ax.set_ylabel('NMSE (dB)')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        ax.tick_params(axis='both', which='major', labelsize=16)

    fig.suptitle(
        f'Conformal quantile bounds vs iteration ({solver_name.upper()}, NMSE)\n'
        f'$N = {N_SAMPLES}$,  $K = {K_MAX}$',
        fontweight='bold'
    )
    plt.tight_layout(rect=(0, 0, 1, 0.90))

    png_path = OUTPUT_DIR / f'comparison_{solver_name}_nmse_N{N_SAMPLES}_K{K_MAX}.png'
    plt.savefig(str(png_path), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {png_path}")


def save_json(results, solver_name):
    """Save full results to JSON."""
    output = {
        'config': {
            'solver': solver_name,
            'metric': 'nmse',
            'n_samples': N_SAMPLES,
            'k_max': K_MAX,
            'quantiles': QUANTILES,
            'seed': SEED,
        },
        'quantile_results': {},
    }
    for q in QUANTILES:
        output['quantile_results'][str(q)] = {
            'empirical_q': results[q]['empirical'].tolist(),
            'conformal_q': results[q]['conformal'].tolist(),
            'coverage_q': results[q]['coverage'].tolist(),
        }

    json_path = OUTPUT_DIR / f'comparison_{solver_name}_nmse_N{N_SAMPLES}_K{K_MAX}.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"JSON saved: {json_path}")


if __name__ == '__main__':
    args = parse_args()
    solver_name = args.solver

    clean, degraded, blur_op = setup_data()
    metrics = run_solver_metrics_only(degraded, clean, blur_op, solver_name)
    results = compute_bounds(metrics)
    validate_against_k1000(results, solver_name)
    save_json(results, solver_name)
    make_plot(results, solver_name)
