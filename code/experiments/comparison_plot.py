"""
Generate a Stellato/Sambharya-style conformal quantile bounds comparison plot.

Runs FISTA with inline metric computation (memory-efficient) for K=10000
iterations on N=500 MNIST images, then plots quantile bounds vs iteration
on a log-scale x-axis.

Reuses the K=1000 FISTA JSON from a previous run for validation of the
overlapping range (iterations 0..1000).

Usage:
    cd code
    conda run -n conformal python experiments/comparison_plot.py
"""

import sys
import json
import time
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
K1000_JSON = OUTPUT_DIR / 'quantile_bounds_fista_nmse_N500_K1000.json'


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


def run_fista_metrics_only(degraded, clean, blur_op):
    """
    Run FISTA for K_MAX iterations, computing NMSE(dB) at each step.

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

    def fista_single(measurement, ground_truth):
        z0 = jnp.zeros_like(measurement)
        score_0 = nmse_db(z0, ground_truth)

        def fista_step(carry, _):
            z, z_prev, t = carry
            grad = adjoint_op(forward_op(z) - measurement)
            # soft-thresholding
            u = z - step_size * grad
            threshold = RHO * step_size
            z_thresh = jnp.sign(u) * jnp.maximum(jnp.abs(u) - threshold, 0)
            # box projection
            z_new = jnp.clip(z_thresh, 0, 1)
            # momentum
            t_new = (1 + jnp.sqrt(1 + 4 * t ** 2)) / 2
            z_accel = z_new + ((t - 1) / t_new) * (z_new - z_prev)
            score = nmse_db(z_new, ground_truth)
            return (z_accel, z_new, t_new), score

        _, scores = jax.lax.scan(fista_step, (z0, z0, 1.0), None, length=K_MAX)
        return jnp.concatenate([score_0[None], scores])

    print(f"Compiling & running FISTA (metrics-only) for K={K_MAX}, N={N_SAMPLES}...")
    t0 = time.time()
    batch_fn = jax.vmap(fista_single)
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


def validate_against_k1000(results):
    """Load K=1000 FISTA JSON and check the overlapping range matches."""
    if not K1000_JSON.exists():
        print(f"K=1000 JSON not found at {K1000_JSON}, skipping validation.")
        return

    with open(K1000_JSON) as f:
        k1000 = json.load(f)

    print("\nValidating against K=1000 FISTA run:")
    for q in QUANTILES:
        key = str(q)
        if key not in k1000['quantile_results']:
            continue
        ref_conf = np.array(k1000['quantile_results'][key]['conformal_q'])
        our_conf = results[q]['conformal'][:len(ref_conf)]
        max_diff = np.max(np.abs(ref_conf - our_conf))
        print(f"  q={q}: max |diff| in conformal bound (iters 0..1000) = {max_diff:.6f}")


def make_plot(results):
    """Generate Stellato-style comparison plot with log-scale x-axis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    iters = np.arange(1, K_MAX + 1)  # skip k=0 for log scale

    n_q = len(QUANTILES)
    fig, axes = plt.subplots(1, n_q, figsize=(6 * n_q, 5), sharey=True, squeeze=False)
    axes = axes[0]

    for i, q in enumerate(QUANTILES):
        ax = axes[i]
        r = results[q]
        ax.plot(iters, r['empirical'][1:],
                label='Empirical (test)', linewidth=1.5)
        ax.plot(iters, r['conformal'][1:],
                label='Conformal bound (cal)', linewidth=1.5, linestyle='--')
        ax.set_xscale('log')
        ax.set_title(f'{int(100 * q)}th quantile bound', fontsize=14)
        ax.set_xlabel('Iteration $k$', fontsize=12)
        if i == 0:
            ax.set_ylabel('NMSE (dB)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')

    fig.suptitle(
        f'Conformal quantile bounds vs iteration (FISTA, NMSE)\n'
        f'$N = {N_SAMPLES}$,  $K = {K_MAX}$',
        fontsize=16, fontweight='bold'
    )
    plt.tight_layout(rect=(0, 0, 1, 0.90))

    png_path = OUTPUT_DIR / f'comparison_fista_nmse_N{N_SAMPLES}_K{K_MAX}.png'
    plt.savefig(str(png_path), dpi=150)
    plt.close(fig)
    print(f"Plot saved: {png_path}")


def save_json(results):
    """Save full results to JSON."""
    output = {
        'config': {
            'solver': 'fista',
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

    json_path = OUTPUT_DIR / f'comparison_fista_nmse_N{N_SAMPLES}_K{K_MAX}.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"JSON saved: {json_path}")


if __name__ == '__main__':
    clean, degraded, blur_op = setup_data()
    metrics = run_fista_metrics_only(degraded, clean, blur_op)
    results = compute_bounds(metrics)
    validate_against_k1000(results)
    save_json(results)
    make_plot(results)
