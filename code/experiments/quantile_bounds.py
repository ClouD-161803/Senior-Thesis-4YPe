"""
Conformal quantile bounds vs iteration experiment.

Produces per-iteration conformal upper quantile bound curves for image deblurring
metrics (NMSE, PSNR), comparable to the Stellato/Sambharya quantile bound plots.
"""

import sys
import json
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data import ImageConfig, BlurConfig, NoiseConfig, DataPipeline, MNISTSource, SyntheticSource
from utils.solver import ISTAConfig, SolverFactory
from utils.conformal import MetricFactory, MetricConfig, conformal_upper_quantile


def parse_arguments():
    """Parse CLI arguments, extending the base parser with quantile-specific flags."""
    parser = argparse.ArgumentParser(
        description="Conformal quantile bounds vs iteration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--n-samples', type=int, default=500, help='Number of images')
    parser.add_argument('--data-source', type=str, choices=['mnist', 'synthetic'], default='mnist', help='Data source')
    parser.add_argument('--k-iterations', type=int, default=100, help='Number of solver iterations')
    parser.add_argument('--rho-param', type=float, default=1e-4, help='Sparsity parameter')
    parser.add_argument('--kernel-size', type=int, default=8, help='Blur kernel size')
    parser.add_argument('--blur-std-dev', type=float, default=1.6, help='Blur kernel std dev')
    parser.add_argument('--noise-std-dev', type=float, default=1e-3, help='Noise std dev')
    parser.add_argument('--calibration-ratio', type=float, default=0.5, help='Proportion of data for calibration')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--solver', type=str, choices=['ista', 'fista'], default='ista', help='Solver algorithm')
    parser.add_argument('--metric', type=str, choices=['nmse', 'psnr'], default='nmse', help='Image quality metric')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.3, 0.8, 0.9], help='Quantile levels to plot')
    parser.add_argument('--no-plot', action='store_true', help='Skip plot generation')

    return parser.parse_args()


def run(args):
    """Run the quantile bounds experiment."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- data pipeline ---
    image_cfg = ImageConfig(shape=(28, 28), dtype=jnp.float32)
    blur_cfg = BlurConfig(kernel_size=args.kernel_size, std_dev=args.blur_std_dev)
    noise_cfg = NoiseConfig(std_dev=args.noise_std_dev, enabled=True)

    if args.data_source == 'mnist':
        source = MNISTSource(image_cfg, seed=args.seed)
    elif args.data_source == 'synthetic':
        source = SyntheticSource(image_cfg, seed=args.seed)
    else:
        raise ValueError(f"Unknown data source: {args.data_source}")

    pipeline = DataPipeline(image_cfg, blur_cfg, noise_cfg, source, seed=args.seed)

    # --- solver ---
    solver_cfg = ISTAConfig(
        max_iterations=args.k_iterations,
        sparsity_param=args.rho_param,
        step_size=None,
        step_size_factor=0.99
    )
    solver = SolverFactory.create(args.solver, solver_cfg)

    # --- metric ---
    metric = MetricFactory.create(args.metric, MetricConfig())

    # --- load and degrade ---
    print(f"----- Quantile Bounds Experiment -----")
    print(f"Solver: {args.solver}  |  Metric: {args.metric}")
    print(f"N = {args.n_samples}  |  K = {args.k_iterations}")
    print(f"Quantiles: {args.quantiles}")

    clean_images = pipeline.load_clean_images(args.n_samples)
    degraded_images, _ = pipeline.apply_degradation(clean_images, seed=args.seed)
    print("Data degraded.")

    # --- solve with history (vmap) ---
    blur_op = pipeline.get_blur_operator()
    lipschitz_constant = blur_op.get_lipschitz_constant()

    def solve_single(measurement):
        return solver.solve_with_history(
            forward_op=blur_op.apply,
            adjoint_op=blur_op.apply_adjoint,
            measurement=measurement,
            lipschitz_constant=lipschitz_constant
        )

    print("Running solver with history...")
    # iterates shape: (n_samples, K, H, W)
    all_iterates = jax.vmap(solve_single)(degraded_images)
    print("Solver complete.")

    # prepend z0 to get (n_samples, K+1, H, W)
    z0 = jnp.zeros_like(clean_images)  # (n_samples, H, W)
    all_iterates = jnp.concatenate(
        [z0[:, None, :, :], all_iterates], axis=1
    )
    K_plus_1 = all_iterates.shape[1]

    # --- compute per-iteration metric ---
    def metric_single(iterate, ground_truth):
        return metric.compute(iterate, ground_truth)

    # vmap over iterations for a single image
    metric_over_iters = jax.vmap(metric_single, in_axes=(0, None))
    # vmap over images
    metric_over_images_iters = jax.vmap(metric_over_iters, in_axes=(0, 0))

    print("Computing per-iteration metrics...")
    # metrics shape: (n_samples, K+1)
    metrics = np.array(metric_over_images_iters(all_iterates, clean_images))
    print(f"Metrics shape: {metrics.shape}")

    # --- cal/test split ---
    np.random.seed(args.seed)
    indices = np.random.permutation(args.n_samples)
    n_cal = int(np.ceil(args.n_samples * args.calibration_ratio))
    cal_indices = indices[:n_cal]
    test_indices = indices[n_cal:]

    cal_metrics = metrics[cal_indices]   # (n_cal, K+1)
    test_metrics = metrics[test_indices]  # (n_test, K+1)

    print(f"Calibration: {len(cal_indices)} | Test: {len(test_indices)}")

    # --- compute quantile curves ---
    results_per_q = {}
    for q in args.quantiles:
        empirical_q = np.zeros(K_plus_1)
        conformal_q = np.zeros(K_plus_1)
        coverage_q = np.zeros(K_plus_1)

        for k in range(K_plus_1):
            empirical_q[k] = float(np.quantile(test_metrics[:, k], q))
            conformal_q[k] = conformal_upper_quantile(cal_metrics[:, k], q)
            coverage_q[k] = float(np.mean(test_metrics[:, k] <= conformal_q[k]))

        results_per_q[str(q)] = {
            'empirical_q': empirical_q.tolist(),
            'conformal_q': conformal_q.tolist(),
            'coverage_q': coverage_q.tolist(),
        }

        print(f"  q={q}: mean conformal bound = {np.mean(conformal_q):.4f}, "
              f"mean coverage = {np.mean(coverage_q):.4f}")

    # --- save JSON ---
    output_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        'config': {
            'solver': args.solver,
            'metric': args.metric,
            'n_samples': args.n_samples,
            'k_iterations': args.k_iterations,
            'quantiles': args.quantiles,
            'calibration_ratio': args.calibration_ratio,
            'seed': args.seed,
        },
        'quantile_results': results_per_q,
    }

    json_path = output_dir / f'quantile_bounds_{args.solver}_{args.metric}_N{args.n_samples}_K{args.k_iterations}.json'
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"JSON saved: {json_path}")

    # --- plot ---
    if not args.no_plot:
        n_q = len(args.quantiles)
        fig, axes = plt.subplots(1, n_q, figsize=(6 * n_q, 5), sharex=True, sharey=True, squeeze=False)
        axes = axes[0]
        iters = np.arange(K_plus_1)

        for i, q in enumerate(args.quantiles):
            ax = axes[i]
            rq = results_per_q[str(q)]
            ax.plot(iters, rq['empirical_q'], label='Empirical (test)', linewidth=1.5)
            ax.plot(iters, rq['conformal_q'], label='Conformal bound (cal)', linewidth=1.5, linestyle='--')
            ax.set_title(f'{int(100 * q)}th quantile bound', fontsize=14)
            ax.set_xlabel('Iteration $k$', fontsize=12)
            if i == 0:
                ax.set_ylabel(f'{args.metric.upper()} (dB)', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f'Quantile bounds vs iteration ({args.solver.upper()}, {args.metric.upper()})',
            fontsize=16, fontweight='bold'
        )
        plt.tight_layout(rect=(0, 0, 1, 0.93))

        png_path = output_dir / f'quantile_bounds_{args.solver}_{args.metric}_N{args.n_samples}_K{args.k_iterations}.png'
        plt.savefig(str(png_path), dpi=150)
        plt.close(fig)
        print(f"Plot saved: {png_path}")


if __name__ == '__main__':
    args = parse_arguments()
    run(args)
