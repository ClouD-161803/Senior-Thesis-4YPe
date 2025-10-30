import argparse
import numpy as np
from pathlib import Path

import jax
import jax.numpy as jnp

from .data import ImageConfig, BlurConfig, NoiseConfig, DataPipeline, MNISTSource, SyntheticSource
from .solver import ISTAConfig, SolverFactory
from .conformal import ConformalConfig, ConformalFactory, ConformalAnalyser, MetricFactory
from .plotter import PlotConfig, PlotterFactory


def setup_data_pipeline(args) -> DataPipeline:
    """Initialise data pipeline from arguments."""
    image_cfg = ImageConfig(
        shape=(28, 28),
        dtype=jnp.float32
    )
    
    blur_cfg = BlurConfig(
        kernel_size=args.kernel_size,
        std_dev=args.blur_std_dev
    )
    
    noise_cfg = NoiseConfig(
        std_dev=args.noise_std_dev,
        enabled=args.add_noise
    )
    
    if args.data_source == 'mnist':
        source = MNISTSource(image_cfg)
    elif args.data_source == 'synthetic':
        source = SyntheticSource(image_cfg)
    else:
        raise ValueError(f"Unknown data source: {args.data_source}")
    
    return DataPipeline(image_cfg, blur_cfg, noise_cfg, source)


def setup_solver(args):
    """Initialise solver from arguments."""
    solver_cfg = ISTAConfig(
        max_iterations=args.k_iterations,
        sparsity_param=args.rho_param,
        step_size=None,
        step_size_factor=0.99
    )
    
    return SolverFactory.create(args.solver, solver_cfg)


def run_experiment(
    data_pipeline: DataPipeline,
    solver_instance,
    n_samples: int,
    seed: int = 42
):
    """
    Execute full experiment pipeline.
    
    Returns:
        scores: Array of NMSE scores.
        clean_images: Array of clean images.
        degraded_images: Array of degraded images.
        reconstructed_images: Array of reconstructed images.
    """
    print(f"\nLoading {n_samples} images...")
    
    clean_images = data_pipeline.load_clean_images(n_samples)
    degraded_images, _ = data_pipeline.apply_degradation(clean_images, seed=seed)
    print(f"Degradation applied.")
    
    blur_op = data_pipeline.get_blur_operator()
    lipschitz_constant = blur_op.get_lipschitz_constant()
    
    metric = MetricFactory.create('nmse')
    
    def reconstruct_single(degraded, clean):
        reconstructed = solver_instance.solve(
            forward_op=blur_op.apply,
            adjoint_op=blur_op.apply_adjoint,
            measurement=degraded,
            lipschitz_constant=lipschitz_constant
        )
        score = metric.compute(reconstructed, clean)
        return score, reconstructed
    
    reconstruct_vmapped = jax.vmap(reconstruct_single)
    scores, reconstructed_images = reconstruct_vmapped(degraded_images, clean_images)
    
    print(f"Reconstruction complete.")
    print(f"  Mean NMSE: {float(jnp.mean(scores)):.4f} dB")
    print(f"  Min NMSE: {float(jnp.min(scores)):.4f} dB")
    print(f"  Max NMSE: {float(jnp.max(scores)):.4f} dB")
    
    return np.array(scores), clean_images, degraded_images, reconstructed_images


def run_conformal_prediction(
    scores: np.ndarray,
    config: ConformalConfig
):
    """
    Execute conformal prediction pipeline.
    
    Returns:
        predictor: Trained conformal predictor.
        calibration_scores: Calibration set scores.
        test_scores: Test set scores.
    """
    
    calibration_scores, test_scores = ConformalAnalyser.split_data(
        scores,
        calibration_ratio=config.calibration_ratio,
        seed=42
    )
    
    print(f"\nCalibration set size: {len(calibration_scores)}")
    print(f"Test set size: {len(test_scores)}")
    
    predictor = ConformalFactory.create('standard', config)
    q_hat = predictor.calibrate(calibration_scores)
    
    print(f"Quantile q̂: {q_hat:.4f} dB")
    
    coverage = predictor.evaluate_coverage(test_scores)
    
    is_valid, status = ConformalAnalyser.validate(
        coverage,
        1 - config.delta
    )
    
    print(f"Empirical coverage: {coverage:.4f}")
    print(f"Target coverage: {1-config.delta:.4f}")
    print(f"Result: {status}")
    
    return predictor, calibration_scores, test_scores, coverage, q_hat


def generate_visualisations(
    clean_images: jnp.ndarray,
    degraded_images: jnp.ndarray,
    reconstructed_images: jnp.ndarray,
    scores: np.ndarray,
    output_dir: Path,
    config: PlotConfig
):
    """Generate and save visualisations."""
    print(f"\nPlots:")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_idx = int(np.argmin(scores))
    worst_idx = int(np.argmax(scores))

    print(f"Best index: {best_idx} | Worst index: {worst_idx}")

    
    best_worst_plotter = PlotterFactory.create('best_worst', config)
    
    data = {
        'y_true_best': clean_images[best_idx],
        'x_best': degraded_images[best_idx],
        'z_K_best': reconstructed_images[best_idx],
        'score_best': scores[best_idx],
        'y_true_worst': clean_images[worst_idx],
        'x_worst': degraded_images[worst_idx],
        'z_K_worst': reconstructed_images[worst_idx],
        'score_worst': scores[worst_idx],
    }
    
    output_file = output_dir / 'reconstruction_best_worst.png'
    best_worst_plotter.plot(data, str(output_file))


def main(args):
    """Main experiment orchestration."""
    print(f"\n----- Experiment Configuration -----")
    print(f"Data source: {args.data_source}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Solver: {args.solver}")
    print(f"Iterations: {args.k_iterations}")
    print(f"Sparsity parameter: {args.rho_param}")
    print(f"Delta (significance level): {args.delta}")
    
    data_pipeline = setup_data_pipeline(args)
    solver_instance = setup_solver(args)
    
    scores, clean_images, degraded_images, reconstructed_images = run_experiment(
        data_pipeline,
        solver_instance,
        n_samples=args.n_samples,
        seed=args.seed
    )
    
    conformal_cfg = ConformalConfig(
        delta=args.delta,
        calibration_ratio=args.calibration_ratio
    )
    
    predictor, cal_scores, test_scores, coverage, q_hat = run_conformal_prediction(
        scores,
        conformal_cfg
    )
    
    plot_cfg = PlotConfig(figsize=(18, 12), dpi=150)
    output_dir = Path(args.output_dir)
    
    if args.visualise:
        generate_visualisations(
            clean_images,
            degraded_images,
            reconstructed_images,
            scores,
            output_dir,
            plot_cfg
        )

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Image deblurring with conformal prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=500,
        help='Number of images to process'
    )
    parser.add_argument(
        '--data-source',
        type=str,
        choices=['mnist', 'synthetic'],
        default='mnist',
        help='Data source'
    )
    parser.add_argument(
        '--k-iterations',
        type=int,
        default=100,
        help='Number of solver iterations'
    )
    parser.add_argument(
        '--rho-param',
        type=float,
        default=1e-4,
        help='Sparsity parameter'
    )
    parser.add_argument(
        '--kernel-size',
        type=int,
        default=8,
        help='Blur kernel size'
    )
    parser.add_argument(
        '--blur-std-dev',
        type=float,
        default=1.6,
        help='Blur kernel standard deviation'
    )
    parser.add_argument(
        '--noise-std-dev',
        type=float,
        default=1e-3,
        help='Noise standard deviation'
    )
    parser.add_argument(
        '--add-noise',
        action='store_true',
        default=True,
        help='Add noise to degraded images'
    )
    parser.add_argument(
        '--solver',
        type=str,
        choices=['ista', 'fista'],
        default='ista',
        help='Optimisation solver'
    )
    parser.add_argument(
        '--delta',
        type=float,
        default=0.1,
        help='Significance level for conformal prediction'
    )
    parser.add_argument(
        '--calibration-ratio',
        type=float,
        default=0.5,
        help='Proportion of data for calibration'
    )
    parser.add_argument(
        '--visualise',
        action='store_true',
        default=True,
        help='Generate visualisations'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
