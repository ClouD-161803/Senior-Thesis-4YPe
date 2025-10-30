import argparse
from pathlib import Path


def main(experiment_class, args):
    """Generic experiment orchestrator."""
    print(f"----- Experiment: {experiment_class.__name__} -----")
    print(f"Data source: {args.data_source}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Solver: {experiment_class.get_solver_name()}")
    print(f"Solver iterations: {args.k_iterations}")
    print(f"Significance Level: {args.delta}")
    print(f"Random seed: {args.seed}")
    
    config = experiment_class.get_config(args)
    
    experiment = experiment_class(config)
    results = experiment.run()
    
    print(f"\nFinished.\n")


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
