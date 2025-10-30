"""
Base experiment class for conformal prediction experiments.
"""

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import json
from datetime import datetime

from .data import ImageConfig, BlurConfig, NoiseConfig, DataPipeline, MNISTSource, SyntheticSource
from .solver import ISTAConfig, SolverFactory
from .conformal import ConformalConfig, ConformalFactory, ConformalAnalyser, MetricConfig
from .plotter import PlotConfig, PlotterFactory
from .conformal import MetricFactory


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    n_samples: int = 500
    calibration_ratio: float = 0.5
    delta: float = 0.1
    data_source: str = 'mnist'
    k_iterations: int = 100
    rho_param: float = 1e-4
    kernel_size: int = 8
    blur_std_dev: float = 1.6
    noise_std_dev: float = 1e-3
    seed: int = 42
    output_dir: str = './results'
    visualise: bool = True
    solver: str = 'ista'
    metric: str = 'nmse'


class BaseExperiment:
    """Base class for conformal prediction experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_pipeline()
        self._setup_solver()
        self._setup_metric()
    
    @classmethod
    def get_config(cls, args) -> ExperimentConfig:
        """Create experiment config from parsed arguments."""
        raise NotImplementedError
    
    @classmethod
    def get_solver_name(cls) -> str:
        """Return solver for the experiment. Should be overridden."""
        return 'solver_name'
    
    @classmethod
    def get_metric_name(cls) -> str:
        """Return metric for the experiment. Should be overridden."""
        return 'metric_name'
    
    @classmethod
    def get_metric_config(cls) -> MetricConfig:
        """
        Return metric configuration for the experiment.
        Override to customize metric parameters per experiment.
        
        Returns:
            MetricConfig instance.
        """
        return MetricConfig()
    
    def _setup_pipeline(self) -> None:
        """Initialise data pipeline."""
        image_cfg = ImageConfig(shape=(28, 28), dtype=jnp.float32)
        blur_cfg = BlurConfig(
            kernel_size=self.config.kernel_size,
            std_dev=self.config.blur_std_dev
        )
        noise_cfg = NoiseConfig(
            std_dev=self.config.noise_std_dev,
            enabled=True
        )
        
        if self.config.data_source == 'mnist':
            source = MNISTSource(image_cfg, seed=self.config.seed)
        elif self.config.data_source == 'synthetic':
            source = SyntheticSource(image_cfg, seed=self.config.seed)
        else:
            raise ValueError(f"Unknown data source: {self.config.data_source}")
        
        self.data_pipeline = DataPipeline(image_cfg, blur_cfg, noise_cfg, source, seed=self.config.seed)
    
    def _setup_solver(self) -> None:
        """Initialise solver."""
        solver_name = self.get_solver_name()
        self.config.solver = solver_name
        
        solver_cfg = ISTAConfig(
            max_iterations=self.config.k_iterations,
            sparsity_param=self.config.rho_param,
            step_size=None,
            step_size_factor=0.99
        )
        self.solver = SolverFactory.create(solver_name, solver_cfg)
    
    def _setup_metric(self) -> None:
        """Initialise metric."""
        metric_name = self.get_metric_name()
        self.config.metric = metric_name
        metric_cfg = self.get_metric_config()
        self.metric = MetricFactory.create(metric_name, metric_cfg)
    
    def run_solver_on_batch(
        self,
        degraded_images: jnp.ndarray,
        clean_images: jnp.ndarray
    ) -> Tuple[np.ndarray, jnp.ndarray]:
        """
        Run solver on a batch of images.
        
        Returns:
            scores: Array of nonconformity scores.
            reconstructed_images: Array of reconstructed images.
        """
        blur_op = self.data_pipeline.get_blur_operator()
        lipschitz_constant = blur_op.get_lipschitz_constant()
        
        def reconstruct_single(degraded, clean):
            reconstructed = self.solver.solve(
                forward_op=blur_op.apply,
                adjoint_op=blur_op.apply_adjoint,
                measurement=degraded,
                lipschitz_constant=lipschitz_constant # type: ignore   # TODO  fix
            )
            score = self.metric.compute(reconstructed, clean)
            return score, reconstructed
        
        reconstruct_vmapped = jax.vmap(reconstruct_single)
        scores, reconstructed_images = reconstruct_vmapped(degraded_images, clean_images)
        
        return np.array(scores), reconstructed_images
    
    def compute_nonconformity_scores(
        self,
        cal_images: jnp.ndarray,
        cal_degraded: jnp.ndarray,
        cal_scores: np.ndarray
    ) -> np.ndarray:
        """
        Compute nonconformity scores for calibration set.
        Override this in subclass to implement different nonconformity functions.
        
        Args:
            cal_images: Clean calibration images.
            cal_degraded: Degraded calibration images.
            cal_scores: NMSE scores on calibration set.
        
        Returns:
            Array of nonconformity scores.
        """
        raise NotImplementedError("Subclasses must implement compute_nonconformity_scores")
    
    def _print_setup(self) -> None:
        """Print experiment configuration."""
        print(f"----- Experiment: {self.__class__.__name__} -----")
        print(f"Configuration:")
        print(f"  Data source: {self.config.data_source}")
        print(f"  Number of samples: {self.config.n_samples}")
        print(f"  Calibration ratio: {self.config.calibration_ratio}")
        print(f"  Delta (significance level): {self.config.delta}")
        print(f"  Solver: {self.config.solver}")
        print(f"  Metric: {self.config.metric}")
        print(f"  Solver iterations: {self.config.k_iterations}")
        print(f"  Sparsity parameter: {self.config.rho_param}")
        print(f"  Kernel size: {self.config.kernel_size}")
        print(f"  Blur std dev: {self.config.blur_std_dev}")
        print(f"  Noise std dev: {self.config.noise_std_dev}")
        print(f"  Output directory: {self.config.output_dir}")
    
    def run(self) -> Dict[str, Any]:
        """Execute full experiment."""
        # self._print_setup()

        clean_images = self.data_pipeline.load_clean_images(self.config.n_samples)
        degraded_images, _ = self.data_pipeline.apply_degradation(
            clean_images, 
            seed=self.config.seed
        )
        print(f"Data successfully degraded.")
        
        print(f"\nRunning solver on all images...")
        scores, reconstructed_images = self.run_solver_on_batch(
            degraded_images,
            clean_images
        )
        print(f"Solver complete.")
        

        np.random.seed(self.config.seed)
        indices = np.random.permutation(self.config.n_samples)
        n_cal = int(np.ceil(self.config.n_samples * self.config.calibration_ratio))
        
        cal_indices = indices[:n_cal]
        test_indices = indices[n_cal:]
        
        cal_images = clean_images[cal_indices]
        cal_degraded = degraded_images[cal_indices]
        cal_scores = scores[cal_indices]
        cal_reconstructed = reconstructed_images[cal_indices]
        
        test_images = clean_images[test_indices]
        test_degraded = degraded_images[test_indices]
        test_scores = scores[test_indices]
        test_reconstructed = reconstructed_images[test_indices]
        
        print(f"Calibration set: {len(cal_indices)} samples")
        print(f"Test set: {len(test_indices)} samples")
        
        cal_nonconformity = self.compute_nonconformity_scores(
            cal_images,
            cal_degraded,
            cal_scores
        )
        print(f"Nonconformity scores computed:")
        print(f"  Mean: {np.mean(cal_nonconformity):.4f}")
        print(f"  Std: {np.std(cal_nonconformity):.4f}")
        print(f"  Min: {np.min(cal_nonconformity):.4f}")
        print(f"  Max: {np.max(cal_nonconformity):.4f}")
        
        print(f"\nCalibrating conformal predictor...")
        conformal_cfg = ConformalConfig(
            delta=self.config.delta,
            calibration_ratio=1.0
        )
        conformal_predictor = ConformalFactory.create('standard', conformal_cfg)
        q_hat = conformal_predictor.calibrate(cal_nonconformity)
        print(f"Quantile q_hat: {q_hat:.4f}")
        
        coverage = conformal_predictor.evaluate_coverage(cal_nonconformity)
        
        is_valid, status = ConformalAnalyser.validate(
            coverage,
            1 - self.config.delta
        )
        print(f"Calibration coverage: {coverage:.4f}")
        print(f"Target coverage: {1 - self.config.delta:.4f}")
        print(f"Result: {status}")
        
        results = {
            'config': {
                'experiment': self.__class__.__name__,
                'n_samples': self.config.n_samples,
                'calibration_ratio': self.config.calibration_ratio,
                'delta': self.config.delta,
                'data_source': self.config.data_source,
                'k_iterations': self.config.k_iterations,
            },
            'calibration': {
                'n_cal': len(cal_indices),
                'mean_score': float(np.mean(cal_scores)),
                'std_score': float(np.std(cal_scores)),
                'mean_nonconformity': float(np.mean(cal_nonconformity)),
                'std_nonconformity': float(np.std(cal_nonconformity)),
            },
            'test': {
                'n_test': len(test_indices),
                'mean_score': float(np.mean(test_scores)),
                'std_score': float(np.std(test_scores)),
            },
            'conformal': {
                'q_hat': float(q_hat),
                'calibration_coverage': float(coverage),
                'target_coverage': 1 - self.config.delta,
                'is_valid': bool(is_valid),
            }
        }
        
        if self.config.visualise:
            print(f"\nGenerating visualisations...")
            self._generate_visualisations(
                clean_images,
                degraded_images,
                reconstructed_images,
                scores,
                test_indices
            )
        
        self._save_results(results)
        
        return results
    
    def _generate_visualisations(
        self,
        clean_images: jnp.ndarray,
        degraded_images: jnp.ndarray,
        reconstructed_images: jnp.ndarray,
        scores: np.ndarray,
        test_indices: np.ndarray
    ) -> None:
        """Generate and save visualisations."""
        test_scores = scores[test_indices]
        
        best_test_idx = np.argmin(test_scores)
        worst_test_idx = np.argmax(test_scores)
        
        best_global_idx = test_indices[best_test_idx]
        worst_global_idx = test_indices[worst_test_idx]
        
        plot_cfg = PlotConfig(figsize=(18, 12), dpi=150)
        best_worst_plotter = PlotterFactory.create('best_worst', plot_cfg)
        
        data = {
            'y_true_best': clean_images[best_global_idx],
            'x_best': degraded_images[best_global_idx],
            'z_K_best': reconstructed_images[best_global_idx],
            'score_best': scores[best_global_idx],
            'y_true_worst': clean_images[worst_global_idx],
            'x_worst': degraded_images[worst_global_idx],
            'z_K_worst': reconstructed_images[worst_global_idx],
            'score_worst': scores[worst_global_idx],
        }
        
        output_file = self.output_dir / f'{self.__class__.__name__}_best_worst.png'
        best_worst_plotter.plot(data, str(output_file), metric_name=self.config.metric.upper()) # type: ignore  # TODO fix
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save experiment results to JSON."""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        experiment_name = self.__class__.__name__
        filename = self.output_dir / f'{experiment_name}_{timestamp}.json'
        
        output_data = {
            'timestamp': timestamp,
            **results
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to: {filename}")
