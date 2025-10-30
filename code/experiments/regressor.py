"""
Conformal Prediction with Regression Experiment

This experiment trains a regression model on the calibration set to predict
performance metrics, then uses the prediction residuals as nonconformity scores.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.experiment import BaseExperiment, ExperimentConfig
from utils.conformal import MetricConfig
from utils.main import run_experiment, parse_arguments


class RegressionExperiment(BaseExperiment):
    
    @classmethod
    def get_config(cls, args) -> ExperimentConfig:
        """Create experiment config from parsed arguments."""
        return ExperimentConfig(
            n_samples=args.n_samples,
            calibration_ratio=args.calibration_ratio,
            delta=args.delta,
            data_source=args.data_source,
            k_iterations=args.k_iterations,
            rho_param=args.rho_param,
            kernel_size=args.kernel_size,
            blur_std_dev=args.blur_std_dev,
            noise_std_dev=args.noise_std_dev,
            seed=args.seed,
            output_dir=args.output_dir,
            visualise=args.visualise,
            solver=args.solver or cls.get_default_solver_name(),
            metric=args.metric or cls.get_default_metric_name()
        )
    
    @classmethod
    def get_default_solver_name(cls) -> str:
        return 'fista'
    
    @classmethod
    def get_default_metric_name(cls) -> str:
        return 'psnr'
    
    @classmethod
    def get_metric_config(cls) -> MetricConfig:
        return MetricConfig(max_pixel_value=1.0)
    
    def extract_features(self, images: jnp.ndarray) -> np.ndarray:
        """
        Extract features from images for regression.
        
        Simple features: mean, std, min, max pixel values
        """
        features = np.array([
            np.mean(images, axis=(1, 2)),
            np.std(images, axis=(1, 2)),
            np.min(images, axis=(1, 2)),
            np.max(images, axis=(1, 2)),
        ]).T
        
        return features
    
    def train_regressor(
        self,
        cal_features: np.ndarray,
        cal_scores: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train a simple linear regression model on calibration set.
        
        Returns:
            Dictionary with regression parameters.
        """
        feature_mean = np.mean(cal_features, axis=0)
        feature_std = np.std(cal_features, axis=0) + 1e-8
        cal_features_norm = (cal_features - feature_mean) / feature_std
        
        cal_features_augmented = np.hstack([
            cal_features_norm,
            np.ones((len(cal_features_norm), 1))
        ])
        
        # Solve least squares: min ||Xw - y||^2
        weights = np.linalg.lstsq(cal_features_augmented, cal_scores, rcond=None)[0]
        
        return {
            'weights': weights[:-1],
            'bias': weights[-1],
            'feature_mean': feature_mean,
            'feature_std': feature_std
        }
    
    def predict_with_regressor(
        self,
        features: np.ndarray,
        regressor: Dict[str, Any]
    ) -> np.ndarray:
        """Use trained regressor to predict scores."""
        features_norm = (
            (features - regressor['feature_mean']) / 
            (regressor['feature_std'] + 1e-8)
        )
        
        predictions = np.dot(features_norm, regressor['weights']) + regressor['bias']
        return predictions
    
    def compute_nonconformity_scores(
        self,
        cal_images: jnp.ndarray,
        cal_degraded: jnp.ndarray,
        cal_scores: np.ndarray
    ) -> np.ndarray:
        """
        Compute nonconformity as absolute prediction residuals.
        
        Args:
            cal_images: Clean calibration images (unused).
            cal_degraded: Degraded calibration images.
            cal_scores: NMSE scores on calibration set.
        
        Returns:
            Array of nonconformity scores (prediction residuals).
        """
        cal_features = self.extract_features(cal_degraded)
        
        regressor = self.train_regressor(cal_features, cal_scores)
        
        cal_predictions = self.predict_with_regressor(cal_features, regressor)
        nonconformity = np.abs(cal_scores - cal_predictions)
        
        return nonconformity


if __name__ == '__main__':
    args = parse_arguments()
    run_experiment(RegressionExperiment, args)