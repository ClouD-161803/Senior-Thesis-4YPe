import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class MetricConfig:
    """Configuration for metrics."""
    max_pixel_value: float = 1.0


class Metric(ABC):
    """Abstract base for image quality metrics."""
    
    @abstractmethod
    def compute(self, reconstructed: jnp.ndarray, ground_truth: jnp.ndarray) -> jnp.ndarray:
        """Compute metric."""
        pass


class NMSE(Metric):
    """Normalised Mean-Squared Error in decibels."""
    
    def compute(self, reconstructed: jnp.ndarray, ground_truth: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate NMSE in dB.
        
        Args:
            reconstructed: Reconstructed image.
            ground_truth: Ground truth image.
        
        Returns:
            NMSE value in dB.
        """
        mse = jnp.mean((reconstructed - ground_truth)**2)
        norm_gt_sq = jnp.mean(ground_truth**2)
        nmse = mse / norm_gt_sq
        return 10 * jnp.log10(nmse)


class PSNR(Metric):
    """Peak Signal-to-Noise Ratio."""
    
    def __init__(self, config: MetricConfig):
        self.config = config
    
    def compute(self, reconstructed: jnp.ndarray, ground_truth: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate PSNR.
        
        Args:
            reconstructed: Reconstructed image.
            ground_truth: Ground truth image.
        
        Returns:
            PSNR value in dB.
        """
        mse = jnp.mean((reconstructed - ground_truth)**2)
        psnr_val = jnp.where(
            mse == 0,
            float('inf'),
            20 * jnp.log10(self.config.max_pixel_value / jnp.sqrt(mse))
        )
        return psnr_val


class MetricFactory:
    """Factory for creating metrics."""
    
    _metrics = {
        'nmse': NMSE,
        'psnr': PSNR,
    }
    
    @classmethod
    def create(cls, metric_name: str, config: Optional[MetricConfig] = None) -> Metric:
        """Create metric by name."""
        if metric_name not in cls._metrics:
            raise ValueError(f"Unknown metric: {metric_name}. Available: {list(cls._metrics.keys())}")
        
        metric_class = cls._metrics[metric_name]
        if config is None:
            config = MetricConfig()
        
        if metric_name == 'psnr':
            return metric_class(config)
        return metric_class()
    
    @classmethod
    def register(cls, name: str, metric_class: type) -> None:
        """Register new metric."""
        cls._metrics[name] = metric_class


def conformal_upper_quantile(values: np.ndarray, q: float) -> float:
    """
    Return the distribution-free conformal upper q-quantile bound based on calibration values.
    Uses p = ceil(q*(n+1)). If p >= n+1, return +inf.
    Otherwise return the p-th smallest value (1-indexed), i.e. sorted_values[p-1].
    """
    n = len(values)
    p = int(np.ceil(q * (n + 1)))
    if p >= n + 1:
        return float('inf')
    sorted_values = np.sort(values)
    return float(sorted_values[p - 1])


@dataclass
class ConformalConfig:
    """Configuration for conformal prediction."""
    delta: float = 0.1
    calibration_ratio: float = 0.5


class ConformalRegressor:
    """Standard conformal predictor for regression."""
    
    def __init__(self, config: ConformalConfig):
        self.config = config
        self.q_hat = None
    
    def calibrate(self, calibration_scores: np.ndarray) -> float:
        """
        Compute quantile from calibration scores.
        
        Args:
            calibration_scores: Array of nonconformity scores on calibration set.
        
        Returns:
            Quantile q_hat (upper confidence bound).
        """
        n_cal = len(calibration_scores)
        quantile_level = np.ceil((n_cal + 1) * (1 - self.config.delta)) / n_cal
        quantile_level = min(quantile_level, 1.0)
        
        self.q_hat = float(np.quantile(calibration_scores, quantile_level))
        return self.q_hat
    
    def predict_set(self, test_score: float) -> Tuple[bool, float]:
        """
        Compute prediction set for single test point.
        
        Args:
            test_score: Nonconformity score for test point.
        
        Returns:
            (is_in_set, q_hat) where is_in_set indicates if score <= q_hat.
        """
        if self.q_hat is None:
            raise ValueError("Must call calibrate() first")
        return test_score <= self.q_hat, self.q_hat
    
    def evaluate_coverage(self, test_scores: np.ndarray) -> float:
        """
        Evaluate empirical coverage on test set.
        
        Args:
            test_scores: Array of nonconformity scores on test set.
        
        Returns:
            Empirical coverage (proportion of scores in prediction set).
        """
        if self.q_hat is None:
            raise ValueError("Must call calibrate() first")
        
        coverage = np.mean(test_scores <= self.q_hat)
        return float(coverage)


class AdaptiveConformalRegressor(ConformalRegressor):
    """Adaptive conformal predictor with online recalibration."""
    
    def __init__(self, config: ConformalConfig):
        super().__init__(config)
        self.seen_scores = []
    
    def calibrate(self, calibration_scores: np.ndarray) -> float:
        """Calibrate and store scores for adaptation."""
        self.seen_scores = list(calibration_scores)
        return super().calibrate(calibration_scores)
    
    def adapt(self, new_score: float) -> None:
        """Update calibration with new observed score."""
        self.seen_scores.append(new_score)
        # Recalibrate quantile
        scores_array = np.array(self.seen_scores)
        super().calibrate(scores_array)


class ConformalFactory:
    """Factory for creating conformal predictors."""
    
    _predictors = {
        'standard': ConformalRegressor,
        'adaptive': AdaptiveConformalRegressor,
    }
    
    @classmethod
    def create(cls, predictor_name: str, config: ConformalConfig) -> ConformalRegressor:
        """Create conformal predictor by name."""
        if predictor_name not in cls._predictors:
            raise ValueError(f"Unknown predictor: {predictor_name}. Available: {list(cls._predictors.keys())}")
        return cls._predictors[predictor_name](config)
    
    @classmethod
    def register(cls, name: str, predictor_class: type) -> None:
        """Register new conformal predictor."""
        cls._predictors[name] = predictor_class


class ConformalAnalyser:
    """Utility for analysing conformal prediction results."""
    
    @staticmethod
    def split_data(
        scores: np.ndarray,
        calibration_ratio: float = 0.5,
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split scores into calibration and test sets.
        
        Args:
            scores: All nonconformity scores.
            calibration_ratio: Proportion for calibration.
            seed: Random seed for reproducibility.
        
        Returns:
            (calibration_scores, test_scores).
        """
        np.random.seed(seed)
        np.random.shuffle(scores)
        
        n_cal = int(np.ceil(len(scores) * calibration_ratio))
        return scores[:n_cal], scores[n_cal:]
    
    @staticmethod
    def validate(
        coverage: float,
        target_coverage: float,
        tolerance: float = 0.01
    ) -> Tuple[bool, str]:
        """
        Validate if coverage meets target.
        
        Args:
            coverage: Empirical coverage achieved.
            target_coverage: Target coverage level (e.g., 1 - delta).
            tolerance: Acceptable deviation from target.
        
        Returns:
            (is_valid, message).
        """
        is_valid = coverage >= (target_coverage - tolerance)
        message = f"VALID" if is_valid else f"INVALID"
        return is_valid, message
    
    @staticmethod
    def print_summary(
        delta: float,
        q_hat: float,
        coverage: float,
        n_cal: int,
        n_test: int
    ) -> None:
        """Print summary of conformal prediction results."""
        print(f"\n--- Conformal Prediction Summary ---")
        print(f"Significance level (δ): {delta:.2f}")
        print(f"Target coverage: {1-delta:.2f}")
        print(f"Quantile q̂: {q_hat:.4f}")
        print(f"Empirical coverage: {coverage:.4f}")
        print(f"Calibration set size: {n_cal}")
        print(f"Test set size: {n_test}")
        
        is_valid, status = ConformalAnalyser.validate(coverage, 1-delta)
        print(f"Result: {status}")
