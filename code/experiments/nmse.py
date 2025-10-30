import numpy as np
import jax.numpy as jnp
from typing import Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.experiment import BaseExperiment, ExperimentConfig
from utils.main import main, parse_arguments


class NMSEExperiment(BaseExperiment):
    """
    Conformal prediction experiment using NMSE residuals as nonconformity scores.
    
    Nonconformity function: |NMSE_score - baseline_NMSE|
    where baseline_NMSE is the median NMSE on calibration set.
    """
    
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
            solver=cls.get_solver_name()
        )
    
    @classmethod
    def get_solver_name(cls) -> str:
        return 'ista'
    
    def compute_nonconformity_scores(
        self,
        cal_images: jnp.ndarray,
        cal_degraded: jnp.ndarray,
        cal_scores: np.ndarray
    ) -> np.ndarray:
        """
        Compute nonconformity as absolute deviation from median NMSE.
        
        Args:
            cal_images: Clean calibration images (unused in this experiment).
            cal_degraded: Degraded calibration images (unused in this experiment).
            cal_scores: NMSE scores on calibration set.
        
        Returns:
            Array of nonconformity scores.
        """
        baseline = np.median(cal_scores)
        nonconformity = np.abs(cal_scores - baseline)
        return nonconformity


if __name__ == '__main__':
    args = parse_arguments()
    main(NMSEExperiment, args)
