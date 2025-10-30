# Senior Thesis Project

## Image Deblurring Experiment

Modular framework for image reconstruction with conformal prediction-based uncertainty quantification.

### Quick Start

```bash
cd code
python experiments/nmse.py --n-samples 100 --solver ista --metric nmse
python experiments/regressor.py --n-samples 100 --solver fista --metric psnr
```

### Structure

```md
code/
├── utils/
│   ├── data.py
│   ├── solver.py
│   ├── conformal.py
│   ├── plotter.py
│   ├── experiment.py    # Orchestrator
│   └── main.py          # Parser
├── experiments/
│   ├── nmse.py 
│   └── regressor.py 
└── README.md
```

### Current Working Experiments

**NMSEExperiment**: Nonconformity = |NMSE_score - median(NMSE)|
**RegressionExperiment**: Nonconformity = |NMSE_score - predicted_NMSE|

### CLI Arguments

```md
--n-samples 500                # Number of images
--data-source mnist            # mnist or synthetic
--k-iterations 100             # Solver iterations
--rho-param 1e-4               # Sparsity parameter
--delta 0.1                    # Significance level
--calibration-ratio 0.5        # Cal/test split
--kernel-size 8                # Blur kernel size
--blur-std-dev 1.6             # Blur std dev
--noise-std-dev 1e-3           # Noise std dev
--output-dir ./results         # Output directory
--seed 42                      # Random seed
--solver {ista,fista}          # Solver
--metric {nmse,psnr}           # Metric
--visualise                    # Generate visualisations
```

## Adding New Experiments

```python
# experiments/my_exp.py
from utils.experiment import BaseExperiment, ExperimentConfig
from utils.conformal import MetricConfig
from utils.main import run_experiment, parse_arguments

class MyExperiment(BaseExperiment):
    @classmethod
    def get_config(cls, args) -> ExperimentConfig:
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
    def get_solver_name(cls) -> str:
        return 'solver_name'  # currently implemented: ista, fista

    @classmethod
    def get_metric_name(cls) -> str:
        return 'metric_name'  # currently implemented: nmse, psnr
    
    @classmethod
    def get_metric_config(cls) -> MetricConfig:
        """Optional: customise metric parameters"""
        return MetricConfig()
    
    def compute_nonconformity_scores(self, cal_images, cal_degraded, cal_scores):
        # Your nonconformity function
        return nonconformity_scores

if __name__ == '__main__':
    args = parse_arguments()
    run_experiment(MyExperiment, args)
```

Run: `python experiments/my_exp.py`
