# Senior Thesis Project

## Image Deblurring Experiment

Modular framework for image reconstruction with conformal prediction-based uncertainty quantification.

### Quick Start

```bash
cd code
python experiments/nmse.py --n-samples 100
python experiments/regressor.py --n-samples 100
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
--n-samples 500            # Number of images
--data-source mnist        # mnist or synthetic
--k-iterations 100         # Solver iterations
--rho-param 1e-4           # Sparsity parameter
--delta 0.1                # Significance level
--calibration-ratio 0.5    # Cal/test split
--kernel-size 8            # Blur kernel size
--blur-std-dev 1.6         # Blur std dev
--noise-std-dev 1e-3       # Noise std dev
--output-dir ./results     # Output directory
--seed 42                  # Random seed
```

## Adding New Experiments

```python
# experiments/my_exp.py
from utils.experiment import BaseExperiment, ExperimentConfig
from utils.main import main, parse_arguments

class MyExperiment(BaseExperiment):
    @classmethod
    def get_config(cls, args) -> ExperimentConfig:
        return ExperimentConfig(..., solver=cls.get_solver_name(), metric=cls.get_metric_name())
    
    @classmethod
    def get_solver_name(cls) -> str:
        return 'solver_name'  # currently implemented: ista, fista

    @classmethod
    def get_metric_name(cls) -> str:
        return 'metric_name'  # currently implemented: nmse, psnr
    
    @classmethod
    def get_solver_config(cls) -> SolverConfig:
        """Optional: customize solver parameters"""
        return ISTAConfig(max_iterations=100, sparsity_param=1e-4, step_size=None, step_size_factor=0.99)
    
    def compute_nonconformity_scores(self, cal_images, cal_degraded, cal_scores):
        # Your nonconformity function
        return nonconformity_scores

if __name__ == '__main__':
    args = parse_arguments()
    main(MyExperiment, args)
```

Run: `python experiments/my_exp.py`