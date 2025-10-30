import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Callable, Optional


@dataclass
class SolverConfig:
    """Base configuration for solvers."""
    max_iterations: int = 100
    tolerance: float = 1e-6


@dataclass
class ISTAConfig(SolverConfig):
    """Configuration for ISTA-based solvers."""
    sparsity_param: float = 1e-4
    step_size: Optional[float] = None
    step_size_factor: float = 0.99


class ProximalOperator(ABC):
    """Abstract base for proximal operators."""
    
    @abstractmethod
    def __call__(self, u: jnp.ndarray, step_size: float) -> jnp.ndarray:
        """Apply proximal operator."""
        pass


class SoftThreshold(ProximalOperator):
    """Soft-thresholding operator S_αρ(u)."""
    
    def __init__(self, sparsity_param: float):
        self.sparsity_param = sparsity_param
    
    def __call__(self, u: jnp.ndarray, step_size: float) -> jnp.ndarray:
        threshold = self.sparsity_param * step_size
        return jnp.sign(u) * jnp.maximum(jnp.abs(u) - threshold, 0)


class BoxConstraint(ProximalOperator):
    """Projection onto [0, 1] box constraint."""
    
    def __call__(self, z: jnp.ndarray, step_size: float) -> jnp.ndarray:
        return jnp.clip(z, 0, 1)


class Solver(ABC):
    """Abstract base class for optimisation solvers."""
    
    def __init__(self, config: SolverConfig):
        self.config = config
    
    @abstractmethod
    def solve(
        self,
        forward_op: Callable,
        adjoint_op: Callable,
        measurement: jnp.ndarray
    ) -> jnp.ndarray:
        """Solve inverse problem: min_x ||Ax - b||^2 + R(x)."""
        pass


class ISTA(Solver):
    """Iterative Soft-Thresholding Algorithm."""
    
    def __init__(self, config: ISTAConfig):
        super().__init__(config)
        self.config = config
        self.soft_threshold = SoftThreshold(config.sparsity_param)
        self.box_proj = BoxConstraint()
    
    def solve(
        self,
        forward_op: Callable,
        adjoint_op: Callable,
        measurement: jnp.ndarray,
        lipschitz_constant: Optional[float] = None
    ) -> jnp.ndarray:
        """
        Solve using ISTA.
        
        Args:
            forward_op: Forward operator A (blur).
            adjoint_op: Adjoint operator A^T.
            measurement: Measurement b (blurred + noisy image).
            lipschitz_constant: Lipschitz constant for step size selection.
        
        Returns:
            Final solution.
        """
        step_size = self._compute_step_size(lipschitz_constant)
        
        def ist_step(z, _):
            grad = adjoint_op(forward_op(z) - measurement)
            z_thresh = self.soft_threshold(z - step_size * grad, step_size)
            z_proj = self.box_proj(z_thresh, step_size)
            return z_proj, z_proj
        
        z0 = jnp.zeros_like(measurement)
        _, iterates = jax.lax.scan(
            ist_step,
            z0,
            None,
            length=self.config.max_iterations
        )
        
        return iterates[-1]
    
    def solve_with_history(
        self,
        forward_op: Callable,
        adjoint_op: Callable,
        measurement: jnp.ndarray,
        lipschitz_constant: Optional[float] = None
    ) -> jnp.ndarray:
        """Solve and return full iterate history."""
        step_size = self._compute_step_size(lipschitz_constant)
        
        def ist_step(z, _):
            grad = adjoint_op(forward_op(z) - measurement)
            z_thresh = self.soft_threshold(z - step_size * grad, step_size)
            z_proj = self.box_proj(z_thresh, step_size)
            return z_proj, z_proj
        
        z0 = jnp.zeros_like(measurement)
        _, iterates = jax.lax.scan(
            ist_step,
            z0,
            None,
            length=self.config.max_iterations
        )
        
        return iterates
    
    def _compute_step_size(self, lipschitz_constant: Optional[float]) -> float:
        if self.config.step_size is not None:
            return self.config.step_size
        
        if lipschitz_constant is None:
            raise ValueError("Either step_size or lipschitz_constant must be provided")
        
        return self.config.step_size_factor / lipschitz_constant


class FISTA(Solver):
    """Fast Iterative Soft-Thresholding Algorithm."""
    
    def __init__(self, config: ISTAConfig):
        super().__init__(config)
        self.config = config
        self.soft_threshold = SoftThreshold(config.sparsity_param)
        self.box_proj = BoxConstraint()
    
    def solve(
        self,
        forward_op: Callable,
        adjoint_op: Callable,
        measurement: jnp.ndarray,
        lipschitz_constant: Optional[float] = None
    ) -> jnp.ndarray:
        """
        Solve using FISTA (accelerated ISTA).
        
        Args:
            forward_op: Forward operator A (blur).
            adjoint_op: Adjoint operator A^T.
            measurement: Measurement b (blurred + noisy image).
            lipschitz_constant: Lipschitz constant for step size selection.
        
        Returns:
            Final solution.
        """
        step_size = self._compute_step_size(lipschitz_constant)
        
        def fista_step(carry, _):
            z, z_prev, t = carry
            grad = adjoint_op(forward_op(z) - measurement)
            z_new_thresh = self.soft_threshold(z - step_size * grad, step_size)
            z_new = self.box_proj(z_new_thresh, step_size)
            t_new = (1 + jnp.sqrt(1 + 4 * t**2)) / 2
            z_accel = z_new + ((t - 1) / t_new) * (z_new - z_prev)
            return (z_accel, z_new, t_new), z_new
        
        z0 = jnp.zeros_like(measurement)
        carry = (z0, z0, 1.0)
        (z_final, _, _), _ = jax.lax.scan(
            fista_step,
            carry,
            None,
            length=self.config.max_iterations
        )
        
        return z_final
    
    def _compute_step_size(self, lipschitz_constant: Optional[float]) -> float:
        if self.config.step_size is not None:
            return self.config.step_size
        
        if lipschitz_constant is None:
            raise ValueError("Either step_size or lipschitz_constant must be provided")
        
        return self.config.step_size_factor / lipschitz_constant


class SolverFactory:
    """Factory for creating solvers."""
    
    _solvers = {
        'ista': ISTA,
        'fista': FISTA,
    }
    
    @classmethod
    def create(cls, solver_name: str, config: SolverConfig) -> Solver:
        """Create solver by name."""
        if solver_name not in cls._solvers:
            raise ValueError(f"Unknown solver: {solver_name}. Available: {list(cls._solvers.keys())}")
        return cls._solvers[solver_name](config)
    
    @classmethod
    def register(cls, name: str, solver_class: type) -> None:
        """Register new solver."""
        cls._solvers[name] = solver_class
