import jax.numpy as jnp
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any


@dataclass
class PlotConfig:
    """Base configuration for plotting."""
    figsize: Tuple[int, int] = (18, 6)
    cmap: str = 'gray'
    vmin: float = 0.0
    vmax: float = 1.0
    dpi: int = 150


class ImageNormaliser:
    """Utility for normalising images to display range."""
    
    @staticmethod
    def normalise(img: jnp.ndarray, target_min: float = 0, target_max: float = 1) -> jnp.ndarray:
        """Normalise image to [target_min, target_max]."""
        img_min = jnp.min(img)
        img_max = jnp.max(img)
        
        if img_max == img_min:
            return jnp.ones_like(img) * target_min
        
        normalised = (img - img_min) / (img_max - img_min)
        return normalised * (target_max - target_min) + target_min


class Plotter(ABC):
    """Abstract base class for image plotters."""
    
    def __init__(self, config: PlotConfig):
        self.config = config
    
    @abstractmethod
    def plot(self, data: Dict[str, Any], filename: str) -> None:
        """Plot and save figure."""
        pass


class ReconstructionPlotter(Plotter):
    """Plots single reconstruction (ground truth, blurred input, reconstruction)."""
    
    def plot(self, data: Dict[str, Any], filename: str) -> None:
        """
        Plot single reconstruction.
        
        Args:
            data: Dictionary with keys 'y_true', 'x', 'z_K'.
            filename: Output filename.
        """
        y_true_norm = ImageNormaliser.normalise(data['y_true'], self.config.vmin, self.config.vmax)
        x_norm = ImageNormaliser.normalise(data['x'], self.config.vmin, self.config.vmax)
        z_K_norm = ImageNormaliser.normalise(data['z_K'], self.config.vmin, self.config.vmax)
        
        fig, axes = plt.subplots(1, 3, figsize=self.config.figsize)
        
        axes[0].imshow(y_true_norm, cmap=self.config.cmap, vmin=self.config.vmin, vmax=self.config.vmax)
        axes[0].set_title('Ground Truth (y*)', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(x_norm, cmap=self.config.cmap, vmin=self.config.vmin, vmax=self.config.vmax)
        axes[1].set_title('Blurred & Noisy (x)', fontsize=14)
        axes[1].axis('off')
        
        axes[2].imshow(z_K_norm, cmap=self.config.cmap, vmin=self.config.vmin, vmax=self.config.vmax)
        axes[2].set_title('Reconstructed (z_K)', fontsize=14)
        axes[2].axis('off')
        
        fig.suptitle('Image Reconstruction', fontsize=18, fontweight='bold')
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        
        try:
            plt.savefig(filename, dpi=self.config.dpi)
            print(f"Saved: {filename}")
        except Exception as e:
            print(f"Error saving {filename}: {e}")
        finally:
            plt.close(fig)


class BestWorstPlotter(Plotter):
    """Plots best and worst reconstructions side-by-side."""
    
    def plot(self, data: Dict[str, Any], filename: str, metric_name: str = 'NMSE') -> None:
        """
        Plot best and worst reconstructions.
        
        Args:
            data: Dictionary with keys:
                - 'y_true_best', 'x_best', 'z_K_best', 'score_best'
                - 'y_true_worst', 'x_worst', 'z_K_worst', 'score_worst'
            filename: Output filename.
            metric_name: Name of metric for display.
        """
        y_best = ImageNormaliser.normalise(data['y_true_best'], self.config.vmin, self.config.vmax)
        x_best = ImageNormaliser.normalise(data['x_best'], self.config.vmin, self.config.vmax)
        z_best = ImageNormaliser.normalise(data['z_K_best'], self.config.vmin, self.config.vmax)
        
        y_worst = ImageNormaliser.normalise(data['y_true_worst'], self.config.vmin, self.config.vmax)
        x_worst = ImageNormaliser.normalise(data['x_worst'], self.config.vmin, self.config.vmax)
        z_worst = ImageNormaliser.normalise(data['z_K_worst'], self.config.vmin, self.config.vmax)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Best row
        axes[0, 0].imshow(y_best, cmap=self.config.cmap, vmin=self.config.vmin, vmax=self.config.vmax)
        axes[0, 0].set_title('Ground Truth (y*)\n[Best]', fontsize=16, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(x_best, cmap=self.config.cmap, vmin=self.config.vmin, vmax=self.config.vmax)
        axes[0, 1].set_title('Blurred & Noisy (x)\n[Best]', fontsize=16, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(z_best, cmap=self.config.cmap, vmin=self.config.vmin, vmax=self.config.vmax)
        axes[0, 2].set_title(
            f'Reconstructed (z_K)\n[Best] {metric_name}: {data["score_best"]:.2f} dB',
            fontsize=16, fontweight='bold', color='green'
        )
        axes[0, 2].axis('off')
        
        # Worst row
        axes[1, 0].imshow(y_worst, cmap=self.config.cmap, vmin=self.config.vmin, vmax=self.config.vmax)
        axes[1, 0].set_title('Ground Truth (y*)\n[Worst]', fontsize=16, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(x_worst, cmap=self.config.cmap, vmin=self.config.vmin, vmax=self.config.vmax)
        axes[1, 1].set_title('Blurred & Noisy (x)\n[Worst]', fontsize=16, fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(z_worst, cmap=self.config.cmap, vmin=self.config.vmin, vmax=self.config.vmax)
        axes[1, 2].set_title(
            f'Reconstructed (z_K)\n[Worst] {metric_name}: {data["score_worst"]:.2f} dB',
            fontsize=16, fontweight='bold', color='red'
        )
        axes[1, 2].axis('off')
        
        fig.suptitle('Image Reconstruction: Best vs Worst', fontsize=20, fontweight='bold')
        plt.tight_layout(rect=(0, 0.02, 1, 0.97))
        
        try:
            plt.savefig(filename, dpi=self.config.dpi)
            print(f"Saved: {filename}")
            print(f"Best {metric_name}: {data['score_best']:.4f} dB | Worst {metric_name}: {data['score_worst']:.4f} dB")
        except Exception as e:
            print(f"Error saving {filename}: {e}")
        finally:
            plt.close(fig)


class PlotterFactory:
    """Factory for creating plotters."""
    
    _plotters = {
        'reconstruction': ReconstructionPlotter,
        'best_worst': BestWorstPlotter,
    }
    
    @classmethod
    def create(cls, plotter_name: str, config: PlotConfig) -> Plotter:
        """Create plotter by name."""
        if plotter_name not in cls._plotters:
            raise ValueError(f"Unknown plotter: {plotter_name}. Available: {list(cls._plotters.keys())}")
        return cls._plotters[plotter_name](config)
    
    @classmethod
    def register(cls, name: str, plotter_class: type) -> None:
        """Register new plotter."""
        cls._plotters[name] = plotter_class
