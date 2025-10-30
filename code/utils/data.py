import jax
import jax.numpy as jnp
import torch
import torchvision
from torchvision.transforms import ToTensor
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, Callable


@dataclass
class ImageConfig:
    """Configuration for image properties."""
    shape: Tuple[int, int] = (28, 28)
    dtype: jnp.dtype = jnp.float32


@dataclass
class NoiseConfig:
    """Configuration for noise addition."""
    std_dev: float = 1e-3
    enabled: bool = True


@dataclass
class BlurConfig:
    """Configuration for blur kernel."""
    kernel_size: int = 8
    std_dev: float = 1.6


class ImageSource(ABC):
    """Abstract base class for image sources."""
    
    @abstractmethod
    def load(self, n_samples: int) -> jnp.ndarray:
        """Load n_samples images of shape (n_samples, *image_shape)."""
        pass


class MNISTSource(ImageSource):
    """MNIST dataset source."""
    
    def __init__(self, config: ImageConfig):
        self.config = config
    
    def load(self, n_samples: int) -> jnp.ndarray:
        """Load MNIST images from torchvision."""
        print("Loading MNIST data...")
        dataset = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=ToTensor()
        )
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=n_samples,
            shuffle=True
        )
        
        images, _ = next(iter(loader))
        images_np = images.squeeze().numpy()
        print(f"MNIST data loaded: {images_np.shape}")
        
        return jnp.array(images_np, dtype=self.config.dtype)


class SyntheticSource(ImageSource):
    """Synthetic image source (simple patterns)."""
    
    def __init__(self, config: ImageConfig):
        self.config = config
    
    def load(self, n_samples: int) -> jnp.ndarray:
        """Generate synthetic images."""
        print(f"Generating {n_samples} synthetic images...")
        
        def generate_single(key) -> jnp.ndarray:
            img = jnp.zeros(self.config.shape, dtype=self.config.dtype)
            img = img.at[5:23, 5:8].set(1.0)
            img = img.at[20:23, 5:18].set(1.0)
            return img
        
        keys = jax.random.split(jax.random.PRNGKey(0), n_samples)
        images = jax.vmap(generate_single)(keys)
        
        print(f"Synthetic data generated: {images.shape}")
        return images


class BlurOperator:
    """Blur operator using FFT-based 2D convolution."""
    
    def __init__(self, kernel: jnp.ndarray, image_shape: Tuple[int, int]):
        """
        Initialise blur operator.
        
        Args:
            kernel: 2D convolution kernel.
            image_shape: Shape of images (height, width).
        """
        self.kernel = kernel
        self.image_shape = image_shape
        self._setup_fft()
    
    def _setup_fft(self) -> None:
        """Precompute FFT components for efficient convolution."""
        kernel_padded = jnp.zeros(self.image_shape)
        k_h, k_w = self.kernel.shape
        start_h = (self.image_shape[0] - k_h) // 2
        start_w = (self.image_shape[1] - k_w) // 2
        kernel_padded = kernel_padded.at[start_h:start_h+k_h, start_w:start_w+k_w].set(self.kernel)
        
        self.kernel_padded = jnp.fft.ifftshift(kernel_padded)
        self.kernel_fft = jnp.fft.fft2(self.kernel_padded)
    
    def apply(self, image: jnp.ndarray) -> jnp.ndarray:
        """Apply blur to image."""
        image_fft = jnp.fft.fft2(image)
        result_fft = image_fft * self.kernel_fft
        return jnp.real(jnp.fft.ifft2(result_fft))
    
    def apply_adjoint(self, image: jnp.ndarray) -> jnp.ndarray:
        """Apply adjoint of blur (for gradient computation)."""
        kernel_flipped = jnp.flip(jnp.flip(self.kernel, axis=0), axis=1)
        
        kernel_padded = jnp.zeros(self.image_shape)
        k_h, k_w = kernel_flipped.shape
        start_h = (self.image_shape[0] - k_h) // 2
        start_w = (self.image_shape[1] - k_w) // 2
        kernel_padded = kernel_padded.at[start_h:start_h+k_h, start_w:start_w+k_w].set(kernel_flipped)
        kernel_padded = jnp.fft.ifftshift(kernel_padded)
        kernel_fft_flipped = jnp.fft.fft2(kernel_padded)
        
        image_fft = jnp.fft.fft2(image)
        result_fft = image_fft * kernel_fft_flipped
        return jnp.real(jnp.fft.ifft2(result_fft))
    
    def get_lipschitz_constant(self) -> float:
        """Estimate Lipschitz constant for step size selection."""
        return float(jnp.max(jnp.abs(self.kernel_fft)**2))


class KernelFactory:
    """Factory for creating blur kernels."""
    
    @staticmethod
    def gaussian(config: BlurConfig) -> jnp.ndarray:
        """Create 2D Gaussian kernel."""
        x = jnp.arange(config.kernel_size) - config.kernel_size // 2
        X, Y = jnp.meshgrid(x, x)
        kernel = jnp.exp(-(X**2 + Y**2) / (2 * config.std_dev**2))
        return kernel / jnp.sum(kernel)


class DataPipeline:
    """Orchestrates data loading, blurring, and noise addition."""
    
    def __init__(
        self,
        image_config: ImageConfig,
        blur_config: BlurConfig,
        noise_config: NoiseConfig,
        source: ImageSource
    ):
        """
        Initialise data pipeline.
        
        Args:
            image_config: Image configuration.
            blur_config: Blur kernel configuration.
            noise_config: Noise configuration.
            source: Image source (MNIST, Synthetic).
        """
        self.image_config = image_config
        self.blur_config = blur_config
        self.noise_config = noise_config
        self.source = source
        
        # Initialise blur operator
        kernel = KernelFactory.gaussian(blur_config)
        self.blur_op = BlurOperator(kernel, image_config.shape)
    
    def load_clean_images(self, n_samples: int) -> jnp.ndarray:
        """Load clean images from source."""
        return self.source.load(n_samples)
    
    def apply_degradation(
        self,
        images: jnp.ndarray,
        seed: int = 0
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply blur and noise to images."""
        blurred = jax.vmap(self.blur_op.apply)(images)
        
        # Apply noise
        if self.noise_config.enabled:
            key = jax.random.PRNGKey(seed)
            keys = jax.random.split(key, len(images))
            
            def add_noise(img: jnp.ndarray, k) -> jnp.ndarray:
                noise = self.noise_config.std_dev * jax.random.normal(
                    k, shape=self.image_config.shape
                )
                return img + noise
            
            degraded = jax.vmap(add_noise)(blurred, keys)
        else:
            degraded = blurred
        
        return degraded, blurred
    
    def get_blur_operator(self) -> BlurOperator:
        """Return the configured blur operator."""
        return self.blur_op
