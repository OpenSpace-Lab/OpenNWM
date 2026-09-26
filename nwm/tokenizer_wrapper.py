import logging
from pathlib import Path

import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL
logger = logging.getLogger(__name__)


class TokenizerWrapper(nn.Module):
    """
    Base class for tokenizer wrappers that provides a common interface for different tokenizers.
    """

    def __init__(self, normalize_mean=None, normalize_std=None):
        super().__init__()
        # Store normalization parameters for unnormalization
        if normalize_mean is not None and normalize_std is not None:
            self.register_buffer(
                "normalize_mean", torch.tensor(normalize_mean).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "normalize_std", torch.tensor(normalize_std).view(1, 3, 1, 1)
            )
        else:
            self.normalize_mean = None
            self.normalize_std = None

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method")

    def encode(self, x):
        raise NotImplementedError("Subclasses must implement encode method")

    def decode(self, x, denormalize=False):
        raise NotImplementedError("Subclasses must implement decode method")

    def unnormalize_image(self, normalized_image):
        """
        Unnormalize images from normalized space back to [0, 1] range.
        Args:
            normalized_image: Tensor of shape (B, C, H, W) in normalized space
        Returns:
            Tensor of shape (B, C, H, W) in [0, 1] range
        """
        if self.normalize_mean is None or self.normalize_std is None:
            logger.warning("Normalization parameters not set, returning image as-is")
            return normalized_image

        denormalized = normalized_image * self.normalize_std + self.normalize_mean
        return denormalized.clamp(0, 1)

    @torch.no_grad()
    def get_code_from_latent(self, latent):
        raise NotImplementedError(
            "Subclasses must implement get_code_from_latent method"
        )


class VAEWrapper(TokenizerWrapper):
    def __init__(self, model_path: str, normalize_mean=None, normalize_std=None):
        super().__init__(normalize_mean=normalize_mean, normalize_std=normalize_std)
        # Public SD-VAE uses .bin; existing local caches may contain safetensors.
        local_safetensors = (Path(model_path)/"diffusion_pytorch_model.safetensors").is_file()
        self.vae = AutoencoderKL.from_pretrained(
            model_path, use_safetensors=local_safetensors
        ).eval()
        self.scaling_factor = 0.18215

    def forward(self, x):
        return self.vae.forward(x)

    def encode(self, x):
        return self.vae.encode(x).latent_dist.sample().mul_(self.scaling_factor)

    def decode(self, x, denormalize=False):
        decoded = self.vae.decode(x / self.scaling_factor).sample
        if denormalize:
            decoded = self.unnormalize_image(decoded)
        return decoded

    @torch.no_grad()
    def get_code_from_latent(self, latent):
        return latent

