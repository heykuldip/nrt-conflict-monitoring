import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List
from .base_ae import BaseAE

class BaseVAE(BaseAE):

    def loss_function(self, input: Tensor, results: List[Tensor], kld_weight: float = 1.0, **kwargs) -> Dict:
        """
        Computes the VAE loss function.
        Overrides BaseAE.loss_function to include KL Divergence.
        """
        recons = results[0]  # Reconstruction
        mu = results[1]  # Latent mean
        log_var = results[2]  # Latent log variance

        # 1. Reconstruction loss, summed per tile: a per-pixel mean collapses the latent at beta = 1
        recons_loss = F.mse_loss(recons, torch.nan_to_num(input), reduction='sum') / input.shape[0]

        # 2. Kullback-Leibler divergence, towards a standard Gaussian
        # Formula: -0.5 * sum(1 + log(var) - mu^2 - var)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        # Total loss
        loss = recons_loss + kld_weight * kld_loss

        return {
            'loss': loss,
            'Reconstruction_Loss': recons_loss,
            'KLD': -kld_loss
        }

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    def _visualise_step(self, batch):
        results = self.forward(batch)
        result = results[0]
        rec_error = (batch - result).abs()

        return (
            batch[:, self.visualisation_channels],
            result[:, self.visualisation_channels],
            rec_error.max(1)[0]
        )
