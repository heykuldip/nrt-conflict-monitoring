from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .base_vae import BaseVAE
from .deeper_ae import ResConvBlock, UpConv

class GaussianBlur(nn.Module):
    def __init__(self, channels, kernel_size=5, sigma=1.0):
        super().__init__()
        ax = torch.arange(kernel_size) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(channels, 1, 1, 1)
        self.register_buffer("weight", kernel)
        self.padding = kernel_size // 2
        self.groups = channels

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=self.padding, groups=self.groups)

class BlurPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.blur = GaussianBlur(channels)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        return self.pool(self.blur(x))

class FrequencyDecomposition(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.blur = GaussianBlur(channels)

    def forward(self, x):
        low = self.blur(x)
        return torch.cat([low, x - low], dim=1)

class MultiScaleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(1, 2, 4)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d)
            for d in dilations
        ])
        self.fuse = nn.Conv2d(len(dilations) * out_ch, out_ch, 1)
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = torch.cat([b(x) for b in self.branches], dim=1)
        return F.leaky_relu(self.norm(self.fuse(x)), 0.2)

class ResolutionAdaptiveVAE(BaseVAE):
    def __init__(self,
                 input_shape: Tuple[int],
                 hidden_channels: List[int],
                 latent_dim: int,
                 extra_depth_on_scale: int,
                 visualisation_channels,
                 **kwargs):
        super().__init__(visualisation_channels)

        self.latent_dim = latent_dim
        in_channels, size = input_shape[0], input_shape[1]
        assert size % (2 ** len(hidden_channels)) == 0
        out_w = size // (2 ** len(hidden_channels))
        self.encoder_output_shape = (hidden_channels[-1], out_w, out_w)

        # Resolution-aware encoder parts
        self.freq = FrequencyDecomposition(in_channels)

        layers = []
        # The encoder input concatenates low and high frequencies, so 2 * in_channels
        ch = in_channels * 2

        for out_ch in hidden_channels:
            layers += [
                MultiScaleConv(ch, out_ch),
                BlurPool(out_ch)
            ]
            if extra_depth_on_scale > 0:
                layers.append(
                    ResConvBlock(out_ch, out_ch, depth=extra_depth_on_scale)
                )
            ch = out_ch

        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc_mu = nn.Linear(hidden_channels[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_channels[-1], latent_dim)
        self.decoder_input = nn.Linear(latent_dim, out_w * out_w * hidden_channels[-1])

        # Decoder input channels are reversed encoder channels + output channel count
        self.decoder = self._build_decoder(
            hidden_channels[::-1] + [in_channels],
            extra_depth_on_scale
        )

    @staticmethod
    def _build_decoder(channels, extra_depth):
        layers = []
        # Reverse iterate through channels to upscale
        for i in range(len(channels) - 1):
            is_output_layer = (i == len(channels) - 2)

            # Linear activation (None) on the output layer, LeakyReLU elsewhere
            activation = None if is_output_layer else nn.LeakyReLU

            # Nearest-neighbor upsampling
            layers.append(
                UpConv(
                    in_channels=channels[i],
                    out_channels=channels[i+1],
                    upsample_method='nearest',
                    activation=activation,
                    batchnorm=not is_output_layer
                )
            )

            # Residual blocks if requested, except on the output layer
            if extra_depth > 0 and not is_output_layer:
                layers.append(
                    ResConvBlock(channels[i+1], channels[i+1], depth=extra_depth)
                )

        return nn.Sequential(*layers)

    def encode(self, input: Tensor):
        # Apply frequency decomposition before encoding
        x = self.encoder(self.freq(torch.nan_to_num(input)))
        x = torch.flatten(self.pool(x), 1)
        return [self.fc_mu(x), self.fc_var(x)]

    def decode(self, z: Tensor):
        x = self.decoder_input(z)
        x = x.view(-1, *self.encoder_output_shape)
        return self.decoder(x)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, input: Tensor, **kwargs):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return [self.decode(z), mu, logvar]
