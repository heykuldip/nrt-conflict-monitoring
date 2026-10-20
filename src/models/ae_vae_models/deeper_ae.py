import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth=2,
                 activation=nn.LeakyReLU, batchnorm=True):
        super().__init__()

        layers = []
        for _ in range(depth):
            layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            if batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            if activation is not None:
                layers.append(activation())
            in_channels = out_channels

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)

class ResConvBlock(ConvBlock):
    def forward(self, x):
        return x + self.conv_block(x)

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation=nn.LeakyReLU, batchnorm=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation:
            layers.append(activation())
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_method='nearest',
                 activation=nn.LeakyReLU, batchnorm=True):
        super().__init__()

        align_corners = None if upsample_method == "nearest" else True
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=upsample_method,
                        align_corners=align_corners),
            ConvBlock(in_channels, out_channels, depth=1,
                      activation=activation, batchnorm=batchnorm)
        )

    def forward(self, x):
        return self.up(x)
