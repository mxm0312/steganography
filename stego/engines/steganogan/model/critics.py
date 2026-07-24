"""Critic-сеть SteganoGAN. В инференсе не участвует — нужна лишь чтобы распаковать
сохранённый .steg (в него запечён critic) и пережить вызов upgrade_legacy.

Перенесено из DAI-Lab/SteganoGAN (steganogan/critics.py) без изменений.
"""

import torch
from torch import nn


class BasicCritic(nn.Module):
    """(N, 3, H, W) → (N,): оценка «cover или stego»."""

    def _conv2d(self, in_channels: int, out_channels: int) -> nn.Conv2d:
        return nn.Conv2d(in_channels, out_channels, kernel_size=3)

    def _build_models(self):
        return nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, 1),
        )

    def __init__(self, hidden_size: int):
        super().__init__()
        self.version = "1"
        self.hidden_size = hidden_size
        self._models = self._build_models()

    def upgrade_legacy(self) -> None:
        if not hasattr(self, "_models"):
            self._models = self.layers
        if not hasattr(self, "version"):
            self.version = "1"

    def forward(self, x):
        x = self._models(x)
        return torch.mean(x.view(x.size(0), -1), dim=1)
