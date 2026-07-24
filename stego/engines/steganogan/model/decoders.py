"""Decoder-сети SteganoGAN: восстанавливают битовый тензор данных из stego-картинки.

Перенесено из DAI-Lab/SteganoGAN (steganogan/decoders.py) без изменений архитектуры.
`upgrade_legacy` восстанавливает `_models` для чекпоинтов, сохранённых до рефактора.
"""

import torch
from torch import nn


class BasicDecoder(nn.Module):
    """(N, 3, H, W) stego → (N, D, H, W) логиты бит."""

    def _conv2d(self, in_channels: int, out_channels: int) -> nn.Conv2d:
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def _build_models(self):
        self.layers = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, self.data_depth),
        )
        return [self.layers]

    def __init__(self, data_depth: int, hidden_size: int):
        super().__init__()
        self.version = "1"
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self._models = self._build_models()

    def upgrade_legacy(self) -> None:
        if not hasattr(self, "_models"):
            self._models = [self.layers]
        if not hasattr(self, "version"):
            self.version = "1"

    def forward(self, x):
        x = self._models[0](x)
        if len(self._models) > 1:
            x_list = [x]
            for layer in self._models[1:]:
                x = layer(torch.cat(x_list, dim=1))
                x_list.append(x)
        return x


class DenseDecoder(BasicDecoder):
    """Плотные связи (DenseNet-стиль) — пара к DenseEncoder."""

    def _build_models(self):
        self.conv1 = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv2 = nn.Sequential(
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv3 = nn.Sequential(
            self._conv2d(self.hidden_size * 2, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv4 = nn.Sequential(
            self._conv2d(self.hidden_size * 3, self.data_depth),
        )
        return self.conv1, self.conv2, self.conv3, self.conv4

    def upgrade_legacy(self) -> None:
        if not hasattr(self, "_models"):
            self._models = (self.conv1, self.conv2, self.conv3, self.conv4)
        if not hasattr(self, "version"):
            self.version = "1"
