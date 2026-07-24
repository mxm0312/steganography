"""Encoder-сети SteganoGAN. Только torch — как model/ у lfvsn.

Перенесено из DAI-Lab/SteganoGAN (steganogan/encoders.py) без изменений архитектуры.
`upgrade_legacy` дополнительно восстанавливает `_models`, если чекпоинт сохранён до рефактора
(в нём атрибута `_models` нет, есть только сами подсети) — иначе forward упал бы на legacy-весах.
"""

import torch
from torch import nn


class BasicEncoder(nn.Module):
    """(N, 3, H, W) cover + (N, D, H, W) data → (N, 3, H, W) stego."""

    add_image = False

    def _conv2d(self, in_channels: int, out_channels: int) -> nn.Conv2d:
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def _build_models(self):
        self.features = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.layers = nn.Sequential(
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, 3),
            nn.Tanh(),
        )
        return self.features, self.layers

    def __init__(self, data_depth: int, hidden_size: int):
        super().__init__()
        self.version = "1"
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self._models = self._build_models()

    def upgrade_legacy(self) -> None:
        if not hasattr(self, "_models"):
            self._models = (self.features, self.layers)
        if not hasattr(self, "version"):
            self.version = "1"

    def forward(self, image, data):
        x = self._models[0](image)
        x_list = [x]
        for layer in self._models[1:]:
            x = layer(torch.cat(x_list + [data], dim=1))
            x_list.append(x)
        if self.add_image:
            x = image + x
        return x


class ResidualEncoder(BasicEncoder):
    """Как BasicEncoder, но выход прибавляется к cover (residual); без финального Tanh."""

    add_image = True

    def _build_models(self):
        self.features = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.layers = nn.Sequential(
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, 3),
        )
        return self.features, self.layers


class DenseEncoder(BasicEncoder):
    """Плотные связи: каждый блок видит выходы всех предыдущих (DenseNet-стиль)."""

    add_image = True

    def _build_models(self):
        self.conv1 = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv2 = nn.Sequential(
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv3 = nn.Sequential(
            self._conv2d(self.hidden_size * 2 + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv4 = nn.Sequential(
            self._conv2d(self.hidden_size * 3 + self.data_depth, 3),
        )
        return self.conv1, self.conv2, self.conv3, self.conv4

    def upgrade_legacy(self) -> None:
        if not hasattr(self, "_models"):
            self._models = (self.conv1, self.conv2, self.conv3, self.conv4)
        if not hasattr(self, "version"):
            self.version = "1"
