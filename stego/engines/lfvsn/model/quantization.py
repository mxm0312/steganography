"""Дифференцируемое квантование в 8 бит. Вендорено из LF-VSN (models/modules/Quantization.py)."""

import torch
import torch.nn as nn


class Quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input = torch.clamp(input, 0, 1)
        return (input * 255.0).round() / 255.0

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Quantization(nn.Module):
    def forward(self, input):
        return Quant.apply(input)
