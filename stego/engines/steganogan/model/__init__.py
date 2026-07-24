"""Вендоренные сети SteganoGAN (только torch). Тяжёлые импорты — не трогать без torch."""

from stego.engines.steganogan.model.critics import BasicCritic
from stego.engines.steganogan.model.decoders import BasicDecoder, DenseDecoder
from stego.engines.steganogan.model.encoders import (
    BasicEncoder,
    DenseEncoder,
    ResidualEncoder,
)
from stego.engines.steganogan.model.models import SteganoGAN

# architecture -> (Encoder, Decoder); residual использует BasicDecoder (своего нет)
ARCHITECTURES = {
    "basic": (BasicEncoder, BasicDecoder),
    "residual": (ResidualEncoder, BasicDecoder),
    "dense": (DenseEncoder, DenseDecoder),
}

DEFAULT_HIDDEN_SIZE = 32  # как в обучении оригинала


def build(architecture: str = "dense", data_depth: int = 1, hidden_size: int = DEFAULT_HIDDEN_SIZE):
    """Свежая (необученная) пара encoder/decoder — для тестов и как fallback-конструктор."""
    enc_cls, dec_cls = ARCHITECTURES[architecture]
    return enc_cls(data_depth, hidden_size), dec_cls(data_depth, hidden_size)


__all__ = [
    "ARCHITECTURES",
    "DEFAULT_HIDDEN_SIZE",
    "BasicEncoder",
    "ResidualEncoder",
    "DenseEncoder",
    "BasicDecoder",
    "DenseDecoder",
    "BasicCritic",
    "SteganoGAN",
    "build",
]
