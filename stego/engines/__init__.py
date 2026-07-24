"""Импорт модуля движка регистрирует его в реестре. Новый движок — добавить строку."""

from stego.engines.lfvsn.engine import LFVSNVideoEngine  # noqa: F401
from stego.engines.lsb.engine import LSBVideoEngine  # noqa: F401
from stego.engines.steganogan.engine import SteganoGANImageEngine  # noqa: F401

__all__ = ["LSBVideoEngine", "LFVSNVideoEngine", "SteganoGANImageEngine"]
