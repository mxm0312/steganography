"""Функциональный фасад — тонкие обёртки над Steganographer для быстрых вызовов."""

from pathlib import Path

from stego.core.steganographer import Steganographer
from stego.core.types import CapacityInfo, ExtractResult, PackResult, Secret

DEFAULT_ENGINE = "lsb"


def capacity(container: str | Path, method: str = DEFAULT_ENGINE, **params) -> CapacityInfo:
    return Steganographer(method).capacity(container, **params)


def pack(
    container: str | Path,
    secret: Secret | str | bytes | Path,
    output: str | Path,
    method: str = DEFAULT_ENGINE,
    **params,
) -> PackResult:
    return Steganographer(method).pack(container, secret, output, **params)


def extract(
    container: str | Path,
    output_dir: str | Path | None = None,
    method: str = DEFAULT_ENGINE,
    **params,
) -> ExtractResult:
    return Steganographer(method).extract(container, output_dir, **params)
