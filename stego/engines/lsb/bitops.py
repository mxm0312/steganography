"""LSB-упаковка бит в плоский массив uint8-сэмплов."""

import numpy as np

from stego.core.exceptions import CapacityExceeded


def capacity_bits(n_samples: int, bits_per_sample: int) -> int:
    return n_samples * bits_per_sample


def embed(cover: np.ndarray, data: bytes, bits_per_sample: int = 1) -> np.ndarray:
    """Пишет `data` в младшие `bits_per_sample` бит каждого сэмпла. Возвращает копию."""
    if not 1 <= bits_per_sample <= 8:
        raise ValueError("bits_per_sample должен быть в диапазоне 1..8")

    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    b = bits_per_sample
    pad = (-len(bits)) % b
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

    weights = (1 << np.arange(b - 1, -1, -1)).astype(np.uint8)
    values = (bits.reshape(-1, b) * weights).sum(axis=1).astype(np.uint8)

    if values.size > cover.size:
        raise CapacityExceeded(need=values.size, have=cover.size)

    mask = np.uint8((1 << b) - 1)
    out = cover.copy()
    out[: values.size] = (out[: values.size] & ~mask) | values
    return out


def extract(stego: np.ndarray, n_bytes: int, bits_per_sample: int = 1) -> bytes:
    """Читает `n_bytes` из младших `bits_per_sample` бит сэмплов."""
    b = bits_per_sample
    n_bits = n_bytes * 8
    n_samples = -(-n_bits // b)  # ceil

    mask = np.uint8((1 << b) - 1)
    values = stego[:n_samples] & mask
    shifts = np.arange(b - 1, -1, -1)
    bits = ((values[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)[:n_bits]
    return np.packbits(bits).tobytes()
