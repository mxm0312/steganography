import numpy as np
import pytest

from stego.core.exceptions import CapacityExceeded
from stego.engines.lsb.bitops import embed, extract


@pytest.mark.parametrize("bps", [1, 2, 3, 4, 8])
def test_roundtrip(bps):
    rng = np.random.default_rng(0)
    cover = rng.integers(0, 256, size=100_000, dtype=np.uint8)
    data = rng.bytes(500)

    stego = embed(cover, data, bps)
    assert extract(stego, len(data), bps) == data


def test_only_low_bits_change():
    cover = np.full(1000, 0b1010_1010, dtype=np.uint8)
    stego = embed(cover, b"\xff\x00\xff", bits_per_sample=1)
    assert np.all((cover ^ stego) <= 1)  # тронут только младший бит


def test_capacity_exceeded():
    cover = np.zeros(10, dtype=np.uint8)
    with pytest.raises(CapacityExceeded):
        embed(cover, b"too much data here", bits_per_sample=1)
