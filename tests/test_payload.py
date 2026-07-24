import pytest

from stego.core.exceptions import PayloadError
from stego.core.payload import HEADER_SIZE, Payload, header_size, total_size


def test_roundtrip():
    p = Payload(method_id=1, data=b"\x00\x01\x02hello", filename="a.png", media_type="image/png")
    decoded = Payload.decode(p.encode())
    assert decoded == p


def test_total_size_matches():
    p = Payload(method_id=1, data=b"x" * 42, filename="f.txt", media_type="text/plain")
    blob = p.encode()
    assert total_size(blob[:HEADER_SIZE]) == len(blob)


def test_header_size():
    p = Payload(method_id=1, data=b"", filename="f", media_type="m")
    assert header_size("f", "m") == len(p.encode())


def test_bad_magic():
    with pytest.raises(PayloadError):
        Payload.decode(b"XXXX" + b"\x00" * 40)


def test_crc_mismatch():
    blob = bytearray(Payload(method_id=1, data=b"abcd").encode())
    blob[-1] ^= 0xFF
    with pytest.raises(PayloadError):
        Payload.decode(bytes(blob))
