"""Кодирование SteganoGAN (coding.py) — чистые тесты без torch и без весов.

Симулируем «канал» (то, что вернул бы decoder) и проверяем RS + фрейминг + повторы + голосование.
Именно здесь живёт логика надёжности, поэтому тесты детерминированные. `recover` принимает логиты,
поэтому жёсткие биты канала прогоняем через `coding.hard_confidence` (0/1 → ±1).
"""

import numpy as np
import pytest

from stego.core.exceptions import CapacityExceeded, PayloadError
from stego.core.types import Secret

pytest.importorskip("reedsolo")  # coding нужен reedsolo (extra steganogan)

from stego.engines.steganogan import coding  # noqa: E402


def _payload(secret: Secret, copies: int) -> np.ndarray:
    # min_copies=1: здесь мы сами задаём число копий и проверяем сам код, а не политику запаса
    return coding.build_payload_bits(secret, coding.payload_bits_for(secret, copies), min_copies=1)


def _channel(bits: np.ndarray) -> np.ndarray:
    """Идеальный канал: биты как есть, с максимальной уверенностью."""
    return coding.hard_confidence(bits)


def test_perfect_channel_roundtrip():
    secret = Secret.from_text("hello, стего", filename="msg.txt")
    got = coding.recover(_channel(_payload(secret, copies=3)))
    assert got.data == secret.data
    assert got.filename == "msg.txt"
    assert got.media_type == "text/plain"


def test_single_copy_exact():
    secret = Secret(b"exactly-one-copy")
    assert coding.recover(_channel(_payload(secret, copies=1))).data == secret.data


def test_majority_vote_survives_noise():
    secret = Secret.from_text("recover me despite noise", filename="n.txt")
    bits = _payload(secret, copies=60)
    rng = np.random.default_rng(1234)
    noisy = bits.copy()
    noisy[rng.random(bits.size) < 0.2] ^= 1  # 20% битовых ошибок на копию
    got = coding.recover(_channel(noisy))
    assert got.data == secret.data and got.filename == "n.txt"


def test_binary_secret_with_zero_runs():
    data = (
        bytes([0, 0, 0, 0, 7, 0, 0, 255])
        + np.random.default_rng(0).integers(0, 256, 120, dtype=np.uint8).tobytes()
    )
    secret = Secret(data, filename="blob.bin", media_type="application/octet-stream")
    got = coding.recover(_channel(_payload(secret, copies=4)))
    assert got.data == data and got.filename == "blob.bin"


def test_survives_systematically_dead_places_in_container():
    """Мёртвые МЕСТА контейнера (а не случайный шум) не должны ронять извлечение.

    У реального декодера точность резко неровная по местам: строка y=0 — граница свёрток с
    zero-padding, а у предобученного `dense` канал 0 читается как монетка. Геометрия здесь
    подобрана резонансной (H=64 при HEADER_FRACTION=64): при регулярной раскладке ВСЕ биты
    длины легли бы ровно в строку y=0, во всех копиях сразу — голосование такое не лечит.
    Проверяем, что перемешивание потока разрывает эту связь.
    """
    depth, w, h = 8, 256, 64
    secret = Secret.from_text("живуч", filename="d.txt")
    conf = coding.hard_confidence(coding.build_payload_bits(secret, depth * w * h))

    idx = np.arange(conf.size)  # индекс бита → канал idx//(w·h), внутри канала — (x, y) по h
    dead = (idx % h == 0) | (idx // (w * h) == 0)  # строка y=0 + весь канал 0
    conf[dead] = np.random.default_rng(5).normal(0.0, 1.0, int(dead.sum()))  # шум вместо сигнала

    got = coding.recover(conf)
    assert got.data == secret.data and got.filename == "d.txt"


def test_pack_refuses_secret_that_fits_but_would_not_decode():
    """Секрет, влезающий «впритык» (1 копия), должен отвергаться на упаковке, а не на извлечении.

    Иначе `pack` молча делает картинку, из которой ничего не достать, — а узнаёт об этом
    пользователь, когда оригинала уже нет.
    """
    secret = Secret(b"x" * 400, filename="x.bin", media_type="application/octet-stream")
    tight = coding.payload_bits_for(secret, 1)  # ровно на одну копию

    coding.build_payload_bits(secret, tight, min_copies=1)  # впритык — технически влезает
    with pytest.raises(CapacityExceeded):
        coding.build_payload_bits(secret, tight)  # но по умолчанию требуем запас на повторы

    roomy = coding.payload_bits_for(secret, coding.RELIABLE_COPIES)
    assert coding.build_payload_bits(secret, roomy).size == roomy  # с запасом — проходит


def test_capacity_exceeded():
    with pytest.raises(CapacityExceeded):
        coding.build_payload_bits(Secret(b"x" * 4000), n_bits=64)


def test_garbage_bits_raise_payload_error():
    noise = np.random.default_rng(7).integers(0, 2, 200_000, dtype=np.uint8)
    with pytest.raises(PayloadError):
        coding.recover(_channel(noise))
