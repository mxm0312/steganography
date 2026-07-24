"""Кодирование секрета в поток бит и обратно.

Идея — как у SteganoGAN: секрет кодируется Рида — Соломоном и МНОГО раз повторяется по всему
битовому тензору, а на извлечении копии усредняются. Но оригинал разделял копии нулевым
терминатором и декодировал их по одной — это рушится на шуме (терминаторы гибнут) и на бинарных
данных с нулевыми байтами (ложные разрезы). Поэтому здесь надёжнее и без терминаторов:

    frame  = magic "SG" + len(name) + len(mime) + name + mime + data
    body   = reed_solomon( zlib( frame ) )                 # M байт
    каждый 8-й бит потока → 32-битная длина M (повторяется); остальные 7/8 → тело (повторяется)

Заголовок и тело ЧЕРЕДУЮТСЯ по всему потоку (а не двумя блоками): иначе заголовок целиком попал бы
в один data-канал тензора, а точность каналов неравномерна — там он бы не восстановился. При
чередовании обе части равномерно сэмплируют все каналы/пиксели.

Извлечение: берём каждый 8-й бит → сворачиваем по 32 и голосуем побитово → длина M (период
крошечный, копий тысячи → голосование безошибочно) → период тела = M·8 → сворачиваем тело,
голосуем → RS → zlib → фрейм. Период берётся из заголовка, а не из содержимого, поэтому метод не
зависит от данных и устойчив к шуму. НЕ bit-exact: держится на коррекции ошибок и числе повторов.

Два условия, без которых голосование не работает (оба выучены на реальных весах):

1. **Поток перемешивается фиксированной перестановкой.** Логический индекс бита сам по себе жёстко
   привязан к месту в тензоре (i → канал i//(W·H), пиксель i%(W·H)), а точность декодера по местам
   ОЧЕНЬ неровная: строка y=0 — граница свёрток с zero-padding, читается на уровне монетки, а у
   предобученного `dense` канал 0 вообще мёртвый (~0.5). При регулярной раскладке период копии
   резонирует с геометрией: например HEADER_FRACTION=64 и H=256 сажали бит j заголовка ВСЕГДА на
   строку y=64·(j%4), т.е. каждый 4-й бит длины — всегда на мёртвую строку y=0. Такая ошибка
   систематическая, она одинакова во всех копиях, и голосование её не лечит в принципе (сколько
   копий ни возьми). Перестановка разрывает связь «индекс бита → место»: каждая копия попадает в
   свои пиксели и каналы, ошибки становятся независимыми — и большинство снова работает.
2. **Голосуем уверенностями, а не битами.** `recover` принимает сырые логиты декодера (знак — бит,
   модуль — уверенность) и складывает `tanh(l/2)`. Уверенный бит перевешивает сомнительные; это
   примерно вдвое поднимает потолок по размеру секрета против голосования жёсткими битами.

torch-free: numpy + zlib + reedsolo (импорт reedsolo ленивый — не тянется при регистрации).
"""

import struct
import zlib

import numpy as np

from stego.core.exceptions import CapacityExceeded, PayloadError
from stego.core.types import Secret

RS_NSYM = 250  # число байт коррекции на блок — как в оригинале
RS_BLOCK = 255  # длина блока Рида — Соломона (GF(256))
RS_DATA = RS_BLOCK - RS_NSYM  # полезных байт на блок (=5)

# Сколько копий тела должно поместиться, чтобы извлечение было надёжным. Замер на предобученном
# `dense` (точность декодера ~0.85 по битам): 1 копия физически не влезает, 2 — не декодируются,
# с 3 начинает выходить, 4 — с запасом. По этому числу `capacity()` считает честный лимит: без
# запаса на повторы он обещал бы в ~4 раза больше, чем реально извлекается.
RELIABLE_COPIES = 4

# Аванс под служебное внутри одной копии: magic + две длины (6 Б), имя файла и mime во фрейме,
# заголовок zlib и его небольшое расширение на несжимаемых данных. Точную величину `capacity()`
# знать не может (имя и mime приходят вместе с секретом), поэтому закладываем фиксированный запас
# — иначе обещанный лимит не паковался бы: ровно на него ушли бы служебные байты.
FRAME_ALLOWANCE = 64

# заголовку — 1/HEADER_FRACTION позиций потока (сырой повтор 32-битной длины → голосование).
# Крупная F: заголовку хватает сотен копий (его период всего 32 бита), а тело получает почти весь
# поток — важно, т.к. точность декодера низкая и телу нужно максимум повторов для голосования+RS.
HEADER_FRACTION = 64
_HDR_UNIT_BITS = 32  # uint32 длины тела
_MAGIC = b"SG"

# Зерно перестановки потока (см. п.1 в шапке). Фиксировано и одинаково у обеих сторон — это не
# ключ и не секрет, а способ развязать «индекс бита ↔ место в картинке».
_SCATTER_SEED = 0x57454741


def _scatter(n_bits: int) -> np.ndarray:
    """Перестановка «логическая позиция k → физическая позиция perm[k]». Своя обратная — гатером."""
    return np.random.default_rng(_SCATTER_SEED).permutation(n_bits)


def _rs():
    from reedsolo import RSCodec  # ленивый импорт: reedsolo нужен только на упаковке/распаковке

    return RSCodec(RS_NSYM)


def _frame(secret: Secret) -> bytes:
    name = secret.filename.encode("utf-8")[:65535]
    mime = secret.media_type.encode("utf-8")[:65535]
    return _MAGIC + struct.pack(">HH", len(name), len(mime)) + name + mime + secret.data


def _unframe(raw: bytes) -> Secret:
    if len(raw) < 6 or raw[:2] != _MAGIC:
        raise ValueError("не наш фрейм")
    name_len, mime_len = struct.unpack(">HH", raw[2:6])
    off = 6
    name = raw[off : off + name_len].decode("utf-8")
    off += name_len
    mime = raw[off : off + mime_len].decode("utf-8")
    off += mime_len
    return Secret(bytes(raw[off:]), filename=name, media_type=mime)


def _bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()


def _encode_body(secret: Secret) -> np.ndarray:
    return _bytes_to_bits(bytes(_rs().encode(bytearray(zlib.compress(_frame(secret))))))


def _rs_decode(data: bytes) -> bytes | None:
    try:
        out = _rs().decode(data)
        return bytes(out[0] if isinstance(out, tuple) else out)  # reedsolo ≥1.0 → кортеж
    except Exception:
        return None


def message_overhead_bits(secret: Secret) -> int:
    """Бит в одной копии тела секрета (для оценки вместимости и в тестах)."""
    return int(_encode_body(secret).size)


def payload_bits_for(secret: Secret, body_copies: int) -> int:
    """Сколько бит контейнера нужно, чтобы вместить ~body_copies копий тела. Для тестов/оценок."""
    body_region = message_overhead_bits(secret) * body_copies
    # body_region = n_bits · (F-1)/F  →  n_bits = ceil(body_region · F/(F-1))
    return -(-body_region * HEADER_FRACTION // (HEADER_FRACTION - 1)) + _HDR_UNIT_BITS


def _header_mask(n_bits: int) -> np.ndarray:
    """Позиции бит длины: каждый HEADER_FRACTION-й — заголовок, остальные — тело."""
    return (np.arange(n_bits) % HEADER_FRACTION) == 0


def build_payload_bits(
    secret: Secret, n_bits: int, *, min_copies: int = RELIABLE_COPIES
) -> np.ndarray:
    """Поток из `n_bits` бит: чередуем длину (1/F позиций) и повторы тела (0/1 uint8).

    Падает `CapacityExceeded`, если в отведённые под тело (F-1)/F потока не влезает `min_copies`
    копий. Требуем именно НЕСКОЛЬКО копий, а не одну: метод держится на голосовании, и одна-две
    копии не декодируются. Влезь оно «впритык», упаковка бы прошла молча, а извлечение упало бы —
    то есть ошибку пользователь увидел бы уже потеряв оригинал. Лучше отказать сразу и тем же
    порогом, каким `capacity()` считает обещанный лимит.
    """
    body = _encode_body(secret)
    header = _bytes_to_bits(struct.pack(">I", body.size // 8))  # 32 бита

    mask = _header_mask(n_bits)
    n_hdr = int(mask.sum())
    n_body = n_bits - n_hdr
    if n_hdr < header.size or body.size * min_copies > n_body:
        need = -(-body.size * min_copies * HEADER_FRACTION // (HEADER_FRACTION - 1))
        raise CapacityExceeded(need=-(-need // 8), have=n_bits // 8)

    stream = np.empty(n_bits, dtype=np.uint8)
    stream[mask] = np.tile(header, n_hdr // header.size + 1)[:n_hdr]
    stream[~mask] = np.tile(body, n_body // body.size + 1)[:n_body]

    out = np.empty(n_bits, dtype=np.uint8)
    out[_scatter(n_bits)] = stream  # разложить логический поток по всей картинке вперемешку
    return out


def hard_confidence(bits) -> np.ndarray:
    """0/1 → ±1: «идеальный» канал для тестов и симуляций (см. формат входа `recover`)."""
    return np.asarray(bits, dtype=np.float32) * 2.0 - 1.0


def _vote(weights: np.ndarray, period: int) -> np.ndarray:
    """Голосование по выровненным копиям длины `period`: бит — знак суммы уверенностей."""
    n = weights.size // period
    if n <= 0:
        return (weights[:period] > 0).astype(np.uint8)
    return (weights[: n * period].reshape(n, period).sum(axis=0) > 0).astype(np.uint8)


def recover(conf) -> Secret:
    """Восстановить секрет из уверенностей декодера: длина тела → тело → фрейм.

    `conf` — массив логитов (знак = бит, модуль = уверенность), по одному на бит контейнера.
    Жёсткие 0/1 биты сначала прогнать через `hard_confidence`.
    """
    conf = np.asarray(conf, dtype=np.float32)
    weights = np.tanh(conf[_scatter(conf.size)] / 2.0)  # снять перестановку и смягчить логиты
    mask = _header_mask(weights.size)
    hdr_stream, body_stream = weights[mask], weights[~mask]

    if hdr_stream.size < _HDR_UNIT_BITS:
        raise PayloadError("контейнер слишком мал — не прочесть длину секрета")
    (body_bytes,) = struct.unpack(">I", _bits_to_bytes(_vote(hdr_stream, _HDR_UNIT_BITS)))

    body_period = body_bytes * 8
    if body_bytes == 0 or body_period > body_stream.size:
        raise PayloadError(
            "длина секрета не читается: контейнер не похож на упакованный этой архитектурой "
            "(секрета в нём нет, паковали другой архитектурой/версией, либо картинку пересжали "
            "или отресайзили после упаковки)"
        )

    decoded = _rs_decode(_bits_to_bytes(_vote(body_stream, body_period)))
    if decoded is None:
        raise PayloadError(
            "длина секрета прочиталась, но тело не восстанавливается: слишком много ошибок "
            "(нужен более короткий секрет или более крупная картинка)"
        )
    try:
        return _unframe(zlib.decompress(decoded))
    except Exception as e:
        raise PayloadError(f"повреждён фрейм секрета: {e}") from e
