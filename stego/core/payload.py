"""Сериализация полезной нагрузки: заголовок + метаданные + данные секрета.

Формат (big-endian):
    magic     4s   b"STG1"
    version   B
    method_id B
    flags     B
    reserved  B
    name_len  H
    mime_len  H
    data_len  Q
    data_crc  I    crc32(data)
    name      name_len байт (utf-8)
    mime      mime_len байт (utf-8)
    data      data_len байт
"""

import struct
import zlib
from dataclasses import dataclass

from stego.core.exceptions import PayloadError

MAGIC = b"STG1"
VERSION = 1
_HEAD = struct.Struct(">4sBBBBHHQI")  # 24 байта
HEADER_SIZE = _HEAD.size


@dataclass
class Payload:
    method_id: int
    data: bytes
    filename: str = ""
    media_type: str = "application/octet-stream"
    flags: int = 0

    def encode(self) -> bytes:
        name = self.filename.encode("utf-8")
        mime = self.media_type.encode("utf-8")
        head = _HEAD.pack(
            MAGIC,
            VERSION,
            self.method_id,
            self.flags,
            0,
            len(name),
            len(mime),
            len(self.data),
            zlib.crc32(self.data),
        )
        return head + name + mime + self.data

    @classmethod
    def decode(cls, raw: bytes) -> "Payload":
        if len(raw) < _HEAD.size:
            raise PayloadError("данных меньше размера заголовка")
        magic, version, method_id, flags, _, name_len, mime_len, data_len, crc = _HEAD.unpack(
            raw[: _HEAD.size]
        )
        if magic != MAGIC:
            raise PayloadError("magic не совпадает — контейнер без полезной нагрузки")
        if version != VERSION:
            raise PayloadError(f"неподдерживаемая версия формата: {version}")

        off = _HEAD.size
        name = raw[off : off + name_len].decode("utf-8")
        off += name_len
        mime = raw[off : off + mime_len].decode("utf-8")
        off += mime_len
        data = raw[off : off + data_len]
        if len(data) != data_len:
            raise PayloadError("данные обрезаны")
        if zlib.crc32(data) != crc:
            raise PayloadError("контрольная сумма не сошлась")

        return cls(method_id=method_id, data=data, filename=name, media_type=mime, flags=flags)


def header_size(filename: str = "", media_type: str = "application/octet-stream") -> int:
    """Накладные расходы в байтах для секрета с данными метаданными."""
    return _HEAD.size + len(filename.encode("utf-8")) + len(media_type.encode("utf-8"))


def total_size(head: bytes) -> int:
    """Полный размер payload по первым HEADER_SIZE байтам."""
    if len(head) < _HEAD.size:
        raise PayloadError("недостаточно данных для чтения заголовка")
    magic, _, _, _, _, name_len, mime_len, data_len, _ = _HEAD.unpack(head[: _HEAD.size])
    if magic != MAGIC:
        raise PayloadError("magic не совпадает — контейнер без полезной нагрузки")
    return _HEAD.size + name_len + mime_len + data_len
