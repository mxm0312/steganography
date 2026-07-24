import mimetypes
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Secret:
    """Скрываемые данные + метаданные для восстановления файла."""

    data: bytes
    filename: str = ""
    media_type: str = "application/octet-stream"

    @property
    def size(self) -> int:
        return len(self.data)

    @classmethod
    def from_file(cls, path: str | Path) -> "Secret":
        path = Path(path)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return cls(path.read_bytes(), filename=path.name, media_type=mime)

    @classmethod
    def from_text(cls, text: str, filename: str = "message.txt") -> "Secret":
        return cls(text.encode("utf-8"), filename=filename, media_type="text/plain")

    @classmethod
    def coerce(cls, value: "Secret | str | bytes | Path") -> "Secret":
        if isinstance(value, cls):
            return value
        if isinstance(value, bytes):
            return cls(value)
        path = Path(value)
        if path.exists():
            return cls.from_file(path)
        if isinstance(value, str):
            return cls.from_text(value)
        raise ValueError("secret: ожидается Secret, путь к файлу, текст или bytes")

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        out = directory / (self.filename or "secret.bin")
        out.write_bytes(self.data)
        return out


@dataclass(frozen=True)
class CapacityInfo:
    total_bytes: int  # сырая ёмкость контейнера
    overhead_bytes: int  # заголовок payload
    usable_bytes: int  # доступно под данные секрета
    details: dict = field(default_factory=dict)

    def fits(self, secret_size: int) -> bool:
        return secret_size <= self.usable_bytes
