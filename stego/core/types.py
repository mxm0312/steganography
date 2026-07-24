import mimetypes
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PackResult:
    """Метрики операции упаковки, возвращаются из Steganographer.pack / api.pack."""

    elapsed_s: float
    secret_bytes: int
    stego_size_bytes: int
    container_size_bytes: int
    capacity_bytes: int
    width: int
    height: int
    frames: int
    device: str | None  # "cpu" / "cuda" или None для LSB

    @property
    def packing_ratio(self) -> float:
        """Доля заполненной ёмкости контейнера (0..1)."""
        return self.secret_bytes / self.capacity_bytes if self.capacity_bytes else 0.0

    @property
    def overhead_ratio(self) -> float:
        """Относительный прирост размера файла: (stego − контейнер) / контейнер."""
        return (
            (self.stego_size_bytes - self.container_size_bytes) / self.container_size_bytes
            if self.container_size_bytes
            else 0.0
        )

    @property
    def bits_per_pixel(self) -> float:
        """Бит секрета на пиксель контейнера."""
        total_pixels = self.width * self.height * max(self.frames, 1)
        return (self.secret_bytes * 8) / total_pixels if total_pixels else 0.0

    @property
    def stego_efficiency(self) -> float:
        """Стеганографическая эффективность: байт секрета / байт stego-файла."""
        return self.secret_bytes / self.stego_size_bytes if self.stego_size_bytes else 0.0

    @property
    def throughput_fps(self) -> float:
        """Скорость обработки, кадров/с."""
        return self.frames / self.elapsed_s if self.elapsed_s else 0.0


@dataclass(frozen=True)
class ExtractResult:
    """Результат извлечения секрета + базовые метрики."""

    secret: "Secret"
    elapsed_s: float
    stego_size_bytes: int
    device: str | None


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
