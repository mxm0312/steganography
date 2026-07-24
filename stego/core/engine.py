from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from stego.core.types import CapacityInfo, Secret


class StegoEngine(ABC):
    """Сменный движок стеганографии. Новый метод = наследник + @register."""

    id: ClassVar[int]  # стабильный id, пишется в payload
    name: ClassVar[str]  # ключ в реестре / CLI
    display_name: ClassVar[str]  # человекочитаемое имя для UI
    container_kind: ClassVar[str] = "video"  # "video" | "image" — на чём работает движок (UI/CLI)

    @abstractmethod
    def capacity(self, container: str | Path, **params) -> CapacityInfo:
        """Сколько байт секрета вмещает контейнер при данных параметрах."""

    @abstractmethod
    def pack(self, container: str | Path, secret: Secret, output: str | Path, **params) -> None:
        """Спрятать `secret` в `container`, записать stego-контейнер в `output`."""

    @abstractmethod
    def extract(self, container: str | Path, **params) -> Secret:
        """Достать секрет из stego-контейнера."""
