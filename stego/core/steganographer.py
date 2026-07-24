from pathlib import Path

from stego.core import registry
from stego.core.engine import StegoEngine
from stego.core.types import CapacityInfo, Secret


class Steganographer:
    """Фасад: инициализируется движком (объектом или именем) и делегирует ему работу."""

    def __init__(self, engine: StegoEngine | str):
        self.engine = registry.get(engine) if isinstance(engine, str) else engine

    def capacity(self, container: str | Path, **params) -> CapacityInfo:
        return self.engine.capacity(container, **params)

    def pack(
        self,
        container: str | Path,
        secret: Secret | str | bytes | Path,
        output: str | Path,
        **params,
    ) -> None:
        self.engine.pack(container, Secret.coerce(secret), output, **params)

    def extract(
        self,
        container: str | Path,
        output_dir: str | Path | None = None,
        **params,
    ) -> Secret:
        secret = self.engine.extract(container, **params)
        if output_dir is not None:
            secret.save(output_dir)
        return secret

    def __repr__(self) -> str:
        return f"Steganographer(engine={self.engine.name!r})"
