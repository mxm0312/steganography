import time
from pathlib import Path

from stego.core import registry
from stego.core.engine import StegoEngine
from stego.core.types import CapacityInfo, ExtractResult, PackResult, Secret

_NEURAL_ENGINES = {"lfvsn", "steganogan"}


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
    ) -> PackResult:
        secret = Secret.coerce(secret)
        container_path = Path(container)
        output_path = Path(output)

        container_size = container_path.stat().st_size

        cap_params = {k: v for k, v in params.items() if k != "progress"}
        try:
            cap = self.engine.capacity(container_path, **cap_params)
            capacity_bytes = cap.usable_bytes
            width = cap.details.get("width", 0)
            height = cap.details.get("height", 0)
            frames = cap.details.get("frames", 1)
        except Exception:
            capacity_bytes = width = height = frames = 0

        device = self._detect_device(params)

        t0 = time.perf_counter()
        self.engine.pack(container_path, secret, output_path, **params)
        elapsed = time.perf_counter() - t0

        return PackResult(
            elapsed_s=elapsed,
            secret_bytes=secret.size,
            stego_size_bytes=output_path.stat().st_size,
            container_size_bytes=container_size,
            capacity_bytes=capacity_bytes,
            width=width,
            height=height,
            frames=frames,
            device=device,
        )

    def extract(
        self,
        container: str | Path,
        output_dir: str | Path | None = None,
        **params,
    ) -> ExtractResult:
        container_path = Path(container)
        stego_size = container_path.stat().st_size
        device = self._detect_device(params)

        t0 = time.perf_counter()
        secret = self.engine.extract(container_path, **params)
        elapsed = time.perf_counter() - t0

        if output_dir is not None:
            secret.save(output_dir)

        return ExtractResult(
            secret=secret,
            elapsed_s=elapsed,
            stego_size_bytes=stego_size,
            device=device,
        )

    def _detect_device(self, params: dict) -> str | None:
        if self.engine.name not in _NEURAL_ENGINES:
            return None
        device_param = params.get("device", "auto")
        try:
            import torch

            if device_param in (None, "auto"):
                return "cuda" if torch.cuda.is_available() else "cpu"
            if device_param in ("cpu", "cuda"):
                return device_param
        except ImportError:
            pass
        return "cpu"

    def __repr__(self) -> str:
        return f"Steganographer(engine={self.engine.name!r})"
