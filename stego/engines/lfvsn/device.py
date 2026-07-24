"""Выбор устройства инференса: GPU/CPU. Импорт torch — ленивый (движок остаётся лёгким)."""

from stego.core.exceptions import EngineUnavailable


def require_torch():
    """Импортирует torch или падает с понятным сообщением, как ставить extra."""
    try:
        import torch
    except ImportError as e:
        raise EngineUnavailable(
            "движку 'lfvsn' нужен PyTorch. Установите extra: `uv sync --extra lfvsn` "
            "(или `pip install 'stego[lfvsn]'`)"
        ) from e
    return torch


def resolve_device(device: str = "auto"):
    """'auto' → cuda при наличии, иначе cpu. 'cuda' без GPU → ошибка. 'cpu' → всегда CPU."""
    torch = require_torch()

    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise EngineUnavailable(
            "запрошено устройство 'cuda', но CUDA недоступна. Используйте device='cpu' "
            "или device='auto', либо запустите контейнер с `--gpus all`"
        )
    if device not in ("cpu", "cuda"):
        raise EngineUnavailable(f"неизвестное устройство '{device}', ожидается auto/cpu/cuda")
    return torch.device(device)
