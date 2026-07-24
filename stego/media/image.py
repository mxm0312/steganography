"""Картиночный IO поверх Pillow. Общий слой для движков, работающих с изображениями.

Массив — ndarray (H, W, 3) uint8 в RGB. Запись только lossless (PNG/BMP/TIFF): методы, прячущие
данные в точных пикселях, не переживут JPEG/lossy-WebP — те исказят пиксели и сигнал погибнет.
Pillow импортируется лениво (базовая зависимость, но держим слой единообразным с media/video.py).
"""

from pathlib import Path

import numpy as np

from stego.core.exceptions import ContainerError

_LOSSLESS_SUFFIXES = {".png", ".bmp", ".tif", ".tiff"}


def read_rgb(container: str | Path) -> np.ndarray:
    """Прочитать изображение как (H, W, 3) uint8 RGB."""
    from PIL import Image

    try:
        img = Image.open(str(container)).convert("RGB")
    except Exception as e:
        raise ContainerError(f"не удалось прочитать изображение: {e}") from e
    return np.asarray(img, dtype=np.uint8)


def dimensions(container: str | Path) -> tuple[int, int]:
    """(width, height) без декодирования всех пикселей."""
    from PIL import Image

    try:
        with Image.open(str(container)) as img:
            return int(img.width), int(img.height)
    except Exception as e:
        raise ContainerError(f"не удалось прочитать изображение: {e}") from e


def write_lossless(output: str | Path, array: np.ndarray) -> None:
    """Записать (H, W, 3) uint8 RGB без потерь. JPEG/lossy убьют спрятанные данные — запрещаем."""
    from PIL import Image

    suffix = Path(output).suffix.lower()
    if suffix not in _LOSSLESS_SUFFIXES:
        raise ContainerError(
            f"lossless-запись картинки не поддерживает '{suffix}', "
            f"доступны: {sorted(_LOSSLESS_SUFFIXES)} (JPEG уничтожит спрятанные данные)"
        )
    Image.fromarray(np.ascontiguousarray(array), mode="RGB").save(str(output))
