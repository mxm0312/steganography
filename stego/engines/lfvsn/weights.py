"""Поиск файла весов LF-VSN. Без torch — только работа с путями.

Приоритет источников: явный `weights_path` → переменная окружения `LFVSN_WEIGHTS` → директория
по умолчанию `stego/engines/lfvsn/weights/`. Путь/переменная могут указывать на файл или директорию.
В директории файл ищется сперва по каноническому имени `lfvsn_{n}video.pth`, затем по маске
`*{n}video*.pth` — чтобы можно было просто положить скачанный файл как есть
(напр. `LF-VSN_1video_hiding_250k.pth`), не переименовывая.
"""

import os
from pathlib import Path

from stego.core.exceptions import EngineUnavailable

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def weights_filename(num_video: int) -> str:
    return f"lfvsn_{num_video}video.pth"


def resolve_weights(num_video: int = 1, weights_path: str | Path | None = None) -> Path:
    name = weights_filename(num_video)

    files: list[Path] = []  # источники, указывающие прямо на файл
    dirs: list[Path] = []  # директории для поиска
    for source in (weights_path, os.environ.get("LFVSN_WEIGHTS")):
        if source:
            p = Path(source)
            (dirs if p.is_dir() else files).append(p)
    dirs.append(WEIGHTS_DIR)

    for f in files:  # 1) явно указанный файл
        if f.is_file():
            return f
    for d in dirs:  # 2) каноническое имя в директории
        if (d / name).is_file():
            return d / name
    for d in dirs:  # 3) по маске — имя как при скачивании
        matches = sorted(d.glob(f"*{num_video}video*.pth"))
        if matches:
            return matches[0]

    raise EngineUnavailable(
        f"не найдены веса LF-VSN для num_video={num_video} (ожидается '{name}' "
        f"или файл вида '*{num_video}video*.pth'). Скачайте чекпоинт и положите в {WEIGHTS_DIR}, "
        f"либо укажите путь параметром weights_path или переменной LFVSN_WEIGHTS. "
        f"Ссылки — в {WEIGHTS_DIR / 'README.md'}"
    )
