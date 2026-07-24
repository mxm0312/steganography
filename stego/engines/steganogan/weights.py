"""Поиск файла весов SteganoGAN (.steg). Без torch — только работа с путями.

Приоритет источников: явный `weights_path` → переменная окружения `STEGANOGAN_WEIGHTS` →
директория по умолчанию `stego/engines/steganogan/weights/`. Путь/переменная могут указывать на
файл или директорию. В директории файл ищется по каноническому имени `{architecture}.steg`
(напр. `dense.steg`), затем по маске `*{architecture}*.steg` — чтобы можно было положить
скачанный файл как есть.
"""

import os
from pathlib import Path

from stego.core.exceptions import EngineUnavailable

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

ARCHITECTURES = ("dense", "basic", "residual")
DEFAULT_ARCHITECTURE = "dense"

# data_depth официальных предобученных весов (для оценки ёмкости без загрузки torch);
# при упаковке реальный depth берётся из самого чекпоинта. residual — оценочно.
ARCH_DATA_DEPTH = {"dense": 8, "basic": 5, "residual": 1}


def resolve_weights(architecture: str = DEFAULT_ARCHITECTURE, weights_path=None) -> Path:
    name = f"{architecture}.steg"

    files: list[Path] = []
    dirs: list[Path] = []
    for source in (weights_path, os.environ.get("STEGANOGAN_WEIGHTS")):
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
        matches = sorted(d.glob(f"*{architecture}*.steg"))
        if matches:
            return matches[0]

    raise EngineUnavailable(
        f"не найдены веса SteganoGAN для architecture='{architecture}' (ожидается '{name}' "
        f"или файл вида '*{architecture}*.steg'). Скачайте чекпоинт и положите в {WEIGHTS_DIR}, "
        f"либо укажите путь параметром weights_path или переменной STEGANOGAN_WEIGHTS. "
        f"Ссылки — в {WEIGHTS_DIR / 'README.md'}"
    )
