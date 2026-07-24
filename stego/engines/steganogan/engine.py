"""Движок SteganoGAN: нейросетевое сокрытие произвольных байт внутри ИЗОБРАЖЕНИЯ (не видео).

GAN-энкодер вплавляет данные в пиксели, GAN-декодер их читает; надёжность держится на
Рида — Соломона + повторах + голосовании (см. coding.py). Как и LF-VSN, метод НЕ bit-exact
(нет payload STG1/crc): метод и архитектура задаются параметрами при извлечении.

torch и вендоренная сеть подгружаются лениво внутри pack/extract — регистрация в реестре
остаётся лёгкой (capacity считается по одному Pillow, без torch и без весов).
"""

from pathlib import Path

from stego.core.engine import StegoEngine
from stego.core.registry import register
from stego.core.types import CapacityInfo, Secret
from stego.engines.steganogan.coding import (
    FRAME_ALLOWANCE,
    HEADER_FRACTION,
    RELIABLE_COPIES,
    RS_BLOCK,
    RS_DATA,
)
from stego.engines.steganogan.weights import (
    ARCH_DATA_DEPTH,
    DEFAULT_ARCHITECTURE,
)
from stego.media import image


@register
class SteganoGANImageEngine(StegoEngine):
    id = 3
    name = "steganogan"
    display_name = "SteganoGAN"
    container_kind = "image"

    def capacity(
        self, container: str | Path, *, architecture: str = DEFAULT_ARCHITECTURE, **_
    ) -> CapacityInfo:
        w, h = image.dimensions(container)
        depth = ARCH_DATA_DEPTH.get(architecture, 1)
        raw_bits = w * h * depth
        raw_bytes = raw_bits // 8
        # RS(250): на 255 байт блока — 5 полезных; тело занимает (F-1)/F потока (остаток — длина).
        # Делим на RELIABLE_COPIES: метод держится на повторах, и лимит «в одну копию» был бы
        # обещанием, которое не извлекается. Реальный depth берётся из весов — число ориентир.
        body_bytes = (raw_bits - raw_bits // HEADER_FRACTION) // 8
        usable = ((body_bytes // RELIABLE_COPIES) // RS_BLOCK) * RS_DATA
        usable = max(0, usable - FRAME_ALLOWANCE)  # служебное внутри копии (фрейм + zlib)
        return CapacityInfo(
            total_bytes=raw_bytes,
            overhead_bytes=raw_bytes - usable,
            usable_bytes=usable,
            details={
                "width": w,
                "height": h,
                "architecture": architecture,
                "data_depth": depth,
                "reliable_copies": RELIABLE_COPIES,
                "note": (
                    f"нейросеть: не bit-exact; лимит уже с запасом на повторы "
                    f"(секрет вшивается ~{RELIABLE_COPIES}x). Нужен обычный снимок или "
                    f"рисунок: в чистом шуме сеть читать не умеет"
                ),
            },
        )

    def pack(
        self,
        container,
        secret: Secret,
        output,
        *,
        architecture: str = DEFAULT_ARCHITECTURE,
        device: str = "auto",
        weights_path=None,
        progress=None,
        **_,
    ) -> None:
        from stego.engines.steganogan.device import resolve_device

        dev = resolve_device(device)  # проверит torch/устройство до тяжёлого импорта runner
        from stego.engines.steganogan import runner

        cover = image.read_rgb(container)
        stego = runner.hide(
            cover,
            secret,
            architecture=architecture,
            device=dev,
            weights_path=weights_path,
            progress=progress,
        )
        # выход обязан быть lossless: данные живут в точных пикселях, JPEG их убьёт
        image.write_lossless(output, stego)

    def extract(
        self,
        container,
        *,
        architecture: str = DEFAULT_ARCHITECTURE,
        device: str = "auto",
        weights_path=None,
        progress=None,
        **_,
    ) -> Secret:
        from stego.engines.steganogan.device import resolve_device

        dev = resolve_device(device)  # проверит torch/устройство до тяжёлого импорта runner
        from stego.engines.steganogan import runner

        stego = image.read_rgb(container)
        return runner.reveal(
            stego,
            architecture=architecture,
            device=dev,
            weights_path=weights_path,
            progress=progress,
        )
