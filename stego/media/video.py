"""Видео-IO поверх PyAV. Общий слой для движков, работающих с кадрами.

Кадры — ndarray (N, H, W, 3) uint8 в RGB. Запись только lossless — для LSB-подобных
методов сжатие с потерями уничтожает младшие биты. Кодек выбирается по расширению:
  .mp4/.mov → libx264rgb (H.264 4:4:4 RGB, qp=0); .mkv → FFV1.
Оба сохраняют RGB побитово точно.
"""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from stego.core.exceptions import ContainerError

CHANNELS = 3  # rgb24

# suffix -> (codec, pix_fmt, options); все варианты lossless и bit-exact по RGB
_ENCODERS = {
    ".mp4": ("libx264rgb", "rgb24", {"qp": "0"}),
    ".m4v": ("libx264rgb", "rgb24", {"qp": "0"}),
    ".mov": ("libx264rgb", "rgb24", {"qp": "0"}),
    ".mkv": ("ffv1", "bgr0", {}),  # упакованный RGB(+пад); rgb24<->bgr0 без потерь
}


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frames: int
    fps: float

    @property
    def samples(self) -> int:
        return self.width * self.height * CHANNELS * self.frames


def probe(container: str | Path) -> VideoInfo:
    import av

    with av.open(str(container)) as c:
        v = c.streams.video[0]
        w, h = v.codec_context.width, v.codec_context.height
        fps = v.average_rate or Fraction(25, 1)
        frames = v.frames
        if not frames:  # matroska и др. не хранят счётчик — считаем пакеты (1 пакет ≈ 1 кадр)
            frames = sum(1 for pkt in c.demux(v) if pkt.size)
    if not frames:
        raise ContainerError("не удалось определить число кадров")
    return VideoInfo(width=w, height=h, frames=frames, fps=float(fps))


def read_frames(container: str | Path) -> tuple[np.ndarray, Fraction]:
    import av

    frames = []
    with av.open(str(container)) as c:
        v = c.streams.video[0]
        fps = v.average_rate or Fraction(25, 1)
        for frame in c.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ContainerError("нет видеокадров")
    return np.stack(frames), fps


def write_viewable(output: str | Path, frames: np.ndarray, fps: Fraction, crf: str = "18") -> None:
    """Обычный H.264 (yuv420p) — открывается в QuickTime/Preview/браузере.

    Для человеко-смотрибельного результата (восстановленный секрет LF-VSN). Не для носителя и не
    для LSB: yuv420p субдискретизирует цвет и убивает младшие биты, из такого не извлечь секрет.
    """
    import av

    _, h, w, _ = frames.shape
    with av.open(str(output), mode="w") as c:
        stream = c.add_stream("libx264", rate=fps)
        stream.width, stream.height = w, h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": crf}
        for f in frames:
            for pkt in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(pkt)
        for pkt in stream.encode():
            c.mux(pkt)


def write_lossless(output: str | Path, frames: np.ndarray, fps: Fraction) -> None:
    import av

    suffix = Path(output).suffix.lower()
    if suffix not in _ENCODERS:
        raise ContainerError(
            f"lossless-запись не поддерживает '{suffix}', доступны: {sorted(_ENCODERS)}"
        )
    codec, pix_fmt, options = _ENCODERS[suffix]

    _, h, w, _ = frames.shape
    with av.open(str(output), mode="w") as c:
        stream = c.add_stream(codec, rate=fps)
        stream.width, stream.height = w, h
        stream.pix_fmt = pix_fmt
        stream.options = options
        for f in frames:
            for pkt in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(pkt)
        for pkt in stream.encode():
            c.mux(pkt)
