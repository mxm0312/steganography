"""Движок LF-VSN: нейросетевое сокрытие видео/изображения внутри видео (CVPR 2023).

В отличие от LSB это НЕ bit-exact: секрет восстанавливается обратимой сетью приближённо
(высокий PSNR, но не байт-в-байт), поэтому формат payload STG1/crc32 здесь не применяется —
движок работает напрямую с кадрами. Секрет — видеоролик или изображение, а не произвольные байты.

Модуль намеренно НЕ импортирует torch на уровне файла (и в __init__): регистрация в реестре
не должна тянуть тяжёлую зависимость. torch и вендоренная сеть подгружаются лениво внутри
pack/extract — так же, как `import av` внутри функций media/video.py.
"""

import io
import tempfile
from pathlib import Path

import numpy as np

from stego.core.engine import StegoEngine
from stego.core.exceptions import ContainerError
from stego.core.registry import register
from stego.core.types import CapacityInfo, Secret
from stego.media import video

_VIDEO_SUFFIXES = {".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm"}


@register
class LFVSNVideoEngine(StegoEngine):
    id = 2
    name = "lfvsn"
    display_name = "LF-VSN"

    def capacity(self, container: str | Path, *, num_video: int = 1, **_) -> CapacityInfo:
        info = video.probe(container)
        usable = info.width * info.height * 3 * info.frames * num_video
        return CapacityInfo(
            total_bytes=usable,
            overhead_bytes=0,
            usable_bytes=usable,
            details={
                "width": info.width,
                "height": info.height,
                "frames": info.frames,
                "fps": info.fps,
                "num_video": num_video,
                "note": "нейросеть: восстановление приблизительное (не bit-exact)",
            },
        )

    def pack(
        self,
        container,
        secret: Secret,
        output,
        *,
        num_video: int = 1,
        device: str = "auto",
        weights_path=None,
        progress=None,
        **_,
    ) -> None:
        from stego.engines.lfvsn.device import resolve_device

        dev = resolve_device(device)  # проверит torch/устройство до тяжёлого импорта runner
        from stego.engines.lfvsn import runner

        cover, fps = video.read_frames(container)
        secret_frames = self._secret_to_frames(
            secret, target_hw=(cover.shape[1], cover.shape[2]), n_frames=len(cover)
        )
        stego = runner.hide(
            cover,
            secret_frames,
            num_video=num_video,
            device=dev,
            weights_path=weights_path,
            progress=progress,
        )
        # Носитель обязан быть lossless: секрет живёт в ТОЧНЫХ пикселях stego, а lossy/yuv420p
        # их портит и восстановление рушится. Как у LSB — открывать в VLC, не в QuickTime.
        video.write_lossless(output, stego, fps)

    def extract(
        self,
        container,
        *,
        num_video: int = 1,
        device: str = "auto",
        weights_path=None,
        progress=None,
        **_,
    ) -> Secret:
        from stego.engines.lfvsn.device import resolve_device

        dev = resolve_device(device)  # проверит torch/устройство до тяжёлого импорта runner
        from stego.engines.lfvsn import runner

        stego, fps = video.read_frames(container)
        frames = runner.reveal(
            stego, num_video=num_video, device=dev, weights_path=weights_path, progress=progress
        )
        with tempfile.TemporaryDirectory(prefix="lfvsn_") as tmp:
            out = Path(tmp) / "recovered_secret.mp4"
            video.write_viewable(out, frames, fps)
            data = out.read_bytes()
        return Secret(data, filename="recovered_secret.mp4", media_type="video/mp4")

    # --- вспомогательное: секрет (видео или изображение) -> кадры под геометрию cover ---

    def _secret_to_frames(self, secret: Secret, target_hw, n_frames: int) -> np.ndarray:
        h, w = target_hw
        frames = self._decode_secret(secret)
        frames = self._resize(frames, (w, h))
        return self._fit_length(frames, n_frames)

    @staticmethod
    def _decode_secret(secret: Secret) -> np.ndarray:
        suffix = Path(secret.filename).suffix.lower()
        is_video = secret.media_type.startswith("video/") or suffix in _VIDEO_SUFFIXES
        if is_video:
            with tempfile.TemporaryDirectory(prefix="lfvsn_sec_") as tmp:
                path = Path(tmp) / (secret.filename or "secret.mkv")
                path.write_bytes(secret.data)
                frames, _ = video.read_frames(path)
            return frames
        # иначе трактуем секрет как изображение
        from PIL import Image  # pillow — базовая зависимость

        try:
            img = Image.open(io.BytesIO(secret.data)).convert("RGB")
        except Exception as e:
            raise ContainerError(
                "LF-VSN прячет видео или изображение; секрет не декодируется ни тем ни другим"
            ) from e
        return np.array(img)[None, ...]

    @staticmethod
    def _resize(frames: np.ndarray, size_wh) -> np.ndarray:
        from PIL import Image

        if (frames.shape[2], frames.shape[1]) == size_wh:
            return frames
        return np.stack(
            [np.array(Image.fromarray(f).resize(size_wh, Image.BILINEAR)) for f in frames]
        )

    @staticmethod
    def _fit_length(frames: np.ndarray, n_frames: int) -> np.ndarray:
        if len(frames) == n_frames:
            return frames
        if len(frames) > n_frames:
            return frames[:n_frames]
        reps = -(-n_frames // len(frames))  # ceil
        return np.tile(frames, (reps, 1, 1, 1))[:n_frames]
