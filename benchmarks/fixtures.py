from pathlib import Path

import numpy as np


def _cosine_frame(width: int, height: int, phase: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width] / max(height, width)
    base = np.stack(
        [
            0.5 + 0.3 * np.sin(8 * xx + phase) * np.cos(6 * yy),
            0.5 + 0.3 * np.cos(7 * yy + 0.5 + phase * 0.7),
            0.5 + 0.25 * np.sin(5 * xx + 3 * yy + phase * 0.3),
        ],
        axis=-1,
    )
    noise = rng.normal(0, 0.025, (height, width, 3))
    return np.clip((base + noise) * 255, 0, 255).astype(np.uint8)


def make_video(
    path: Path, width: int, height: int, frames: int, fps: float = 25.0, seed: int = 0
) -> Path:
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("ffv1", rate=int(fps))
        stream.width, stream.height = width, height
        stream.pix_fmt = "bgr0"
        for i in range(frames):
            phase = 2 * np.pi * i / max(frames, 1)
            arr = _cosine_frame(width, height, phase=phase, seed=seed + i)
            for pkt in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
    return path


def make_image(path: Path, width: int, height: int, seed: int = 0) -> Path:
    from PIL import Image

    arr = _cosine_frame(width, height, phase=0.0, seed=seed)
    Image.fromarray(arr).save(str(path))
    return path


def make_secret_bytes(size_bytes: int, seed: int = 1) -> bytes:
    rng = np.random.default_rng(seed)
    return bytes(rng.integers(0, 256, size_bytes, dtype=np.uint8).tolist())
