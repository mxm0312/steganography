from pathlib import Path

import numpy as np


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB between two uint8 arrays. Returns inf if identical."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / mse))


def read_video_frames(path: Path) -> np.ndarray:
    """Returns (N, H, W, 3) uint8 RGB."""
    import av

    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames) if frames else np.empty((0, 1, 1, 3), dtype=np.uint8)


def read_image_rgb(path: Path) -> np.ndarray:
    """Returns (H, W, 3) uint8 RGB."""
    from PIL import Image

    return np.array(Image.open(str(path)).convert("RGB"))
