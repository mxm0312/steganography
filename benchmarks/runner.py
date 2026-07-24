import dataclasses
import math
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

from stego import Secret, extract, pack
from stego.core import registry

from .fixtures import make_image, make_secret_bytes, make_video
from .metrics import psnr, read_image_rgb, read_video_frames


@dataclasses.dataclass(frozen=True)
class ScenarioConfig:
    name: str
    width: int
    height: int
    frames: int
    fps: float = 25.0

    @property
    def pixels(self) -> int:
        return self.width * self.height


ALL_SCENARIOS: list[ScenarioConfig] = [
    ScenarioConfig("tiny", 64, 48, 10),
    ScenarioConfig("small", 320, 240, 30),
    ScenarioConfig("medium", 640, 480, 60),
    ScenarioConfig("large", 1280, 720, 90),
]

SCENARIO_MAP: dict[str, ScenarioConfig] = {s.name: s for s in ALL_SCENARIOS}


@dataclasses.dataclass
class TrialResult:
    method: str
    scenario: str
    container_kind: str
    width: int
    height: int
    frames: int
    fps: float
    device: str

    secret_bytes: int

    pack_s: float
    extract_s: float
    pack_fps: float
    extract_fps: float
    pack_mbs: float

    container_size_bytes: int
    stego_size_bytes: int

    packing_ratio: float
    bits_per_pixel: float
    stego_overhead_ratio: float
    stego_efficiency: float

    psnr_stego_db: float
    psnr_secret_db: float | None
    bit_exact: bool | None

    peak_ram_mb: float
    peak_gpu_mb: float | None

    engine_params: dict[str, Any]
    error: str | None = None

    def is_ok(self) -> bool:
        return self.error is None and not math.isnan(self.pack_s)


def _reset_gpu_stats() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _peak_gpu_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
    except ImportError:
        pass
    return None


def run_trial(
    method: str,
    scenario: ScenarioConfig,
    workdir: Path,
    engine_params: dict[str, Any],
    n_repeats: int = 3,
) -> TrialResult:
    eng = registry.get(method)
    kind = eng.container_kind

    if kind == "video":
        cover_path = workdir / f"cover_{scenario.name}_{method}.mkv"
        make_video(cover_path, scenario.width, scenario.height, scenario.frames, scenario.fps)
        frames_count = scenario.frames
    else:
        cover_path = workdir / f"cover_{scenario.name}_{method}.png"
        make_image(cover_path, scenario.width, scenario.height)
        frames_count = 1

    container_size = cover_path.stat().st_size

    cap_params = {k: v for k, v in engine_params.items() if k != "progress"}
    try:
        cap = eng.capacity(cover_path, **cap_params)
    except Exception as e:
        return _error_result(method, scenario, kind, engine_params, f"capacity(): {e}")

    if method == "lfvsn":
        w, h = max(32, scenario.width // 4), max(24, scenario.height // 4)
        secret_path = workdir / f"secret_{scenario.name}_lfvsn.png"
        make_image(secret_path, w, h, seed=99)
    else:
        target_bytes = max(16, int(cap.usable_bytes * 0.5))
        secret_path = workdir / f"secret_{scenario.name}_{method}.bin"
        secret_path.write_bytes(make_secret_bytes(target_bytes, seed=42))

    secret = Secret.from_file(secret_path)
    secret_size_bytes = secret.size

    suffix = ".mkv" if kind == "video" else ".png"
    stego_path = workdir / f"stego_{scenario.name}_{method}{suffix}"

    pack_times: list[float] = []
    peak_ram_mb = 0.0

    for _ in range(n_repeats):
        _reset_gpu_stats()
        tracemalloc.start()
        t0 = time.perf_counter()
        try:
            pack_result = pack(cover_path, secret, stego_path, method=method, **engine_params)
        except Exception as e:
            tracemalloc.stop()
            return _error_result(method, scenario, kind, engine_params, f"pack(): {e}")
        pack_times.append(time.perf_counter() - t0)
        _, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_ram_mb = max(peak_ram_mb, traced_peak / 1024**2)

    peak_gpu = _peak_gpu_mb()
    pack_s = float(np.mean(pack_times))
    stego_size = stego_path.stat().st_size
    device_str = pack_result.device or "cpu"

    extract_times: list[float] = []
    last_extract_result = None

    for _ in range(n_repeats):
        t0 = time.perf_counter()
        try:
            last_extract_result = extract(stego_path, method=method, **engine_params)
        except Exception as e:
            return _error_result(method, scenario, kind, engine_params, f"extract(): {e}")
        extract_times.append(time.perf_counter() - t0)

    extract_s = float(np.mean(extract_times))
    recovered = last_extract_result.secret

    psnr_stego = _psnr_stego(cover_path, stego_path, kind)
    psnr_secret_val, bit_exact_val = _secret_quality(
        method, secret, secret_path, recovered, workdir
    )

    total_pixels = scenario.width * scenario.height * frames_count
    packing_ratio = secret_size_bytes / cap.usable_bytes if cap.usable_bytes else 0.0
    bpp = (secret_size_bytes * 8) / total_pixels if total_pixels else 0.0
    overhead = (stego_size - container_size) / container_size if container_size else 0.0
    efficiency = secret_size_bytes / stego_size if stego_size else 0.0

    return TrialResult(
        method=method,
        scenario=scenario.name,
        container_kind=kind,
        width=scenario.width,
        height=scenario.height,
        frames=frames_count,
        fps=scenario.fps,
        device=device_str,
        secret_bytes=secret_size_bytes,
        pack_s=pack_s,
        extract_s=extract_s,
        pack_fps=frames_count / pack_s if pack_s else 0.0,
        extract_fps=frames_count / extract_s if extract_s else 0.0,
        pack_mbs=(stego_size / 1024**2) / pack_s if pack_s else 0.0,
        container_size_bytes=container_size,
        stego_size_bytes=stego_size,
        packing_ratio=packing_ratio,
        bits_per_pixel=bpp,
        stego_overhead_ratio=overhead,
        stego_efficiency=efficiency,
        psnr_stego_db=psnr_stego,
        psnr_secret_db=psnr_secret_val,
        bit_exact=bit_exact_val,
        peak_ram_mb=peak_ram_mb,
        peak_gpu_mb=peak_gpu,
        engine_params=engine_params,
    )


def _psnr_stego(cover_path: Path, stego_path: Path, kind: str) -> float:
    try:
        if kind == "video":
            cover = read_video_frames(cover_path)
            stego = read_video_frames(stego_path)
            n = min(len(cover), len(stego))
            return psnr(cover[:n], stego[:n]) if n else float("nan")
        return psnr(read_image_rgb(cover_path), read_image_rgb(stego_path))
    except Exception:
        return float("nan")


def _secret_quality(
    method: str,
    original_secret: Secret,
    secret_path: Path,
    recovered: Secret,
    workdir: Path,
) -> tuple[float | None, bool | None]:
    if method == "lsb":
        return None, recovered.data == original_secret.data

    if method == "lfvsn":
        try:
            orig = read_image_rgb(secret_path)
            rec_path = workdir / "_rec_secret.mp4"
            rec_path.write_bytes(recovered.data)
            rec_frames = read_video_frames(rec_path)
            if len(rec_frames) == 0:
                return None, None
            rec_frame = rec_frames[0]
            if orig.shape[:2] != rec_frame.shape[:2]:
                from PIL import Image

                h, w = rec_frame.shape[:2]
                orig = np.array(Image.fromarray(orig).resize((w, h), Image.BILINEAR))
            return psnr(orig, rec_frame), None
        except Exception:
            return None, None

    return None, None


def _error_result(
    method: str, scenario: ScenarioConfig, kind: str, params: dict, error_msg: str
) -> TrialResult:
    nan = float("nan")
    return TrialResult(
        method=method,
        scenario=scenario.name,
        container_kind=kind,
        width=scenario.width,
        height=scenario.height,
        frames=scenario.frames,
        fps=scenario.fps,
        device="N/A",
        secret_bytes=0,
        pack_s=nan,
        extract_s=nan,
        pack_fps=nan,
        extract_fps=nan,
        pack_mbs=nan,
        container_size_bytes=0,
        stego_size_bytes=0,
        packing_ratio=nan,
        bits_per_pixel=nan,
        stego_overhead_ratio=nan,
        stego_efficiency=nan,
        psnr_stego_db=nan,
        psnr_secret_db=None,
        bit_exact=None,
        peak_ram_mb=nan,
        peak_gpu_mb=None,
        engine_params=params,
        error=error_msg,
    )
