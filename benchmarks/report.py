"""Сохранение результатов бенчмарка в JSON, CSV и Markdown."""

import csv
import dataclasses
import datetime
import json
import math
import platform
from pathlib import Path
from typing import Any

from .runner import TrialResult

# ---------------------------------------------------------------------------
# Системная информация
# ---------------------------------------------------------------------------


def _system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }
    try:
        import psutil  # опционально

        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["gpu_vram_gb"] = round(props.total_memory / 1024**3, 1)
            info["gpu_count"] = torch.cuda.device_count()
            info["cuda_version"] = torch.version.cuda
        else:
            info["gpu"] = "none (CUDA unavailable)"
    except ImportError:
        info["gpu"] = "torch not installed"
    return info


# ---------------------------------------------------------------------------
# Форматирование
# ---------------------------------------------------------------------------


def _v(value: Any, fmt: str = ".2f", na: str = "N/A") -> str:
    if value is None:
        return na
    if isinstance(value, float) and math.isnan(value):
        return "ERR"
    if isinstance(value, bool):
        return "✓" if value else "✗"
    if isinstance(value, float) and math.isinf(value):
        return "∞"
    try:
        return format(value, fmt)
    except Exception:
        return str(value)


def _psnr_cell(db: float | None) -> str:
    if db is None:
        return "—"
    if math.isinf(db):
        return "∞ dB"
    if math.isnan(db):
        return "ERR"
    return f"{db:.2f} dB"


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

_MD_HEADER = """\
# Steganography Benchmark Report

{sys_block}

> **Reproducibility note:** synthetic cosine+noise fixtures, fixed seeds,
> timings = mean over {repeats} repeat(s) per trial.

---

## Results

| Method | Scenario | Resolution | Frames | Pack fps | Extract fps | Pack MB/s \
| PSNR stego ↑ | PSNR secret ↑ | Bit-exact | Fill | BPP | Overhead | RAM MB | GPU MB | Device |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|:---|
"""

_MD_ROW = (
    "| {method} | {scenario} | {res} | {frames} | {pack_fps} | {extract_fps} | {pack_mbs}"
    " | {psnr_stego} | {psnr_secret} | {bit_exact}"
    " | {fill} | {bpp} | {overhead} | {ram} | {gpu} | {device} |"
)

_MD_GLOSSARY = """
---

## Metric definitions

| Metric | Formula / meaning |
|:---|:---|
| **PSNR stego** | `10·log₁₀(255²/MSE(cover,stego))` — invisibility. >40 dB imperceptible. |
| **PSNR secret** | `10·log₁₀(255²/MSE(orig,recovered))` — recovery fidelity (neural only). |
| **Bit-exact** | Recovered payload is bit-for-bit identical to the original (LSB only). |
| **Fill** | `secret_bytes / usable_capacity` — fraction of container capacity used. |
| **BPP** | `secret_bits / (W × H × frames)` — bits of secret per pixel. |
| **Overhead** | `(stego_size − cover_size) / cover_size` — relative file size increase. |
| **Pack MB/s** | Stego output size (MB) per second of encoding time. |
"""


def _render_markdown(results: list[TrialResult], sys_info: dict, repeats: int) -> str:
    sys_lines = [
        f"**Date:** {sys_info.get('timestamp', '')}  ",
        f"**OS:** {sys_info.get('os', 'N/A')}  ",
        f"**CPU:** {sys_info.get('cpu', 'N/A')}  ",
        f"**GPU:** {sys_info.get('gpu', 'N/A')}"
        + (f" ({sys_info['gpu_vram_gb']} GB VRAM)" if "gpu_vram_gb" in sys_info else "")
        + "  ",
    ]
    if "ram_gb" in sys_info:
        sys_lines.append(f"**RAM:** {sys_info['ram_gb']} GB  ")

    rows: list[str] = []
    for r in results:
        if r.error:
            row = _MD_ROW.format(
                method=r.method,
                scenario=r.scenario,
                res=f"{r.width}×{r.height}",
                frames=r.frames,
                pack_fps="—",
                extract_fps="—",
                pack_mbs="—",
                psnr_stego="—",
                psnr_secret="—",
                bit_exact="—",
                fill="—",
                bpp="—",
                overhead="—",
                ram="—",
                gpu="—",
                device=f"ERROR: {r.error[:40]}",
            )
        else:
            row = _MD_ROW.format(
                method=r.method,
                scenario=r.scenario,
                res=f"{r.width}×{r.height}",
                frames=r.frames,
                pack_fps=_v(r.pack_fps),
                extract_fps=_v(r.extract_fps),
                pack_mbs=_v(r.pack_mbs),
                psnr_stego=_psnr_cell(r.psnr_stego_db),
                psnr_secret=_psnr_cell(r.psnr_secret_db),
                bit_exact=("✓" if r.bit_exact else ("✗" if r.bit_exact is False else "—")),
                fill=_v(r.packing_ratio, ".1%"),
                bpp=_v(r.bits_per_pixel, ".4f"),
                overhead=_v(r.stego_overhead_ratio, "+.1%"),
                ram=_v(r.peak_ram_mb, ".1f"),
                gpu=_v(r.peak_gpu_mb, ".1f") if r.peak_gpu_mb is not None else "—",
                device=r.device,
            )
        rows.append(row)

    body = _MD_HEADER.format(sys_block="\n".join(sys_lines), repeats=repeats)
    body += "\n".join(rows)
    body += _MD_GLOSSARY
    return body


# ---------------------------------------------------------------------------
# Основная точка входа
# ---------------------------------------------------------------------------


def save_results(
    results: list[TrialResult],
    output_dir: Path,
    repeats: int = 3,
) -> Path:
    """Сохраняет results.json, results.csv, report.md в output_dir. Возвращает output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sys_info = _system_info()

    # --- JSON ---
    payload = {
        "system": sys_info,
        "repeats": repeats,
        "results": [dataclasses.asdict(r) for r in results],
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    # --- CSV ---
    if results:
        fields = [f.name for f in dataclasses.fields(results[0])]
        with open(output_dir / "results.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for r in results:
                writer.writerow(dataclasses.asdict(r))

    # --- Markdown ---
    (output_dir / "report.md").write_text(
        _render_markdown(results, sys_info, repeats), encoding="utf-8"
    )

    return output_dir
