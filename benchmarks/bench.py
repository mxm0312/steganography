"""Benchmark runner.

make bench
make bench METHODS="lsb lfvsn" SCENARIOS="small medium" REPEATS=5
uv run python -m benchmarks.bench --help
"""

import argparse
import datetime
import sys
import tempfile
from pathlib import Path

from stego.core import registry

from .report import save_results
from .runner import SCENARIO_MAP, TrialResult, run_trial

_DEFAULT_ENGINE_PARAMS: dict[str, dict] = {
    "lsb": {"bits_per_channel": 1},
    "lfvsn": {"num_video": 1, "device": "cuda"},
    "steganogan": {"architecture": "dense", "device": "cuda"},
}


def _build_params(method: str, args: argparse.Namespace) -> dict:
    params = dict(_DEFAULT_ENGINE_PARAMS.get(method, {}))
    if method == "lsb":
        params["bits_per_channel"] = args.bits_per_channel
    if method in ("lfvsn", "steganogan"):
        params["device"] = "cuda"
    if method == "lfvsn":
        params["num_video"] = args.num_video
    if method == "steganogan":
        params["architecture"] = args.architecture
    return params


def _check_cuda(methods: list[str]) -> None:
    neural = [m for m in methods if m in ("lfvsn", "steganogan")]
    if not neural:
        return
    try:
        import torch

        if not torch.cuda.is_available():
            print(
                f"Error: CUDA not available but required by {neural}. "
                "Run with --gpus on a CUDA machine.",
                file=sys.stderr,
            )
            raise SystemExit(1)
    except ImportError:
        print("Error: torch not installed (uv sync --extra lfvsn).", file=sys.stderr)
        raise SystemExit(1) from None


def _print_header(methods: list[str], scenarios: list[str], repeats: int) -> None:
    print("=" * 70)
    print("  Steganography Benchmark  [GPU]")
    print("=" * 70)
    print(f"  Methods   : {', '.join(methods)}")
    print(f"  Scenarios : {', '.join(scenarios)}")
    print(f"  Repeats   : {repeats}")
    print("=" * 70)
    print()


def _print_result(r: TrialResult) -> None:
    import math

    tag = f"[{r.method:>10}] {r.scenario:<7}"
    if r.error:
        print(f"  {tag}  SKIP — {r.error[:60]}")
        return
    psnr_s = f"{r.psnr_stego_db:.1f} dB" if not math.isnan(r.psnr_stego_db) else "—"
    quality = (
        "bit-exact ✓"
        if r.bit_exact
        else f"PSNR secret {r.psnr_secret_db:.1f} dB"
        if r.psnr_secret_db
        else "approx"
    )
    gpu_tag = f"  GPU {r.peak_gpu_mb:.0f} MB" if r.peak_gpu_mb else ""
    print(
        f"  {tag}  "
        f"pack {r.pack_fps:>7.1f} fps  "
        f"extract {r.extract_fps:>7.1f} fps  "
        f"PSNR stego {psnr_s:>9}  "
        f"{quality:<22}"
        f"  RAM {r.peak_ram_mb:.0f} MB{gpu_tag}"
        f"  [{r.device}]"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.bench",
        description="GPU benchmark: pack/extract throughput, PSNR, memory → JSON/CSV/Markdown.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        metavar="METHOD",
        help="Methods to benchmark (default: all registered)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["tiny", "small", "medium", "large"],
        default=["tiny", "small"],
        metavar="SCENARIO",
        help="tiny(64×48×10) small(320×240×30) medium(640×480×60) large(1280×720×90)",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Timing repeats per trial")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/results"), metavar="DIR"
    )
    parser.add_argument(
        "--bits-per-channel", type=int, default=1, help="LSB: bits per channel (1–4)"
    )
    parser.add_argument("--num-video", type=int, default=1, help="LF-VSN: number of hidden videos")
    parser.add_argument(
        "--architecture",
        default="dense",
        choices=["dense", "basic", "residual"],
        help="SteganoGAN: model architecture",
    )

    args = parser.parse_args(argv)

    available = {e.name for e in registry.engines()}
    requested = args.methods or sorted(available)
    unknown = set(requested) - available
    if unknown:
        print(
            f"Error: unknown methods {sorted(unknown)}. Available: {sorted(available)}",
            file=sys.stderr,
        )
        return 1

    _check_cuda(requested)

    selected_scenarios = [SCENARIO_MAP[n] for n in args.scenarios]
    _print_header(requested, args.scenarios, args.repeats)

    results: list[TrialResult] = []

    with tempfile.TemporaryDirectory(prefix="stego_bench_") as tmp:
        workdir = Path(tmp)
        total = len(requested) * len(selected_scenarios)
        pairs = [(m, s) for m in requested for s in selected_scenarios]
        for idx, (method, scenario) in enumerate(pairs, start=1):
            print(
                f"  ({idx}/{total}) [{method}] {scenario.name} "
                f"{scenario.width}×{scenario.height}×{scenario.frames}f ...",
                end="  ",
                flush=True,
            )
            try:
                trial = run_trial(
                    method, scenario, workdir, _build_params(method, args), args.repeats
                )
            except Exception as exc:
                print(f"EXCEPTION: {exc}")
                continue
            results.append(trial)
            _print_result(trial)

    if not results:
        print("\nNo results to save.", file=sys.stderr)
        return 1

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / ts
    saved = save_results(results, out_dir, repeats=args.repeats)

    ok = sum(1 for r in results if r.is_ok())
    print()
    print("=" * 70)
    print(f"  Done: {ok}/{len(results)} trials OK")
    print(f"  {saved}/")
    print("    report.md  results.json  results.csv")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
