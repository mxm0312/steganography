import argparse
import sys

from stego import capacity, extract, pack
from stego.core.exceptions import StegoError


def _fmt_bytes(n: int) -> str:
    x = float(n)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if x < 1024 or unit == "ГБ":
            return f"{x:.0f} {unit}" if unit == "Б" else f"{x:.1f} {unit}"
        x /= 1024


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stego", description="Стеганография в видео и картинках")
    parser.add_argument("--method", default="lsb")
    parser.add_argument("--bits-per-channel", type=int, default=1, help="LSB: бит на канал 1..4")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="нейросети (LF-VSN / SteganoGAN): устройство (auto — GPU при наличии, иначе CPU)",
    )
    parser.add_argument("--num-video", type=int, default=1, help="LF-VSN: число секретных роликов")
    parser.add_argument(
        "--architecture",
        choices=["dense", "basic", "residual"],
        default="dense",
        help="SteganoGAN: архитектура сети (веса {architecture}.steg)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capacity", help="показать ёмкость контейнера")
    c.add_argument("container")

    p = sub.add_parser("pack", help="упаковать секрет в видео")
    p.add_argument("container")
    p.add_argument("secret", help="путь к файлу-секрету")
    p.add_argument("-o", "--output", required=True)

    e = sub.add_parser("extract", help="извлечь секрет из видео")
    e.add_argument("container")
    e.add_argument("-o", "--output-dir", required=True)

    args = parser.parse_args(argv)
    # каждый движок берёт из params свои параметры, лишнее игнорирует через **_
    params = {
        "bits_per_channel": args.bits_per_channel,
        "device": args.device,
        "num_video": args.num_video,
        "architecture": args.architecture,
    }

    try:
        if args.cmd == "capacity":
            info = capacity(args.container, method=args.method, **params)
            print(
                f"Ёмкость: {_fmt_bytes(info.usable_bytes)} "
                f"(сырьё {_fmt_bytes(info.total_bytes)}, заголовок {info.overhead_bytes} Б)"
            )
            print(f"Параметры: {info.details}")
        elif args.cmd == "pack":
            pack(args.container, args.secret, args.output, method=args.method, **params)
            print(f"Готово: {args.output}")
        elif args.cmd == "extract":
            secret = extract(args.container, args.output_dir, method=args.method, **params)
            print(f"Извлечено: {secret.filename} ({_fmt_bytes(secret.size)})")
    except StegoError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
