import tempfile
from pathlib import Path

import streamlit as st

from stego import Secret, capacity, extract, pack
from stego.core import registry
from stego.core.exceptions import CapacityExceeded, PayloadError, StegoError

st.set_page_config(page_title="Стеганография", page_icon="🔏", layout="centered")


def _fmt(n: int) -> str:
    x = float(n)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if x < 1024 or unit == "ГБ":
            return f"{x:.0f} {unit}" if unit == "Б" else f"{x:.1f} {unit}"
        x /= 1024


def _save_upload(upload, workdir: Path) -> Path:
    path = workdir / upload.name
    path.write_bytes(upload.getbuffer())
    return path


# Пояснения к методам (что во что прячет) — чисто UI-слой, ключ = engine.name
METHOD_HELP = {
    "lsb": (
        "**Any2Video** — прячет любой файл (байты) в видео. "
        "Bit-exact: секрет восстанавливается точь-в-точь."
    ),
    "lfvsn": (
        "**Video2Video / Image2Video** — прячет видео или картинку в видео нейросетью. "
        "Восстановление приближённое (не bit-exact)."
    ),
    "steganogan": (
        "**Any2Image** — прячет файл/сообщение в картинку нейросетью (GAN). Не bit-exact: держится "
        "на коррекции ошибок и повторах — надёжнее короткий секрет в крупной картинке."
    ),
}

# Настройки загрузчиков и вывода по типу контейнера движка (engine.container_kind)
_KIND = {
    "video": {
        "pack_label": "Видео-контейнер",
        "extract_label": "Stego-видео",
        "in_types": ["mp4", "mkv", "mov", "avi", "webm"],
        "out_types": ["mp4", "mkv", "mov"],
        "suffix": ".mkv",
        "mime": "video/x-matroska",
        "noun": "видео",
    },
    "image": {
        "pack_label": "Картинка-контейнер",
        "extract_label": "Stego-картинка",
        "in_types": ["png", "bmp", "tif", "tiff", "jpg", "jpeg", "webp"],
        "out_types": ["png", "bmp", "tif", "tiff"],
        "suffix": ".png",
        "mime": "image/png",
        "noun": "картинку",
    },
}

st.title("Стеганография")

engines = {e.name: e for e in registry.engines()}
method = st.selectbox("Метод", list(engines), format_func=lambda n: engines[n].display_name)
if method in METHOD_HELP:
    st.caption(METHOD_HELP[method])
kind = engines[method].container_kind
cfg = _KIND[kind]
mode = st.radio("Режим", ["Упаковать секрет", "Извлечь секрет"], horizontal=True)

# Параметры движка — условно по выбранному методу
params: dict = {}
if method == "lsb":
    params["bits_per_channel"] = st.slider(
        "Бит на канал", 1, 4, 1, help="Больше бит — выше ёмкость, но заметнее искажения"
    )
elif method == "lfvsn":
    params["num_video"] = 1  # устройство определяется автоматически (GPU при наличии, иначе CPU)
elif method == "steganogan":
    params["architecture"] = st.selectbox(
        "Архитектура",
        ["dense", "basic", "residual"],
        help="Веса {architecture}.steg. dense — ёмче, basic — устойчивее на мелких картинках",
    )
    if mode.startswith("Извлечь"):
        st.caption(
            "Архитектура должна совпадать с той, которой паковали, а картинка — быть тем самым "
            "PNG/BMP/TIFF: пересжатие или ресайз стирают секрет."
        )


def _progress_cb(bar, verb: str):
    """Колбэк для st.progress: обновляет бар по кадрам (видео-движки)."""
    return lambda done, total: bar.progress(
        (done / total) if total else 1.0, text=f"{verb}… {done}/{total} кадров"
    )


workdir = Path(tempfile.mkdtemp(prefix="stego_"))

if mode == "Упаковать секрет":
    container_up = st.file_uploader(cfg["pack_label"], type=cfg["in_types"])
    secret_file = st.file_uploader("Секрет (любой файл)")

    if container_up:
        container = _save_upload(container_up, workdir)
        try:
            info = capacity(container, method=method, **params)
            if kind == "image":
                st.info(
                    f"Контейнер: **{info.details['width']}×{info.details['height']}**, "
                    f"архитектура **{info.details['architecture']}** · прячет "
                    f"**≈{_fmt(info.usable_bytes)}** (не bit-exact, лучше короткий секрет)"
                )
                if secret_file:
                    st.caption(
                        f"Секрет: {_fmt(secret_file.size)}. Оценка приблизительная — текст обычно "
                        "сжимается, а чем крупнее картинка, тем надёжнее извлечение."
                    )
            elif method == "lfvsn":
                st.info(
                    f"Контейнер: **{info.details['width']}×{info.details['height']}**, "
                    f"{info.details['frames']} кадров · секрет прячется нейросетью "
                    f"(видео/изображение), восстановление приблизительное"
                )
            else:
                st.info(
                    f"Ёмкость контейнера: **{_fmt(info.usable_bytes)}** "
                    f"· {info.details['width']}×{info.details['height']}, "
                    f"{info.details['frames']} кадров"
                )
                if secret_file:
                    fits = info.fits(secret_file.size)
                    st.metric(
                        "Секрет",
                        _fmt(secret_file.size),
                        delta="помещается" if fits else "не помещается",
                        delta_color="normal" if fits else "inverse",
                    )
        except StegoError as e:
            st.error(f"Не удалось прочитать контейнер: {e}")

    if container_up and secret_file and st.button("Упаковать", type="primary"):
        container = _save_upload(container_up, workdir)
        secret_path = _save_upload(secret_file, workdir)
        output = workdir / f"{Path(container_up.name).stem}_stego{cfg['suffix']}"
        try:
            if kind == "image":
                with st.spinner("Упаковка…"):  # один проход сети — прогресс по кадрам не нужен
                    pack(container, Secret.from_file(secret_path), output, method=method, **params)
                st.success("Готово.")
                st.caption(
                    "Носитель — **PNG (lossless)**. Не пересохраняйте в JPEG: сжатие убьёт секрет."
                )
            else:
                bar = st.progress(0, text="Упаковка…")
                pack(
                    container,
                    Secret.from_file(secret_path),
                    output,
                    method=method,
                    progress=_progress_cb(bar, "Упаковка"),
                    **params,
                )
                bar.empty()
                st.success("Готово.")
                st.caption(
                    "Носитель lossless — открывать в **VLC** (не QuickTime): "
                    "секрет в точных пикселях."
                )
            st.download_button(
                f"Скачать stego-{cfg['noun']}",
                output.read_bytes(),
                file_name=output.name,
                mime=cfg["mime"],
            )
        except CapacityExceeded as e:
            st.error(f"Секрет не помещается: {e}")
        except StegoError as e:
            st.error(f"Ошибка упаковки: {e}")

else:
    container_up = st.file_uploader(cfg["extract_label"], type=cfg["out_types"])
    if container_up and st.button("Извлечь", type="primary"):
        container = _save_upload(container_up, workdir)
        try:
            if kind == "image":
                with st.spinner("Извлечение…"):
                    secret = extract(container, method=method, **params)
            else:
                bar = st.progress(0, text="Извлечение…")
                secret = extract(
                    container, method=method, progress=_progress_cb(bar, "Извлечение"), **params
                )
                bar.empty()
            st.success(f"Извлечено: **{secret.filename}** ({_fmt(secret.size)})")
            st.download_button(
                "Скачать секрет",
                secret.data,
                file_name=secret.filename or "secret.bin",
                mime=secret.media_type,
            )
        except PayloadError as e:
            st.error(f"В контейнере нет корректной нагрузки: {e}")
        except StegoError as e:
            st.error(f"Ошибка извлечения: {e}")
