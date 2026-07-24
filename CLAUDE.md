# CLAUDE.md

Стеганография: сокрытие данных внутри видеоконтейнеров. Движки: `lsb` (байты, bit-exact),
`lfvsn` (нейросеть, видео-в-видео, приблизительно). Сменные движки, публичный фасад
`Steganographer`, Streamlit UI + CLI, Docker.

## Команды

```bash
make install   # uv sync --all-extras (ядро + ui + lfvsn/torch + dev). Только ядро+ui: uv sync --extra ui
make test      # uv run pytest
make lint      # ruff check + ruff format --check
make format    # ruff check --fix + ruff format
make ui        # streamlit
```

Зависимости — **только через uv** (`uv add ...`), не через pip. Python ≥ 3.11.

## Архитектура

Три слоя, зависимости строго внутрь: `engines` → `core`/`media`, `core` ни от кого.

```
stego/
  core/        фреймворк
    engine.py          StegoEngine (ABC): capacity/pack/extract
    steganographer.py  Steganographer(engine) — фасад, делегирует движку
    types.py           Secret, CapacityInfo
    payload.py         формат payload (magic STG1, метаданные, crc32)
    registry.py        реестр движков (@register, lazy-import engines)
    api.py             функции capacity/pack/extract поверх фасада
    exceptions.py
  media/video.py       PyAV: probe / read_frames / write_lossless (переиспользуемо)
  engines/lsb/         движок LSB: engine.py + bitops.py
  engines/lfvsn/       движок LF-VSN: engine.py + device/weights/runner + model/ (вендоренная сеть)
  ui/app.py  cli.py
```

- Движок самоописателен: `engine.id` пишется в payload → по контейнеру видно, чем паковали.
  (LF-VSN не bit-exact и payload не пишет — метод задаётся при извлечении параметрами.)
- `Steganographer("lsb")` (по имени) или `Steganographer(LSBVideoEngine())` (объектом).

## Добавление движка

1. `engines/<name>/engine.py`: наследник `StegoEngine` с `id/name/display_name` и
   `capacity/pack/extract`, декоратор `@register`. Кадры брать из `media.video`.
2. Импортировать в `engines/__init__.py` (одна строка) — регистрирует его.

Движок сразу доступен в `Steganographer`, CLI и UI (дропдаун строится из `registry.engines()`).

## Важные детали

- **Выход только lossless.** Кодек выбирается по расширению в `media/video.py` (`_ENCODERS`):
  `.mp4/.mov` → `libx264rgb` (H.264 4:4:4 RGB, `qp=0`), `.mkv` → `ffv1` (pix_fmt `bgr0`,
  т.к. FFV1 в этой сборке не ест `rgb24`). Оба bit-exact по RGB. Обычный lossy-H.264 убил бы
  младшие биты. Stego-видео крупнее оригинала — плата за метод.
- **Ёмкость** считается по числу пакетов (`probe`), т.к. matroska не хранит счётчик кадров.
- Параметры движка передаются как `**params` (у LSB — `bits_per_channel` 1..4;
  у LF-VSN — `device` auto/cpu/cuda, `num_video`, `weights_path`).
  UI показывает контролы параметров условно по выбранному движку.

## LF-VSN (нейросетевой движок)

- **torch — опциональный extra `lfvsn`**, импортируется **лениво** внутри `pack/extract`
  (как `import av` в `media/video.py`). `engine.py` НЕ импортирует torch на уровне модуля/`__init__`,
  иначе регистрация тянула бы torch даже для LSB. `capacity` — без torch (только `video.probe`).
- Сеть вендорена в `engines/lfvsn/model/` (только torch, без `basicsr`). При переносе из репозитория:
  убраны неиспользуемые импорты `basicsr`/`cv2`; `.cuda()` → device-aware (иначе нет CPU-инференса).
- **Веса** — крупные `.pth` c Google Drive, качаются вручную в `engines/lfvsn/weights/`
  (в git не коммитятся, см. там README). Путь переопределяется `weights_path`/`LFVSN_WEIGHTS`.
- **Не bit-exact**: секрет (видео/изображение) восстанавливается приближённо. Секрет-видео/картинка
  масштабируется под геометрию cover; обработка группами по `gop=3` кадра.
- **GPU/CPU**: `device.resolve_device("auto")` берёт GPU при наличии, иначе CPU; `"cuda"` без карты
  → `EngineUnavailable`. В Docker — `Dockerfile.cuda` + `docker run --gpus all`.

## Зависимости и воспроизводимость

- `uv.lock` **коммитим**. Docker ставит с `uv sync --frozen` — строго версии из lock,
  без пере-резолва. После правок в `pyproject.toml` обновить lock: `uv lock`.

## Тесты

- `pytest`. `test_lsb_video.py` требует `av` (через `pytest.importorskip`) и гоняет полный
  round-trip на синтетическом видео. `test_bitops`/`test_payload` — чистые, без видео.
