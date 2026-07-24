# stego

Сокрытие данных внутри видеоконтейнеров. Сменные движки за общим фасадом.

Методы:

- **`lsb`** — произвольные байты (текст / файл / изображение) в младших битах пикселей.
  **Bit-exact**: секрет восстанавливается байт-в-байт (magic `STG1` + crc32). Выход —
  **lossless**, кодек по расширению: `.mp4` → H.264 4:4:4 RGB (qp=0), `.mkv` → FFV1.
  Сжатие с потерями убивает младшие биты, поэтому stego-видео крупнее оригинала.
- **`lfvsn`** — [LF-VSN](https://github.com/MC-E/LF-VSN) (CVPR 2023): обратимая нейросеть
  прячет **видео/изображение внутрь видео**. Восстановление **приблизительное** (высокий
  PSNR, не байт-в-байт), поэтому payload/crc не используется. Требует PyTorch (extra `lfvsn`)
  и предобученных весов; работает на **GPU и CPU** (параметр `device`). См. ниже.

## Структура

```
stego/
  core/                    фреймворк
    engine.py              StegoEngine — интерфейс движка (capacity/pack/extract)
    steganographer.py      Steganographer(engine) — фасад
    types.py               Secret, CapacityInfo
    payload.py             формат payload (magic, метаданные, crc32)
    registry.py            реестр движков  ·  exceptions.py  ·  api.py
  media/
    video.py               PyAV: read_frames / write_lossless / probe (переиспользуемо)
  engines/                 сменные движки
    lsb/
      engine.py            LSBVideoEngine
      bitops.py            LSB-упаковка бит
    lfvsn/                 LF-VSN (нейросеть, видео-в-видео)
      engine.py            LFVSNVideoEngine (torch — лениво, движок остаётся лёгким)
      device.py            выбор устройства (auto/cpu/cuda)
      weights.py           поиск файла весов
      runner.py            инференс: hide / reveal
      model/               вендоренная сеть (только torch)
      weights/             сюда кладут скачанные .pth (в git не коммитятся)
  ui/app.py                Streamlit, два режима
  cli.py                   CLI
tests/
```

Слои: `core` — абстракции и фасад, `engines` — реализации, `media` — видео-IO.
Зависимости идут внутрь: `engines` → `core`/`media`, `core` ни от кого не зависит.

## Установка

```bash
make install        # uv sync --all-extras
make test           # прогон тестов
make lint           # ruff check + format
```

## Публичный API

Класс-фасад, инициализируется движком (по имени из реестра или объектом):

```python
from stego import Steganographer
from stego.engines.lsb.engine import LSBVideoEngine

sg = Steganographer("lsb")                 # или Steganographer(LSBVideoEngine())
info = sg.capacity("in.mp4", bits_per_channel=1)     # CapacityInfo
sg.pack("in.mp4", "secret.png", "out.mkv", bits_per_channel=1)
secret = sg.extract("out.mkv", output_dir="./recovered")   # -> Secret
```

Либо функции для быстрых вызовов (`method="lsb"` по умолчанию):

```python
from stego import capacity, pack, extract
pack("in.mp4", "secret.png", "out.mkv")
```

`secret` принимает `Secret`, путь к файлу, текст или `bytes`.

## CLI

```bash
uv run stego capacity in.mp4
uv run stego pack in.mp4 secret.png -o out.mkv
uv run stego extract out.mkv -o ./recovered

# LF-VSN (нейросеть): секрет — видео/изображение, --device auto|cpu|cuda
uv run stego --method lfvsn --device cpu pack cover.mkv secret.mkv -o stego.mkv
uv run stego --method lfvsn --device cuda extract stego.mkv -o ./recovered
```

## UI

```bash
uv run streamlit run stego/ui/app.py
```

## Docker

Образ универсальный: без аргументов — UI, с аргументами — CLI/код.

```bash
docker build -t stego .

# 1) веб-интерфейс на http://localhost:8501
docker run --rm -p 8501:8501 stego

# 2) вызвать методы из CLI (файлы прокидываем через volume)
docker run --rm -v "$PWD:/data" stego stego pack /data/in.mp4 /data/secret.png -o /data/out.mkv
docker run --rm -v "$PWD:/data" stego stego extract /data/out.mkv -o /data/recovered

# 3) произвольный код
docker run --rm -v "$PWD:/data" stego python -c "import stego; print(stego.capacity('/data/in.mp4'))"
```

### LF-VSN и GPU в Docker

Дефолтный образ — CPU (без torch). Для GPU-инференса собирайте CUDA-вариант и пробрасывайте
видеокарту флагом `--gpus`, а веса монтируйте в рантайме:

```bash
docker build -f Dockerfile.cuda -t stego:cuda .

# UI с GPU на http://localhost:8501
docker run --rm --gpus all -p 8501:8501 \
  -v "$PWD/weights:/app/stego/engines/lfvsn/weights" stego:cuda

# CLI-упаковка на GPU (веса + данные через volume)
docker run --rm --gpus all \
  -v "$PWD/weights:/app/stego/engines/lfvsn/weights" -v "$PWD:/data" \
  stego:cuda stego --method lfvsn --device cuda pack /data/cover.mkv /data/secret.mkv -o /data/stego.mkv
```

Без `--gpus` контейнер `stego:cuda` тоже работает — `device=auto` откатится на CPU.
Веса LF-VSN скачиваются вручную — см. [stego/engines/lfvsn/weights/README.md](stego/engines/lfvsn/weights/README.md).

## Добавление движка

1. `engines/<name>/engine.py` — класс-наследник `StegoEngine` с `id/name/display_name`
   и методами `capacity/pack/extract`, декоратор `@register`.
2. Импортировать его в `engines/__init__.py` (одна строка).

Движок сразу доступен через `Steganographer("<name>")`, `api`, CLI и UI.
Видео-IO берётся готовым из `media.video`. Готовые движки: `lsb`, `lfvsn`. На очереди: SteganoGAN.
