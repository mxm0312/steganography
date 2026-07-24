"""Инференс SteganoGAN над одной картинкой: hide (спрятать) / reveal (восстановить).

Препроцессинг как в оригинале (DAI-Lab/SteganoGAN, models.py) — `permute(2, 1, 0)` и `/127.5 - 1`
в [-1,1], **но симметрично на обе стороны**. В оригинале `decode()` нормирует stego как `/255` в
[0,1], хотя при обучении декодер видел выход энкодера в [-1,1] (`_encode_decode`); это баг
оригинала, а не особенность весов. Замер на предобученном `dense`: `/255` даёт 0.66 точности по
битам (шум), `/127.5 - 1` — 0.85. Возвращаем декодеру ту же шкалу, на которой он обучался.

`reveal` отдаёт в coding.py сырые логиты, а не пороговые биты: модуль логита — уверенность, и
голосование по копиям взвешивается ею (см. coding.py). Байтовое кодирование секрета — там же.

Загрузка .steg: официальные веса сохранены как pickled-объект `steganogan.*` под старый torch и
с запечённым состоянием оптимизатора. Грузим без оригинального пакета — подсовываем vendored-классы
через sys.modules и выбрасываем классы `torch.optim.*` кастомным Unpickler (в инференсе не нужны).

Импортируется лениво (из pack/extract движка) — здесь можно держать torch на уровне модуля.
"""

import pickle
import sys
import types
import warnings

import numpy as np
import torch

from stego.core.types import Secret
from stego.engines.steganogan import coding
from stego.engines.steganogan.weights import resolve_weights


class _DropTrainingCruft:
    """Заглушка под `torch.optim.*` из .steg: для инференса не нужны, состояние глотаем."""

    def __setstate__(self, state) -> None:
        pass


def _install_import_shims() -> None:
    """Подсунуть vendored-классы под именами модулей оригинала, чтобы pickle распаковался."""
    from stego.engines.steganogan import model as vendored

    if "steganogan" not in sys.modules:
        pkg = types.ModuleType("steganogan")
        pkg.__path__ = []  # пометить как пакет
        sys.modules["steganogan"] = pkg
    for suffix, mod in (
        ("models", vendored.models),
        ("encoders", vendored.encoders),
        ("decoders", vendored.decoders),
        ("critics", vendored.critics),
    ):
        sys.modules.setdefault(f"steganogan.{suffix}", mod)


def _inference_pickle_module():
    """pickle-модуль для torch.load, отбрасывающий классы оптимизаторов из чекпоинта."""

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("torch.optim"):
                return _DropTrainingCruft
            return super().find_class(module, name)

    shim = types.ModuleType("stego_steganogan_pickle")  # torch читает pickle_module.__name__
    shim.Unpickler = _Unpickler
    for attr in ("load", "loads", "Pickler", "dump", "dumps", "UnpicklingError"):
        setattr(shim, attr, getattr(pickle, attr))
    shim.HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL
    return shim


def _load(architecture: str, device, weights_path):
    """Достать encoder, decoder и data_depth из .steg."""
    path = resolve_weights(architecture, weights_path)
    _install_import_shims()
    with warnings.catch_warnings():
        # SourceChangeWarning здесь ожидаем и не важен: классы намеренно вендоренные, их исходник
        # не совпадает с запечённым в .steg. Веса (state_dict) читаются как есть.
        warnings.simplefilter("ignore", torch.serialization.SourceChangeWarning)
        obj = torch.load(
            str(path),
            map_location=device,
            pickle_module=_inference_pickle_module(),
            weights_only=False,
        )
    encoder, decoder = obj.encoder, obj.decoder
    for part in (encoder, decoder, getattr(obj, "critic", None)):
        if part is not None and hasattr(part, "upgrade_legacy"):
            part.upgrade_legacy()  # достроить _models, если чекпоинт из legacy-формата
    encoder.to(device).eval()
    decoder.to(device).eval()
    depth = getattr(obj, "data_depth", None) or getattr(decoder, "data_depth", 1)
    return encoder, decoder, int(depth)


def _report(progress, done: int, total: int) -> None:
    if progress is not None:
        progress(done, total)


def hide(
    cover: np.ndarray,
    secret: Secret,
    *,
    architecture: str = "dense",
    device="cpu",
    weights_path=None,
    progress=None,
) -> np.ndarray:
    """Спрятать `secret` в картинку `cover` (H, W, 3) uint8. Возвращает stego (H, W, 3) uint8."""
    _report(progress, 0, 1)
    encoder, _, depth = _load(architecture, device, weights_path)

    t = torch.from_numpy(cover.astype(np.float32) / 127.5 - 1.0)
    t = t.permute(2, 1, 0).unsqueeze(0).to(device)  # (1, 3, W, H) — как в оригинале
    _, _, w, h = t.shape
    bits = coding.build_payload_bits(secret, depth * w * h)  # бросит CapacityExceeded, если не влез
    payload = torch.from_numpy(bits.astype(np.float32)).view(1, depth, w, h).to(device)

    with torch.no_grad():
        gen = encoder(t, payload)[0].clamp(-1.0, 1.0)
    out = (gen.permute(2, 1, 0).detach().cpu().numpy() + 1.0) * 127.5
    _report(progress, 1, 1)
    return out.astype(np.uint8)


def reveal(
    stego: np.ndarray,
    *,
    architecture: str = "dense",
    device="cpu",
    weights_path=None,
    progress=None,
) -> Secret:
    """Восстановить секрет из stego-картинки (H, W, 3) uint8."""
    _report(progress, 0, 1)
    _, decoder, _ = _load(architecture, device, weights_path)

    t = torch.from_numpy(stego.astype(np.float32) / 127.5 - 1.0)  # та же шкала, что у энкодера
    t = t.permute(2, 1, 0).unsqueeze(0).to(device)
    with torch.no_grad():
        conf = decoder(t).view(-1).float().cpu().numpy()  # логиты: знак — бит, модуль — уверенность
    secret = coding.recover(conf)
    _report(progress, 1, 1)
    return secret
