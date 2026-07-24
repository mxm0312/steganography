import importlib

from stego.core.engine import StegoEngine
from stego.core.exceptions import MethodNotFound

_BY_NAME: dict[str, StegoEngine] = {}
_BY_ID: dict[int, StegoEngine] = {}
_loaded = False


def register(cls: type[StegoEngine]) -> type[StegoEngine]:
    if cls.name in _BY_NAME:
        raise ValueError(f"движок '{cls.name}' уже зарегистрирован")
    if cls.id in _BY_ID:
        raise ValueError(f"id {cls.id} занят движком '{_BY_ID[cls.id].name}'")
    inst = cls()
    _BY_NAME[cls.name] = inst
    _BY_ID[cls.id] = inst
    return cls


def _ensure_loaded() -> None:
    global _loaded
    if not _loaded:
        _loaded = True
        importlib.import_module("stego.engines")  # импорт регистрирует движки


def get(name: str) -> StegoEngine:
    _ensure_loaded()
    try:
        return _BY_NAME[name]
    except KeyError:
        raise MethodNotFound(f"неизвестный движок '{name}', доступны: {available()}") from None


def get_by_id(engine_id: int) -> StegoEngine:
    _ensure_loaded()
    try:
        return _BY_ID[engine_id]
    except KeyError:
        raise MethodNotFound(f"неизвестный id движка: {engine_id}") from None


def available() -> list[str]:
    _ensure_loaded()
    return sorted(_BY_NAME)


def engines() -> list[StegoEngine]:
    _ensure_loaded()
    return [_BY_NAME[name] for name in sorted(_BY_NAME)]
