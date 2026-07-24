class StegoError(Exception):
    pass


class MethodNotFound(StegoError):
    pass


class CapacityExceeded(StegoError):
    """Секрет не помещается в контейнер."""

    def __init__(self, need: int, have: int):
        self.need = need
        self.have = have
        super().__init__(f"нужно {need} Б, доступно {have} Б")


class PayloadError(StegoError):
    """Битый заголовок или контейнер без полезной нагрузки."""


class ContainerError(StegoError):
    """Не удалось прочитать/записать видео."""


class EngineUnavailable(StegoError):
    """Движок не готов к работе: нет зависимости, весов или запрошенного устройства."""
