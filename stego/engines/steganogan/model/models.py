"""Заглушка верхнеуровневого объекта для распаковки .steg.

Предобученные веса SteganoGAN сохранены как `torch.save(self)` — pickled-объект класса
`steganogan.models.SteganoGAN` (обычный объект, не nn.Module). Чтобы распаковать его без
оригинального пакета (несовместимого по зависимостям), подсовываем эту заглушку под тем же
именем модуля/класса (см. runner._install_import_shims). Нам из объекта нужны только
`encoder`, `decoder` и `data_depth`.
"""


class SteganoGAN:
    """Держатель атрибутов распакованного .steg. __init__ при unpickle не вызывается."""

    encoder = None
    decoder = None
    critic = None
    data_depth = None

    def __repr__(self) -> str:
        return f"SteganoGAN(data_depth={self.data_depth!r})"
