# Веса SteganoGAN

Движок `steganogan` использует предобученные чекпоинты из репозитория
[DAI-Lab/SteganoGAN](https://github.com/DAI-Lab/SteganoGAN) (файлы `*.steg`). Они небольшие
(~1 МБ), но в git **не** коммитятся — скачайте нужный и положите сюда.

## Куда класть

Положите файл в **эту директорию** под каноническим именем `{architecture}.steg`. Поиск идёт так:
сперва `{architecture}.steg`, затем по маске `*{architecture}*.steg`. По умолчанию используется
`dense` (параметр `--architecture` / `architecture=`).

Переопределить путь можно:
- параметром `weights_path=` в `pack`/`extract` (файл или директория с чекпоинтами);
- переменной окружения `STEGANOGAN_WEIGHTS` (файл или директория).

## Скачать

Файлы лежат прямо в исходниках оригинального пакета. Например:

```bash
cd stego/engines/steganogan/weights
curl -sSL -O https://raw.githubusercontent.com/DAI-Lab/SteganoGAN/master/steganogan/pretrained/dense.steg
curl -sSL -O https://raw.githubusercontent.com/DAI-Lab/SteganoGAN/master/steganogan/pretrained/basic.steg
```

| architecture | Файл          | Encoder / Decoder            | data_depth |
| :----------- | :------------ | :--------------------------- | :--------: |
| `dense`      | `dense.steg`  | DenseEncoder / DenseDecoder  |     8      |
| `basic`      | `basic.steg`  | BasicEncoder / DenseDecoder  |     5      |
| `residual`   | `residual.steg` | ResidualEncoder / BasicDecoder |   —      |

## Замечания

- Чекпоинты сохранены как pickled-объект под старый torch и с запечённым состоянием оптимизатора.
  Загрузчик (`runner._load`) распаковывает их без оригинального пакета: подменяет модули
  `steganogan.*` вендоренными классами и отбрасывает `torch.optim.*`.
- Метод **не bit-exact**: секрет держится на коррекции Рида — Соломона и повторах. Надёжнее всего
  прячется короткое сообщение в крупной картинке (больше повторов → увереннее голосование).
