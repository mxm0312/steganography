# Веса LF-VSN

Движок `lfvsn` использует предобученные чекпоинты из репозитория
[MC-E/LF-VSN](https://github.com/MC-E/LF-VSN). Они большие и в git **не** коммитятся —
скачайте нужный вручную и положите сюда с ожидаемым именем.

## Куда класть

Просто положите скачанный файл в **эту директорию** — переименовывать не обязательно.
Поиск идёт так: сперва каноническое имя `lfvsn_{num_video}video.pth`, затем по маске
`*{num_video}video*.pth`. Для одного ролика (`num_video=1`) подходит скачанный как есть
**`LF-VSN_1video_hiding_250k.pth`** (ссылка «One video hiding» ниже) — он найдётся по маске.

Переопределить путь можно:
- параметром `weights_path=` в `pack`/`extract` (файл или директория с чекпоинтами);
- переменной окружения `LFVSN_WEIGHTS` (файл или директория).

## Ссылки (Google Drive)

| num_video | Ожидаемое имя         | Ссылка |
| :-------: | :-------------------- | :----- |
| 1         | `lfvsn_1video.pth`    | https://drive.google.com/file/d/1aEMZaigkMd2NUNXnOu2r0oa5IuLPCtTh/view |
| 2         | `lfvsn_2video.pth`    | https://drive.google.com/file/d/1Yd7tK9Y-J4fkXoL-5u8VifEVsW7OmZN0/view |
| 3         | `lfvsn_3video.pth`    | https://drive.google.com/file/d/1oeDDzkYMZ6tKpPnIUwSI2v_Rbn7vLQJo/view |
| 4         | `lfvsn_4video.pth`    | https://drive.google.com/file/d/1kyMKdfAG_gq6ArWChv6ZMLBsqT-QpS9j/view |
| 5         | `lfvsn_5video.pth`    | https://drive.google.com/file/d/1OlTL6_ZgsThPeYfxbpGrGvNoaisqThq2/view |
| 6         | `lfvsn_6video.pth`    | https://drive.google.com/file/d/1dr-ZIL-VP0ol4fRO7bGZYQoxRetA-GXW/view |
| 7         | `lfvsn_7video.pth`    | https://drive.google.com/file/d/178cqpz_vS-mPlYwLuZP2qFc7pV7vrXrr/view |

Скачанный файл может называться иначе — переименуйте его в `lfvsn_1video.pth` (или укажите
свой путь через `weights_path`/`LFVSN_WEIGHTS`).

> Загрузка весов терпима к префиксу `module.` (чекпоинты сохранены из `DataParallel`) и к
> обёрткам вида `{"state_dict": ...}` — см. `runner._load_net`.
