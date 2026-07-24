"""Сборка сети VSN из конфигурации. Аналог `define_G_v2` из LF-VSN (models/networks.py).

`down_num` — число групп обратимых блоков; при `block_num=[8, 8]` и `down_num=2` получается
16 блоков — ровно столько в опубликованных чекпоинтах (проверено по state_dict `*_1video_*`).
"""

from stego.engines.lfvsn.model.inv_arch import VSN
from stego.engines.lfvsn.model.subnet import subnet


def build_vsn(
    num_video: int = 1,
    gop: int = 3,
    in_nc: int = 12,
    out_nc: int = 12,
    block_num=(8, 8),
    block_num_rbm: int = 8,
    down_num: int = 2,
    model: str = "MIMO-VRN-h",
) -> VSN:
    opt = {
        "model": model,
        "num_video": num_video,
        "gop": gop,
        "network_G": {
            "in_nc": in_nc,
            "out_nc": out_nc,
            "block_num": list(block_num),
            "block_num_rbm": block_num_rbm,
        },
    }
    if num_video == 1:
        return VSN(opt, subnet("DBNet", "xavier"), subnet("DBNet", "xavier"), down_num)
    return VSN(opt, subnet("DBNet", "xavier"), subnet("DBNet", "xavier_v2"), down_num)
