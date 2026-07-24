"""LSB в видеокадрах. Выход — lossless FFV1/.mkv, иначе младшие биты гибнут при сжатии."""

from pathlib import Path

from stego.core import payload
from stego.core.engine import StegoEngine
from stego.core.exceptions import CapacityExceeded
from stego.core.registry import register
from stego.core.types import CapacityInfo, Secret
from stego.engines.lsb.bitops import embed, extract
from stego.media import video


@register
class LSBVideoEngine(StegoEngine):
    id = 1
    name = "lsb"
    display_name = "LSB"

    def capacity(self, container: str | Path, *, bits_per_channel: int = 1, **_) -> CapacityInfo:
        info = video.probe(container)
        total = (info.samples * bits_per_channel) // 8
        overhead = payload.header_size()
        return CapacityInfo(
            total_bytes=total,
            overhead_bytes=overhead,
            usable_bytes=max(0, total - overhead),
            details={
                "width": info.width,
                "height": info.height,
                "frames": info.frames,
                "fps": info.fps,
                "bits_per_channel": bits_per_channel,
            },
        )

    def pack(self, container, secret: Secret, output, *, bits_per_channel: int = 1, **_) -> None:
        blob = payload.Payload(self.id, secret.data, secret.filename, secret.media_type).encode()

        frames, fps = video.read_frames(container)
        have = frames.size * bits_per_channel // 8
        if len(blob) > have:
            raise CapacityExceeded(need=len(blob), have=have)

        stego = embed(frames.reshape(-1), blob, bits_per_channel).reshape(frames.shape)
        video.write_lossless(output, stego, fps)

    def extract(self, container, *, bits_per_channel: int = 1, **_) -> Secret:
        frames, _ = video.read_frames(container)
        flat = frames.reshape(-1)
        head = extract(flat, payload.HEADER_SIZE, bits_per_channel)
        blob = extract(flat, payload.total_size(head), bits_per_channel)
        p = payload.Payload.decode(blob)
        return Secret(p.data, filename=p.filename, media_type=p.media_type)
