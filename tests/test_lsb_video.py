import numpy as np
import pytest

from stego import Secret, capacity, extract, pack

av = pytest.importorskip("av")


@pytest.fixture
def sample_video(tmp_path):
    """Синтетическое lossless-видео 64×48, 10 кадров."""
    path = tmp_path / "cover.mkv"
    rng = np.random.default_rng(1)
    with av.open(str(path), mode="w") as c:
        stream = c.add_stream("ffv1", rate=10)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "bgr0"
        for _ in range(10):
            arr = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
            for pkt in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                c.mux(pkt)
        for pkt in stream.encode():
            c.mux(pkt)
    return path


def test_capacity(sample_video):
    info = capacity(sample_video, bits_per_channel=1)
    assert info.details["frames"] == 10
    assert info.total_bytes == 64 * 48 * 3 * 10 // 8


@pytest.mark.parametrize("ext", [".mkv", ".mp4"])
def test_pack_extract_roundtrip(sample_video, tmp_path, ext):
    secret = Secret.from_text("совершенно секретно", filename="msg.txt")
    output = tmp_path / f"stego{ext}"

    pack(sample_video, secret, output, bits_per_channel=1)
    result = extract(output, bits_per_channel=1)

    assert result.secret.data == secret.data
    assert result.secret.filename == "msg.txt"
