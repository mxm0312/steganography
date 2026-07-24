import pytest

from stego import Steganographer, available
from stego.core.exceptions import MethodNotFound
from stego.engines.lsb.engine import LSBVideoEngine


def test_registry_lists_lsb():
    assert "lsb" in available()


def test_by_name_and_by_instance_same_engine():
    assert isinstance(Steganographer("lsb").engine, LSBVideoEngine)
    assert isinstance(Steganographer(LSBVideoEngine()).engine, LSBVideoEngine)


def test_unknown_engine_raises():
    with pytest.raises(MethodNotFound):
        Steganographer("nope")
