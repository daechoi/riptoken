"""Round-trip: encode then decode must return the original bytes."""

from __future__ import annotations

import pytest

from riptoken import CoreBPE

# Minimal toy vocab that doesn't need an external file.
_TOY_VOCAB = {
    b" ": 0,
    b"a": 1,
    b"b": 2,
    b"c": 3,
    b"d": 4,
    b"e": 5,
    b"h": 6,
    b"l": 7,
    b"o": 8,
    b"r": 9,
    b"w": 10,
    b"ab": 100,
    b"cd": 101,
    b"he": 102,
    b"ll": 103,
    b"world": 104,
}


@pytest.fixture
def toy_bpe() -> CoreBPE:
    return CoreBPE(_TOY_VOCAB, {}, r"\w+|\s+")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "a",
        "ab",
        "abcd",
        "hello world",
        "ab cd ab cd ab",
    ],
)
def test_roundtrip_toy(toy_bpe: CoreBPE, text: str) -> None:
    tokens = toy_bpe.encode_ordinary(text)
    assert toy_bpe.decode_bytes(tokens) == text.encode()


@pytest.mark.parametrize(
    "text",
    [
        "",
        "The quick brown fox jumps over the lazy dog.",
        "def f(x):\n    return x * 2\n",
        "     leading whitespace",
        "trailing whitespace     ",
        "\n\n\ttabs and newlines\n",
        "Émoji: 🚀🔥💯",
        "中文 日本語 한국어",
        "mixed 中文 English emoji 🎉 numbers 12345",
    ],
)
def test_roundtrip_o200k(o200k_bpe: CoreBPE, text: str) -> None:
    tokens = o200k_bpe.encode_ordinary(text)
    assert o200k_bpe.decode_bytes(tokens) == text.encode()
