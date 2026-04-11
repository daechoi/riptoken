"""Tests for the ``get_encoding`` / ``encoding_for_model`` convenience helpers.

These helpers soft-depend on ``tiktoken`` and are skipped if it is not
installed.
"""

from __future__ import annotations

import pytest

import riptoken

tiktoken = pytest.importorskip("tiktoken")


CORPUS = [
    "",
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "def main():\n    print('hello, world!')\n    return 0\n",
    "Émoji: 🚀🔥💯   多言語テスト",
    "trailing whitespace     ",
    "     leading whitespace",
]


@pytest.mark.parametrize(
    "encoding_name", ["gpt2", "r50k_base", "p50k_base", "cl100k_base", "o200k_base"]
)
def test_get_encoding_parity_with_tiktoken(encoding_name: str) -> None:
    try:
        tik = tiktoken.get_encoding(encoding_name)
    except Exception as e:
        # gpt2 / r50k_base require downloading encoder.json + vocab.bpe on
        # first use; skip if the network or cache isn't available.
        pytest.skip(f"tiktoken.get_encoding({encoding_name!r}) failed: {e}")
    rip = riptoken.get_encoding(encoding_name)
    assert rip.n_vocab == tik.n_vocab
    for text in CORPUS:
        assert rip.encode_ordinary(text) == tik.encode_ordinary(text), (
            f"mismatch on {encoding_name!r} / {text!r}"
        )


def test_encoding_for_model_parity_with_tiktoken() -> None:
    # gpt-4o uses o200k_base, which loads from a .tiktoken file (no network
    # round-trip needed if tiktoken's on-disk cache is present).
    try:
        tik = tiktoken.encoding_for_model("gpt-4o")
    except Exception as e:
        pytest.skip(f"tiktoken.encoding_for_model('gpt-4o') failed: {e}")
    rip = riptoken.encoding_for_model("gpt-4o")
    assert rip.n_vocab == tik.n_vocab
    for text in CORPUS:
        assert rip.encode_ordinary(text) == tik.encode_ordinary(text)


def test_get_encoding_missing_tiktoken_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """If tiktoken cannot be imported, get_encoding should raise ImportError."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("no tiktoken for you")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="tiktoken"):
        riptoken.get_encoding("o200k_base")
