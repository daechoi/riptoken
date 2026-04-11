"""Parity tests: riptoken output must match tiktoken byte-for-byte.

These tests require ``tiktoken`` to be importable; the ``o200k_bpe``
fixture loads the encoder via ``riptoken.get_encoding("o200k_base")``,
which goes through tiktoken's on-disk cache so no local
``.tiktoken`` file is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from riptoken import CoreBPE


PARITY_CORPUS = [
    "",
    "a",
    "hello",
    "hello world",
    "The quick brown fox jumps over the lazy dog.",
    "     leading whitespace",
    "trailing whitespace     ",
    "mixed\twhitespace\nand\rreturn",
    "punctuation !!! ??? ,,, ... ;;; ~~~",
    "def main():\n    print('hello, world!')\n    return 0\n",
    "#include <stdio.h>\nint main(void) { printf(\"hi\\n\"); return 0; }",
    "SELECT id, name FROM users WHERE age > 18 AND status = 'active';",
    "Émoji: 🚀🔥💯🎉✨💡",
    "中文测试 日本語テスト 한국어 시험",
    "Русский язык тест",
    "Γειά σας, κόσμε",
    "مرحبا بالعالم",
    "café au lait naïve façade résumé",
    "zero-width\u200bjoiner\u200dtest",
    "combining: e\u0301 a\u0300 n\u0303",
    "0123456789 " * 20,
    "aaaa " * 100,
    "\n\n\n\n\n",
    "tab\tseparated\tvalues",
]


@pytest.mark.parametrize("text", PARITY_CORPUS)
def test_encode_ordinary_matches_tiktoken(
    o200k_bpe: "CoreBPE", tiktoken_o200k, text: str
) -> None:
    rip = o200k_bpe.encode_ordinary(text)
    tik = tiktoken_o200k.encode_ordinary(text)
    assert rip == tik, (
        f"mismatch on {text!r}: "
        f"riptoken={rip[:10]}... (len={len(rip)}), "
        f"tiktoken={tik[:10]}... (len={len(tik)})"
    )


def test_long_text_parity(o200k_bpe: "CoreBPE", tiktoken_o200k) -> None:
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
    ) * 500
    assert o200k_bpe.encode_ordinary(text) == tiktoken_o200k.encode_ordinary(text)


def test_long_unicode_parity(o200k_bpe: "CoreBPE", tiktoken_o200k) -> None:
    text = "これはテストです。" * 300 + "🌍" * 100 + "café " * 100
    assert o200k_bpe.encode_ordinary(text) == tiktoken_o200k.encode_ordinary(text)


def test_vocabulary_size_matches_tiktoken(
    o200k_bpe: "CoreBPE", tiktoken_o200k
) -> None:
    # tiktoken's n_vocab includes special tokens; so does ours.
    assert o200k_bpe.n_vocab() == tiktoken_o200k.n_vocab
