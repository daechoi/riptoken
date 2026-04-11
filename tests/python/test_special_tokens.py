"""Tests for special-token handling (the ``encode`` method)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from riptoken import CoreBPE


def test_encode_ordinary_ignores_special_tokens(o200k_bpe: "CoreBPE") -> None:
    # encode_ordinary must NOT emit special-token ranks even if the string
    # contains a special-token substring.
    text = "hello <|endoftext|> world"
    tokens = o200k_bpe.encode_ordinary(text)
    assert 199999 not in tokens


def test_encode_with_allowed_special_emits_token(o200k_bpe: "CoreBPE") -> None:
    text = "hello <|endoftext|> world"
    tokens = o200k_bpe.encode(text, {"<|endoftext|>"})
    assert 199999 in tokens


def test_encode_with_disallowed_special_does_not_emit(o200k_bpe: "CoreBPE") -> None:
    text = "hello <|endoftext|> world"
    tokens = o200k_bpe.encode(text, set())
    assert 199999 not in tokens


def test_encode_multiple_specials(o200k_bpe: "CoreBPE") -> None:
    text = "a <|endoftext|> b <|endofprompt|> c"
    tokens = o200k_bpe.encode(
        text, {"<|endoftext|>", "<|endofprompt|>"}
    )
    assert 199999 in tokens
    assert 200018 in tokens


def test_encode_empty_string(o200k_bpe: "CoreBPE") -> None:
    assert o200k_bpe.encode_ordinary("") == []
    assert o200k_bpe.encode("", set()) == []


def test_decode_single_token(o200k_bpe: "CoreBPE") -> None:
    # 199999 is <|endoftext|>
    assert o200k_bpe.decode_single_token_bytes(199999) == b"<|endoftext|>"


def test_decode_single_token_unknown(o200k_bpe: "CoreBPE") -> None:
    with pytest.raises(KeyError):
        o200k_bpe.decode_single_token_bytes(999_999_999)
