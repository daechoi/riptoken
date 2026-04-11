"""Tests for :func:`riptoken.load_tiktoken_bpe`."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from riptoken import load_tiktoken_bpe


def _write_vocab(path: Path, entries: list[tuple[bytes, int]]) -> None:
    lines = [
        b"%s %d\n" % (base64.b64encode(tok), rank) for tok, rank in entries
    ]
    path.write_bytes(b"".join(lines))


def test_load_basic(tmp_path: Path) -> None:
    vocab = tmp_path / "test.tiktoken"
    _write_vocab(
        vocab,
        [
            (b"hello", 0),
            (b" world", 1),
            (b"!", 2),
        ],
    )

    ranks = load_tiktoken_bpe(vocab)
    assert ranks[b"hello"] == 0
    assert ranks[b" world"] == 1
    assert ranks[b"!"] == 2
    assert len(ranks) == 3


def test_load_skips_blank_lines(tmp_path: Path) -> None:
    vocab = tmp_path / "test.tiktoken"
    vocab.write_bytes(
        b"\n"
        + base64.b64encode(b"a")
        + b" 0\n\n"
        + base64.b64encode(b"b")
        + b" 1\n\n"
    )
    ranks = load_tiktoken_bpe(vocab)
    assert ranks == {b"a": 0, b"b": 1}


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_tiktoken_bpe(tmp_path / "nonexistent.tiktoken")


def test_load_malformed_line(tmp_path: Path) -> None:
    vocab = tmp_path / "bad.tiktoken"
    vocab.write_bytes(b"not a real line\n")
    with pytest.raises(ValueError):
        load_tiktoken_bpe(vocab)


def test_load_accepts_str_and_path(tmp_path: Path) -> None:
    vocab = tmp_path / "test.tiktoken"
    _write_vocab(vocab, [(b"x", 0)])
    assert load_tiktoken_bpe(str(vocab)) == {b"x": 0}
    assert load_tiktoken_bpe(vocab) == {b"x": 0}
