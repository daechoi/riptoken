"""Shared fixtures for riptoken's Python test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from riptoken import CoreBPE, load_tiktoken_bpe

# The o200k_base pattern used by GPT-4o / GPT-4.1 family models.
O200K_PATTERN = "|".join(
    [
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)

O200K_SPECIALS = {
    "<|endoftext|>": 199999,
    "<|endofprompt|>": 200018,
}


def _find_vocab() -> Path | None:
    """Find o200k_base.tiktoken in the project root or tiktoken cache."""
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor / "o200k_base.tiktoken"
        if candidate.exists():
            return candidate
    return None


@pytest.fixture(scope="session")
def o200k_vocab_path() -> Path:
    path = _find_vocab()
    if path is None:
        pytest.skip("o200k_base.tiktoken not available in project root")
    return path


@pytest.fixture(scope="session")
def o200k_ranks(o200k_vocab_path: Path) -> dict[bytes, int]:
    return load_tiktoken_bpe(o200k_vocab_path)


@pytest.fixture(scope="session")
def o200k_bpe(o200k_ranks: dict[bytes, int]) -> CoreBPE:
    return CoreBPE(o200k_ranks, O200K_SPECIALS, O200K_PATTERN)


@pytest.fixture(scope="session")
def tiktoken_o200k():
    tiktoken = pytest.importorskip("tiktoken")
    return tiktoken.get_encoding("o200k_base")
