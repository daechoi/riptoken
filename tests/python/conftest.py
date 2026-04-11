"""Shared fixtures for riptoken's Python test suite."""

from __future__ import annotations

import pytest

import riptoken
from riptoken import CoreBPE


@pytest.fixture(scope="session")
def o200k_bpe() -> CoreBPE:
    """o200k_base encoder loaded via the tiktoken-backed helper.

    Uses ``riptoken.get_encoding``, which reads the vocabulary, regex
    pattern, and special-token map from tiktoken (and tiktoken's on-disk
    cache) — no local ``o200k_base.tiktoken`` file required.
    """
    pytest.importorskip("tiktoken")
    return riptoken.get_encoding("o200k_base")


@pytest.fixture(scope="session")
def tiktoken_o200k():
    tiktoken = pytest.importorskip("tiktoken")
    return tiktoken.get_encoding("o200k_base")
