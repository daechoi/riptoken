"""riptoken — fast BPE tokenizer for LLMs.

A drop-in compatible, faster reimplementation of OpenAI's tiktoken. riptoken
reads the same ``.tiktoken`` vocabulary files and produces byte-identical
output for every tested corpus.

Quick start
-----------

The simplest way to get a tokenizer, if you have ``tiktoken`` installed:

>>> import riptoken
>>> enc = riptoken.get_encoding("o200k_base")
>>> enc.encode_ordinary("Hello, world!")
[13225, 11, 2375, 0]

Or construct a :class:`CoreBPE` directly from a ``.tiktoken`` file:

>>> import riptoken
>>> ranks = riptoken.load_tiktoken_bpe("o200k_base.tiktoken")
>>> special_tokens = {"<|endoftext|>": 199999}
>>> pat = r"\\w+|\\s+"  # simplified pattern; use the full o200k pattern in prod
>>> enc = riptoken.CoreBPE(ranks, special_tokens, pat)
>>> enc.decode_bytes(enc.encode_ordinary("Hello, world!"))
b'Hello, world!'

See https://github.com/daechoi/riptoken for full documentation, benchmarks,
and the canonical ``o200k_base`` regex pattern.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Union

from riptoken._riptoken import CoreBPE

__version__ = "0.2.0"
__all__ = ["CoreBPE", "encoding_for_model", "get_encoding", "load_tiktoken_bpe"]


def load_tiktoken_bpe(path: Union[str, Path]) -> dict[bytes, int]:
    """Load a tiktoken ``.tiktoken`` vocabulary file.

    The file format is one token per line::

        <base64-encoded token bytes> <integer rank>

    This is the same format OpenAI ships for ``cl100k_base``, ``o200k_base``,
    etc. You can obtain vocabularies from the ``tiktoken`` package's cache
    directory or download them directly from OpenAI's public CDN.

    Parameters
    ----------
    path:
        Filesystem path to the ``.tiktoken`` file.

    Returns
    -------
    dict[bytes, int]
        A dictionary mapping token bytes to their integer rank, suitable for
        passing to :class:`CoreBPE`'s first positional argument.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If a line in the file is malformed.
    """
    ranks: dict[bytes, int] = {}
    with open(path, "rb") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"{path}: line {lineno}: expected '<b64> <rank>', got {line!r}"
                )
            token_b64, rank_str = parts
            try:
                token_bytes = base64.b64decode(token_b64)
                rank = int(rank_str)
            except (ValueError, base64.binascii.Error) as e:
                raise ValueError(
                    f"{path}: line {lineno}: failed to parse token/rank: {e}"
                ) from e
            ranks[token_bytes] = rank
    return ranks


def _wrap_tiktoken_encoding(enc: object) -> CoreBPE:
    """Build a :class:`CoreBPE` from a :class:`tiktoken.Encoding` instance.

    Reads the vocab, pattern string, and special tokens from stable (if
    technically private) attributes that tiktoken has exposed since its
    first release.
    """
    # These three attrs have been stable since tiktoken 0.3 and are what
    # tiktoken itself passes into its own Rust constructor.
    mergeable_ranks = enc._mergeable_ranks  # type: ignore[attr-defined]
    special_tokens = enc._special_tokens  # type: ignore[attr-defined]
    pat_str = enc._pat_str  # type: ignore[attr-defined]
    return CoreBPE(mergeable_ranks, special_tokens, pat_str)


def get_encoding(name: str) -> CoreBPE:
    """Load a named tiktoken encoding (``"gpt2"``, ``"cl100k_base"``, ``"o200k_base"``, ...).

    This is the riptoken equivalent of :func:`tiktoken.get_encoding`. It
    soft-depends on the ``tiktoken`` package to supply the vocabulary file,
    regex pattern, and special-token map, then wraps them in riptoken's
    faster Rust core. Subsequent calls with the same ``name`` reuse
    tiktoken's in-process cache, and the vocabulary download (if needed)
    uses tiktoken's on-disk cache at ``~/.cache/tiktoken/`` (or wherever
    ``TIKTOKEN_CACHE_DIR`` points).

    Parameters
    ----------
    name:
        A tiktoken encoding name, e.g. ``"gpt2"``, ``"r50k_base"``,
        ``"p50k_base"``, ``"cl100k_base"``, ``"o200k_base"``.

    Returns
    -------
    CoreBPE
        A riptoken encoder producing byte-identical output to
        ``tiktoken.get_encoding(name)``.

    Raises
    ------
    ImportError
        If the ``tiktoken`` package is not installed. Install it with
        ``pip install tiktoken``, or construct :class:`CoreBPE` directly
        from a ``.tiktoken`` file via :func:`load_tiktoken_bpe`.
    """
    try:
        import tiktoken
    except ImportError as e:
        raise ImportError(
            "riptoken.get_encoding requires the `tiktoken` package to supply "
            "vocabulary files and regex patterns. Install it with "
            "`pip install tiktoken`, or construct CoreBPE directly using "
            "riptoken.load_tiktoken_bpe()."
        ) from e
    return _wrap_tiktoken_encoding(tiktoken.get_encoding(name))


def encoding_for_model(model_name: str) -> CoreBPE:
    """Load the encoding used by a specific OpenAI model.

    This is the riptoken equivalent of :func:`tiktoken.encoding_for_model`.
    For example, ``encoding_for_model("gpt-4o")`` returns an ``o200k_base``
    encoder. See :func:`get_encoding` for dependency notes.

    Parameters
    ----------
    model_name:
        An OpenAI model name, e.g. ``"gpt-4"``, ``"gpt-4o"``, ``"gpt-3.5-turbo"``.

    Returns
    -------
    CoreBPE
        A riptoken encoder producing byte-identical output to
        ``tiktoken.encoding_for_model(model_name)``.

    Raises
    ------
    ImportError
        If the ``tiktoken`` package is not installed.
    """
    try:
        import tiktoken
    except ImportError as e:
        raise ImportError(
            "riptoken.encoding_for_model requires the `tiktoken` package to "
            "supply vocabulary files and regex patterns. Install it with "
            "`pip install tiktoken`, or construct CoreBPE directly using "
            "riptoken.load_tiktoken_bpe()."
        ) from e
    return _wrap_tiktoken_encoding(tiktoken.encoding_for_model(model_name))
