"""riptoken — fast BPE tokenizer for LLMs.

A drop-in compatible, faster reimplementation of OpenAI's tiktoken. riptoken
reads the same ``.tiktoken`` vocabulary files and produces byte-identical
output for every tested corpus.

Quick start
-----------

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

__version__ = "0.1.0"
__all__ = ["CoreBPE", "load_tiktoken_bpe"]


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
