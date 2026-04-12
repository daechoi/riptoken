"""riptoken — fast BPE tokenizer for LLMs.

A drop-in compatible, faster reimplementation of OpenAI's tiktoken. riptoken
reads the same ``.tiktoken`` vocabulary files and produces byte-identical
output for every tested corpus.

Quick start
-----------

The simplest way to get a tokenizer:

>>> import riptoken
>>> enc = riptoken.get_encoding("o200k_base")
>>> enc.encode_ordinary("Hello, world!")
[13225, 11, 2375, 0]

The returned object is a :class:`riptoken.Encoding` — a drop-in replacement
for :class:`tiktoken.Encoding`. The hot-path methods (``encode``,
``encode_ordinary``, ``decode``, ``decode_bytes``, and their batch variants)
run through riptoken's fast Rust core; every other attribute and method
forwards transparently to the underlying ``tiktoken.Encoding`` instance, so
code written against the ``tiktoken`` API works unchanged.

Or construct a :class:`CoreBPE` directly from a ``.tiktoken`` file if you
want to avoid the ``tiktoken`` dependency entirely:

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
from typing import Any, Iterable, Union

from riptoken._riptoken import CoreBPE

__version__ = "0.3.0"
__all__ = [
    "CoreBPE",
    "Encoding",
    "encoding_for_model",
    "get_encoding",
    "load_tiktoken_bpe",
]


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


class Encoding:
    """Drop-in replacement for :class:`tiktoken.Encoding`.

    Holds a riptoken :class:`CoreBPE` and the underlying
    :class:`tiktoken.Encoding` it was built from. Hot-path methods —
    ``encode``, ``encode_ordinary``, ``decode``, ``decode_bytes``, and their
    batch variants — execute in riptoken's Rust core and release the GIL.
    Every other attribute and method is forwarded to the wrapped
    ``tiktoken.Encoding`` via :meth:`__getattr__`, so properties like
    ``n_vocab``, ``eot_token``, ``special_tokens_set``, and less-common
    methods like ``encode_with_unstable`` work transparently.

    You usually obtain an ``Encoding`` via :func:`riptoken.get_encoding` or
    :func:`riptoken.encoding_for_model`; constructing one directly is
    unusual.

    Notes
    -----
    This class is duck-typed as a ``tiktoken.Encoding`` — it does not
    subclass it. Code that checks ``isinstance(enc, tiktoken.Encoding)``
    will need to be updated; code that calls methods on it will not.
    """

    __slots__ = ("_core", "_tiktoken")

    def __init__(self, core: CoreBPE, tiktoken_encoding: Any) -> None:
        object.__setattr__(self, "_core", core)
        object.__setattr__(self, "_tiktoken", tiktoken_encoding)

    # ---- Hot-path methods: run in riptoken's Rust core ----

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode ``text`` without recognizing any special tokens.

        Equivalent to ``tiktoken.Encoding.encode_ordinary``. Releases the
        GIL.
        """
        return self._core.encode_ordinary(text)

    def encode(
        self,
        text: str,
        allowed_special: Union[set[str], str, None] = None,
        disallowed_special: Union[set[str], str, None] = None,
    ) -> list[int]:
        """Encode ``text``, optionally recognizing special tokens.

        ``allowed_special`` may be a set of special-token strings to
        recognize, or the sentinel string ``"all"`` to allow every special
        token in the vocabulary. ``None`` (the default) disallows all.

        ``disallowed_special`` is accepted for ``tiktoken`` signature
        compatibility but is currently treated as a hint — unrecognized
        specials pass through as ordinary text rather than raising.
        """
        if allowed_special is None:
            allowed: set[str] = set()
        elif allowed_special == "all":
            allowed = set(self._tiktoken.special_tokens_set)
        else:
            allowed = set(allowed_special)
        return self._core.encode(text, allowed)

    def encode_ordinary_batch(self, texts: Iterable[str]) -> list[list[int]]:
        """Parallel batch version of :meth:`encode_ordinary`.

        Fans out to ``rayon``'s global thread pool, releasing the GIL for
        the full batch.
        """
        return self._core.encode_ordinary_batch(list(texts))

    def encode_batch(
        self,
        texts: Iterable[str],
        allowed_special: Union[set[str], str, None] = None,
        disallowed_special: Union[set[str], str, None] = None,
    ) -> list[list[int]]:
        """Parallel batch version of :meth:`encode`.

        Fans out to ``rayon``'s global thread pool, releasing the GIL for
        the full batch.
        """
        if allowed_special is None:
            allowed: set[str] = set()
        elif allowed_special == "all":
            allowed = set(self._tiktoken.special_tokens_set)
        else:
            allowed = set(allowed_special)
        return self._core.encode_batch(list(texts), allowed)

    def decode(self, tokens: list[int], errors: str = "replace") -> str:
        """Decode ``tokens`` into a string.

        ``errors`` is accepted for tiktoken signature compatibility; only
        ``"replace"`` (the default, matching tiktoken) is honored — invalid
        UTF-8 byte sequences are replaced with U+FFFD.
        """
        return self._core.decode(tokens)

    def decode_bytes(self, tokens: list[int]) -> bytes:
        """Decode ``tokens`` into raw bytes (no UTF-8 conversion)."""
        return self._core.decode_bytes(tokens)

    # ---- Everything else: forward to tiktoken ----

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is only called when normal attribute lookup fails, so
        # the hot-path methods defined above are never routed through here.
        return getattr(self._tiktoken, name)

    def __repr__(self) -> str:
        name = getattr(self._tiktoken, "name", "?")
        return f"<riptoken.Encoding {name!r}>"


def _build_encoding(tiktoken_encoding: Any) -> Encoding:
    """Build a :class:`riptoken.Encoding` from a :class:`tiktoken.Encoding`.

    Reads the vocab, pattern string, and special tokens from stable (if
    technically private) attributes that tiktoken has exposed since its
    first release.
    """
    # These three attrs have been stable since tiktoken 0.3 and are what
    # tiktoken itself passes into its own Rust constructor.
    mergeable_ranks = tiktoken_encoding._mergeable_ranks  # type: ignore[attr-defined]
    special_tokens = tiktoken_encoding._special_tokens  # type: ignore[attr-defined]
    pat_str = tiktoken_encoding._pat_str  # type: ignore[attr-defined]
    core = CoreBPE(mergeable_ranks, special_tokens, pat_str)
    return Encoding(core, tiktoken_encoding)


def get_encoding(name: str) -> Encoding:
    """Load a named tiktoken encoding (``"gpt2"``, ``"cl100k_base"``, ``"o200k_base"``, ...).

    This is the riptoken equivalent of :func:`tiktoken.get_encoding`. It
    uses the ``tiktoken`` package to supply the vocabulary file, regex
    pattern, and special-token map, then wraps them in a
    :class:`riptoken.Encoding` whose hot-path methods execute in
    riptoken's faster Rust core. Subsequent calls with the same ``name``
    reuse tiktoken's in-process cache, and the vocabulary download (if
    needed) uses tiktoken's on-disk cache at ``~/.cache/tiktoken/`` (or
    wherever ``TIKTOKEN_CACHE_DIR`` points).

    Parameters
    ----------
    name:
        A tiktoken encoding name, e.g. ``"gpt2"``, ``"r50k_base"``,
        ``"p50k_base"``, ``"cl100k_base"``, ``"o200k_base"``.

    Returns
    -------
    Encoding
        A drop-in replacement for the corresponding
        :class:`tiktoken.Encoding`, producing byte-identical output on all
        hot-path methods.

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
    return _build_encoding(tiktoken.get_encoding(name))


def encoding_for_model(model_name: str) -> Encoding:
    """Load the encoding used by a specific OpenAI model.

    This is the riptoken equivalent of :func:`tiktoken.encoding_for_model`.
    For example, ``encoding_for_model("gpt-4o")`` returns an ``o200k_base``
    encoder wrapped in a :class:`riptoken.Encoding`. See
    :func:`get_encoding` for dependency notes.

    Parameters
    ----------
    model_name:
        An OpenAI model name, e.g. ``"gpt-4"``, ``"gpt-4o"``, ``"gpt-3.5-turbo"``.

    Returns
    -------
    Encoding
        A drop-in replacement for the corresponding
        :class:`tiktoken.Encoding`.

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
    return _build_encoding(tiktoken.encoding_for_model(model_name))
