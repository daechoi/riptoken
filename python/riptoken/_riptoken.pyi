"""Type stubs for the riptoken Rust extension module."""

from __future__ import annotations

from collections.abc import Sequence, Set

class CoreBPE:
    """Core BPE encoder/decoder. Backed by a Rust implementation.

    Instances are safe to share across threads — all heavy work releases the
    GIL and uses per-thread regex state internally.
    """

    def __init__(
        self,
        encoder: dict[bytes, int],
        special_tokens_encoder: dict[str, int],
        pattern: str,
    ) -> None:
        """Construct a new BPE encoder.

        Parameters
        ----------
        encoder:
            The vocabulary: a mapping from token bytes to integer rank.
            Usually loaded via :func:`riptoken.load_tiktoken_bpe`.
        special_tokens_encoder:
            Mapping from special-token string (e.g. ``"<|endoftext|>"``)
            to its assigned rank. Pass an empty dict if you have none.
        pattern:
            The Unicode-aware regex used to split text into pieces before
            BPE is applied. Must be compatible with the ``fancy-regex`` crate
            (supports ``\\p{L}``, ``\\p{N}``, lookaheads, etc.).

        Raises
        ------
        ValueError
            If the regex pattern fails to compile.
        """

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode ``text`` as ordinary text, ignoring special tokens.

        Any special-token substrings in the input are tokenized as regular
        text. For strict behavior use :meth:`encode` with an explicit allow
        set. Releases the GIL during encoding.
        """

    def encode(self, text: str, allowed_special: Set[str]) -> list[int]:
        """Encode ``text`` with the given set of allowed special tokens.

        Special tokens in ``allowed_special`` are emitted as their assigned
        ranks. Special tokens NOT in the set are tokenized as ordinary text.
        Releases the GIL during encoding.
        """

    def encode_with_special_tokens(self, text: str) -> list[int]:
        """Encode ``text``, allowing every registered special token.

        This is equivalent to :meth:`encode` with all special-token strings in
        the allow set, without materializing that set in Python. Releases the
        GIL during encoding.
        """

    def encode_ordinary_batch(self, texts: Sequence[str]) -> list[list[int]]:
        """Encode many texts in parallel, ignoring special tokens.

        Each text is tokenized on a rayon worker thread using the global
        rayon pool. Level of parallelism is controlled by the
        ``RAYON_NUM_THREADS`` environment variable. Results preserve input
        order. Releases the GIL for the duration of the call.
        """

    def encode_batch(
        self, texts: Sequence[str], allowed_special: Set[str]
    ) -> list[list[int]]:
        """Encode many texts in parallel with the given allowed special tokens.

        See :meth:`encode` for the per-text semantics and
        :meth:`encode_ordinary_batch` for the parallelism model.
        """

    def encode_with_special_tokens_batch(
        self, texts: Sequence[str]
    ) -> list[list[int]]:
        """Encode many texts in parallel, allowing every registered special token.

        See :meth:`encode_with_special_tokens` for the per-text semantics and
        :meth:`encode_ordinary_batch` for the parallelism model.
        """

    def encode_single_token(self, piece: bytes) -> int:
        """Look up a single token by its exact byte sequence.

        Raises
        ------
        KeyError
            If ``piece`` is not in the vocabulary.
        """

    def decode_bytes(self, tokens: list[int]) -> bytes:
        """Decode a sequence of tokens into the underlying bytes.

        Unknown token ids are silently skipped — this matches tiktoken's
        ``decode_bytes`` behavior. Releases the GIL.
        """

    def decode_single_token_bytes(self, token: int) -> bytes:
        """Look up the bytes of a single token.

        Raises
        ------
        KeyError
            If ``token`` is not in the vocabulary.
        """

    def n_vocab(self) -> int:
        """Total number of tokens in the vocabulary (including special tokens)."""

    def token_byte_values(self) -> list[bytes]:
        """A sorted list of every non-special token's bytes."""
