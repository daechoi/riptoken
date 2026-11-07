# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Pieces of 500 bytes or more now merge equal-rank pairs from the left,
  as tiktoken does. Before, the heap merge picked the rightmost pair, so
  long runs of one character (e.g. `"a" + "-" * 500 + "b"` with
  `cl100k_base`) could encode differently from tiktoken (#2).

## [0.3.0] — 2026-04-12

Performance release. Eliminates the ~55 ms lazy-DFA cold-start on
Unicode-heavy text by pre-compiling fully-materialized dense DFAs for
all stock tiktoken patterns at build time. Construction time is unchanged;
every encode call is now the fast path from the first invocation.

### Added

- **Pre-compiled dense DFAs.** Stock tiktoken patterns (`gpt2`,
  `r50k_base`, `p50k_base`, `p50k_edit`, `cl100k_base`, `o200k_base`)
  are compiled into dense DFAs at build time via `regex-automata` and
  embedded in the binary. At runtime, stock patterns are detected by
  exact string match and the pre-built DFA is deserialized at near-zero
  cost — no lazy state-building on the first call.
- `precompiled-dfa` Cargo feature (default on). Disable with
  `--no-default-features` for smaller binaries at the cost of a
  first-call warm-up.
- `build.rs` — compiles and serializes forward + reverse dense DFAs for
  each stock pattern, with correct endianness for cross-compilation via
  `CARGO_CFG_TARGET_ENDIAN`.
- `examples/dfa_spike.rs` — measurement spike that reports dense/sparse
  DFA sizes and build times for each stock pattern.
- New `PrecompiledDfa` variant in the internal `SplitEngine` enum. The
  dense DFA has no mutable state, so one instance is shared across all
  threads without per-thread cloning.

### Changed

- Engine selection priority is now: pre-built DFA (stock patterns) →
  eager dense DFA build (non-stock) → lazy DFA → fancy-regex.
- Pattern transformation logic extracted into a shared `transform_pattern`
  function, reused by both the lazy-DFA and dense-DFA builders.
- Whitespace-shrink post-processing extracted into `apply_shrink` helper,
  shared across the `PrecompiledDfa` and `Fast` engine variants.
- `regex-automata = "0.4"` added as a direct dependency (previously only
  a transitive dependency via `regex`).

### Performance

First-call latency on 141 K character CJK text, `o200k_base`:

| Metric               | v0.2.4 (lazy DFA) | v0.3.0 (precompiled) |
| -------------------- | -----------------:| --------------------:|
| Construction         |          ~200 ms  |            ~210 ms   |
| First encode (cold)  |           ~55 ms  |           **~5.7 ms**|
| Second encode (warm) |            ~8 ms  |           **~5.7 ms**|

## [0.2.4] — 2026-04-12

Bugfix release. Fixes incorrect tokenization on the `cl100k_base`
encoding for text containing `\r\n` sequences. All five stock tiktoken
encodings are now verified byte-identical against 23.8 MB of real-world
internet text (Project Gutenberg, Linux/CPython/Rust source, multilingual
Wikipedia).

### Fixed

- `cl100k_base` produced wrong tokens around `\r\n`: the SIMD
  fast-path whitespace-shrink rule incorrectly classified `cl100k_base`
  as `Unified` mode instead of `PlainOnly`, causing `\r\n` matches from
  the `\s*[\r\n]` alternative to be split into separate `\r` and `\n`
  pieces. The fix detects the presence of a dedicated newline alternative
  (`\s*[\r\n]`) in the pattern rather than relying on the `\s+` vs `\s`
  suffix to distinguish the two families.

### Added

- `scripts/parity_internet.py` — downloads large real-world corpora
  (Gutenberg books, kernel source, multilingual Wikipedia) and verifies
  byte-identical output against tiktoken across all encodings.

## [0.2.3] — 2026-04-11

Minor release. Closes the remaining tiktoken API surface gap by wrapping
the Rust `CoreBPE` in a new Python `Encoding` class that forwards every
non-hot-path attribute and method to the underlying `tiktoken.Encoding`.
Code written against `tiktoken.Encoding` — `n_vocab`, `eot_token`,
`special_tokens_set`, `encode_with_unstable`, `decode_single_token_bytes`,
etc. — now works unchanged against a `riptoken.get_encoding` result.
Also adds Python 3.14 to CI and the release wheels matrix.

### Added

- `riptoken.Encoding` — a drop-in replacement for `tiktoken.Encoding`.
  Hot-path methods (`encode`, `encode_ordinary`, `decode`, `decode_bytes`,
  `encode_batch`, `encode_ordinary_batch`) execute in riptoken's Rust
  core and release the GIL; every other attribute falls through to the
  wrapped `tiktoken.Encoding` via `__getattr__`. `get_encoding` and
  `encoding_for_model` now return `Encoding` instances instead of bare
  `CoreBPE` instances.
- `encode` and `encode_batch` accept `allowed_special="all"` as a
  sentinel meaning "every special token in the vocabulary", matching
  tiktoken.
- Python 3.14 support — added to the CI matrix and to the release
  workflow's interpreter list for Linux and macOS wheel builds.

### Changed

- `n_vocab` is now a property (int), not a callable, matching tiktoken.
  Code that called `enc.n_vocab()` needs to drop the parentheses.
  `CoreBPE.n_vocab()` is unchanged for direct Rust-core users.

## [0.2.2] — 2026-04-11

Patch release. Closes two tiktoken API compatibility gaps in the Python
bindings so that code written against `tiktoken.Encoding` — including
the examples in Sebastian Raschka's *Build a Large Language Model
(From Scratch)* — runs unchanged against a `riptoken.get_encoding`
instance.

### Added

- `CoreBPE.decode(tokens)` — returns a Python `str`, matching
  `tiktoken.Encoding.decode`. Invalid UTF-8 sequences (which can occur
  mid-stream when a multi-byte character spans a token boundary) are
  replaced with U+FFFD, matching tiktoken's default
  `errors="replace"` behavior. The existing `decode_bytes` method is
  unchanged and remains the right choice for strict / streaming
  decoding.

### Fixed

- `CoreBPE.encode(text)` now works without an explicit
  `allowed_special` argument, matching tiktoken. Previously
  `allowed_special` was a required positional parameter, so
  `tokenizer.encode(raw_text)` raised a `TypeError`. The parameter is
  now optional and defaults to an empty set (no special tokens
  recognized), and can still be passed as a positional or keyword
  argument. `CoreBPE.encode_batch` received the same treatment.

## [0.2.1] — 2026-04-11

Patch release. Makes `tiktoken` a required runtime dependency so that
`riptoken.get_encoding("gpt2")` works out of the box after a plain
`pip install riptoken`. Previously `tiktoken` was a soft dependency
imported on demand, and first-time users hit an `ImportError` at the
most obvious entry point.

### Fixed

- `pip install riptoken` now pulls in `tiktoken>=0.7` automatically, so
  `riptoken.get_encoding` and `riptoken.encoding_for_model` work without
  a second install step. Users who want a tiktoken-free install can
  still construct `CoreBPE` directly from a `.tiktoken` file via
  `riptoken.load_tiktoken_bpe`.

## [0.2.0] — 2026-04-11

Second release. Headline changes: `get_encoding` / `encoding_for_model`
helpers, a parallel batch API, and a SIMD regex fast path that covers
every stock tiktoken encoding. Single-threaded throughput on `o200k_base`
improves from the 1.8×–2.7× range to 2.4×–6.2× over tiktoken; parallel
batch encoding scales to 19× on wide machines.

### Added

- `riptoken.get_encoding(name)` and `riptoken.encoding_for_model(model)` —
  drop-in equivalents of the tiktoken helpers. These soft-depend on
  `tiktoken` to supply vocabulary files, regex patterns, and special-token
  maps, then wrap them in riptoken's faster Rust core. Parity tested
  against tiktoken on `gpt2`, `r50k_base`, `p50k_base`, `cl100k_base`,
  and `o200k_base`.
- `CoreBPE::encode_ordinary_batch` and `CoreBPE::encode_batch` for parallel
  batch encoding, fanning out to rayon's global thread pool. Python bindings
  (`encode_ordinary_batch`, `encode_batch`) release the GIL for the full batch.

### Changed

- The SIMD fast-path engine now uses the `regex` crate for every stock
  tiktoken pattern — `gpt2`, `r50k_base`, `p50k_base`, `cl100k_base`, and
  `o200k_base`. The `\s+(?!\S)` whitespace-shrink rule is reproduced in
  Rust code, producing byte-identical output to tiktoken. Patterns with
  other lookarounds fall back to `fancy-regex` as before.
- Fast-path transform recognizes both tiktoken pattern families and
  selects the shrink rule accordingly: `o200k`/`cl100k` have a separate
  `\s*[\r\n]+` alternative, so the shrink only fires on plain whitespace
  runs; `gpt2`/`r50k`/`p50k` have no such alternative, so the shrink fires
  on any whitespace run including ones that contain newlines.
- Fast-path transform also converts possessive quantifiers (`?+`, `++`,
  `*+`, `{n,m}+`) to greedy form so the DFA engine interprets them
  correctly. In a DFA engine possessive markers are semantically
  unnecessary (no backtracking to disable) and in every tiktoken pattern
  the alternatives are disjoint enough that the rewrite is safe.
- Fast-path regex now uses per-thread clones (same pattern as the fancy
  engine) so the DFA cache pool is not contended at high thread counts.
  On 32-core Sapphire Rapids, parallel throughput improved from
  ~95M tps (6.2×) to ~290M tps (19×) for o200k_base batch encode.
- Tests, benchmark, and CI no longer require a local `o200k_base.tiktoken`
  vocab file. They route through `riptoken.get_encoding("o200k_base")`
  (and therefore tiktoken's on-disk cache) instead.
- Micro-optimizations: pre-allocated output vectors sized from input length,
  `#[inline(always)]` on the hot `rank_of` helper, zero-allocation
  `HashMap::get(&[u8])` lookup replacing the prior `.to_vec()`.

### Performance

Measured against `tiktoken 0.7+` on an M-series Mac, Python 3.13, release
builds of both libraries. Vocab: `o200k_base`. Median of 3 runs.

| Corpus                 |  Tokens | riptoken tps | tiktoken tps | Speedup |
| ---------------------- | ------: | -----------: | -----------: | ------- |
| English prose          |  40,001 |        15.7M |         3.1M | 5.03×   |
| Python source code     |  72,501 |        16.4M |         2.7M | 6.13×   |
| Rust source code       |  88,001 |        18.0M |         3.1M | 5.88×   |
| Multilingual + emoji   |  85,600 |         8.9M |         3.6M | 2.47×   |
| Random-ish bytes       | 120,000 |        18.0M |         4.3M | 4.17×   |

## [0.1.0] — 2026-04-11

First stable public release. Published to
[PyPI](https://pypi.org/project/riptoken/) and
[crates.io](https://crates.io/crates/riptoken).

### Added

- First public release. BPE tokenizer compatible with OpenAI's `tiktoken`
  vocabularies (`cl100k_base`, `o200k_base`, etc.).
- Rust crate `riptoken` with `CoreBPE` exposing `encode_ordinary`, `encode`,
  `decode_bytes`, `decode`, `encode_single_token`, `decode_single_token_bytes`,
  `n_vocab`, `token_byte_values`.
- Python package `riptoken` (PyO3 bindings), with `CoreBPE` and
  `load_tiktoken_bpe` helper.
- Linear-scan merge path for pieces under 500 bytes and a heap + intrusive
  linked-list merge path for longer pieces.
- Thread-local regex clones to avoid `fancy-regex` scratch-state contention.
- GIL released during all encode/decode work via `py.detach`.
- Comprehensive test suite: Rust unit tests, Rust integration tests, Python
  parity tests against `tiktoken`.

### Performance

Measured against `tiktoken 0.7+` on an M-series Mac, Python 3.13, release
builds of both libraries. Vocab: `o200k_base`.

| Corpus                 | Tokens   | riptoken tps | tiktoken tps | Speedup |
| ---------------------- | -------- | ------------ | ------------ | ------- |
| English prose          | 40,001   | 8.5M         | 3.2M         | 2.71×   |
| Python source code     | 72,501   | 6.7M         | 2.7M         | 2.46×   |
| Rust source code       | 88,001   | 7.2M         | 3.1M         | 2.33×   |
| Multilingual + emoji   | 85,600   | 6.8M         | 3.6M         | 1.87×   |
| Random-ish bytes       | 120,000  | 8.8M         | 4.4M         | 2.02×   |

[Unreleased]: https://github.com/daechoi/riptoken/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/daechoi/riptoken/compare/v0.2.4...v0.3.0
[0.2.4]: https://github.com/daechoi/riptoken/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/daechoi/riptoken/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/daechoi/riptoken/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/daechoi/riptoken/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/daechoi/riptoken/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/daechoi/riptoken/releases/tag/v0.1.0
