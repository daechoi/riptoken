# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/daechoi/riptoken/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/daechoi/riptoken/releases/tag/v0.1.0
