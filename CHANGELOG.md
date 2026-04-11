# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `CoreBPE::encode_ordinary_batch` and `CoreBPE::encode_batch` for parallel
  batch encoding, fanning out to rayon's global thread pool. Python bindings
  (`encode_ordinary_batch`, `encode_batch`) release the GIL for the full batch.

### Changed

- The SIMD fast-path engine now uses the `regex` crate for tiktoken-style
  patterns whose lookaround can be peeled off (all of `cl100k_base`,
  `o200k_base`, `p50k_base`). The `\s+(?!\S)` whitespace-shrink rule is
  reproduced in Rust code, producing byte-identical output to tiktoken.
  Patterns with other lookarounds fall back to `fancy-regex` as before.
- Fast-path regex now uses per-thread clones (same pattern as the fancy
  engine) so the DFA cache pool is not contended at high thread counts.
  On 32-core Sapphire Rapids, parallel throughput improved from
  ~95M tps (6.2×) to ~290M tps (19×) for o200k_base batch encode.
- Micro-optimizations: pre-allocated output vectors sized from input length,
  `#[inline(always)]` on the hot `rank_of` helper, zero-allocation
  `HashMap::get(&[u8])` lookup replacing the prior `.to_vec()`.

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
