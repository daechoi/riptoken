# riptoken

**A fast BPE tokenizer for LLMs. Drop-in compatible with OpenAI's [`tiktoken`](https://github.com/openai/tiktoken), 2.4×–6.2× faster single-threaded and up to ~4× faster in parallel batch mode.**

[![PyPI](https://img.shields.io/pypi/v/riptoken.svg)](https://pypi.org/project/riptoken/)
[![Crates.io](https://img.shields.io/crates/v/riptoken.svg)](https://crates.io/crates/riptoken)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python versions](https://img.shields.io/pypi/pyversions/riptoken.svg)](https://pypi.org/project/riptoken/)

riptoken is a Rust-core BPE tokenizer that reads `tiktoken`-format vocabularies
and produces byte-identical output to `tiktoken`. It is written from scratch in
Rust, with a thin PyO3 layer, and is designed to be the fastest open-source
tokenizer you can drop into an existing `tiktoken` pipeline.

---

## Why

If you are running an LLM service and tokenizing millions of requests per hour,
every microsecond of tokenizer overhead shows up on your invoice. `tiktoken` is
great but leaves performance on the table — in its own source code the authors
comment "I tried using rayon. It wasn't really faster." riptoken is a
ground-up re-implementation that takes a different set of trade-offs and comes
out ahead on every corpus tested.

## Benchmarks

Apple Silicon (M-series), Python 3.13, `o200k_base` vocab, release builds of
both libraries, outputs verified byte-identical. Median of 3 runs.

### Single-threaded

| Corpus                 |  Tokens | riptoken (tok/s) | tiktoken (tok/s) | Speedup   |
| ---------------------- | ------: | ---------------: | ---------------: | --------- |
| English prose          |  40,001 |       15,660,106 |        3,111,537 | **5.03×** |
| Python source code     |  72,501 |       16,373,214 |        2,669,412 | **6.13×** |
| Rust source code       |  88,001 |       18,028,338 |        3,066,479 | **5.88×** |
| Multilingual + emoji   |  85,600 |        8,866,190 |        3,590,639 | **2.47×** |
| Random-ish bytes       | 120,000 |       18,028,338 |        4,328,077 | **4.17×** |

### Parallel batch (256 docs, rayon + GIL release)

| Corpus                 |     Tokens | riptoken (tok/s) | tiktoken (tok/s) | Speedup   |
| ---------------------- | ---------: | ---------------: | ---------------: | --------- |
| English prose          | 10,240,256 |       33,966,313 |       13,783,321 | **2.51×** |
| Python source code     | 18,560,256 |       43,965,336 |       11,430,058 | **3.86×** |
| Rust source code       | 22,528,256 |       48,320,152 |       13,880,179 | **3.60×** |
| Multilingual + emoji   | 21,913,600 |       31,110,914 |       15,041,445 | **2.03×** |
| Random-ish bytes       | 30,720,000 |       46,700,000 |       18,188,264 | **2.56×** |

Parallel batch scaling improves further on wider machines: on a 32-core
Sapphire Rapids box, `o200k_base` throughput hits ~290 M tok/s (19× the
single-threaded baseline).

Reproduce with:

```bash
python scripts/bench.py
```

## Install

### Python

```bash
pip install riptoken
```

Pre-built wheels are published for CPython 3.9–3.14 on Linux (x86_64, aarch64),
macOS (x86_64, arm64), and Windows (x86_64).

### Rust

```bash
cargo add riptoken
```

The `python` Cargo feature is for the PyO3 bindings — you do not need it
unless you are building the Python extension yourself.

## Quick start

### Python

```python
import riptoken

# One-liner: load any tiktoken encoding by name or model.
enc = riptoken.get_encoding("o200k_base")
# or: enc = riptoken.encoding_for_model("gpt-4o")

tokens = enc.encode_ordinary("Hello, world!")
assert enc.decode(tokens) == "Hello, world!"

# With allowed special tokens
tokens = enc.encode("Hi <|endoftext|>", allowed_special={"<|endoftext|>"})

# Every tiktoken.Encoding attribute works transparently
enc.n_vocab           # 200_019
enc.eot_token         # 199_999
enc.special_tokens_set
```

`riptoken.get_encoding` and `riptoken.encoding_for_model` are drop-in
equivalents of the `tiktoken` helpers of the same name. They return a
`riptoken.Encoding` wrapper whose hot-path methods (`encode`,
`encode_ordinary`, `decode`, `decode_bytes`, and their batch variants)
execute in riptoken's faster Rust core; every other attribute and method
— `n_vocab`, `eot_token`, `special_tokens_set`, `encode_with_unstable`,
etc. — forwards transparently to the underlying `tiktoken.Encoding`.
Vocabulary files and regex patterns come from tiktoken's on-disk cache
at `~/.cache/tiktoken/`. Byte-identical output, single import change.

If you'd rather skip the `tiktoken` dependency and load a `.tiktoken` file
yourself:

```python
import riptoken

ranks = riptoken.load_tiktoken_bpe("o200k_base.tiktoken")
special_tokens = {"<|endoftext|>": 199999, "<|endofprompt|>": 200018}
pat = (
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"""
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"""
    r"""\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)
enc = riptoken.CoreBPE(ranks, special_tokens, pat)
```

### Rust

```rust
use riptoken::CoreBPE;
use rustc_hash::FxHashMap;

// Populate `encoder` from your vocabulary file (see `load_tiktoken_bpe` in
// the Python package for the format).
let encoder: FxHashMap<Vec<u8>, u32> = load_ranks("o200k_base.tiktoken");
let specials: FxHashMap<String, u32> = FxHashMap::default();
let pat = r"\w+|\s+";

let bpe = CoreBPE::new(encoder, specials, pat)?;
let tokens = bpe.encode_ordinary("Hello, world!");
let bytes = bpe.decode_bytes(&tokens);
assert_eq!(bytes, b"Hello, world!");
```

## How it works

riptoken ports tiktoken's algorithm to Rust and applies a small set of targeted
optimizations:

1. **Zero-allocation hash lookups.** The BPE merge loop queries the
   vocabulary thousands of times per input. We store the vocab as
   `FxHashMap<Vec<u8>, Rank>` and look up with `&[u8]` directly via
   `Vec<u8>: Borrow<[u8]>` — no per-lookup `Vec` allocation.
2. **Inlined initial min-scan.** The first pass that populates the `parts`
   vector also tracks the minimum rank, avoiding a redundant linear scan.
3. **Cache-aware merge update.** When the linear-scan path merges two
   adjacent parts, we update `parts[i-1]` and `parts[i]` **before** calling
   `Vec::remove(i+1)`. The remove shifts memory leftwards, evicting the cells
   we just read — doing the reads first keeps them hot.
4. **Heap path for long pieces.** Pieces ≥ 500 bytes use an `O(m log n)`
   min-heap with lazy invalidation and an intrusive doubly-linked list inside
   a flat `Vec<State>`. This avoids the `O(n²)` cliff of repeated
   `Vec::remove`.
5. **Whole-piece fast path.** Before running BPE on any regex-split piece,
   we check whether the piece is already a full vocabulary entry. For common
   English text, this hits over 99 % of the time and skips BPE entirely.
6. **Pre-compiled dense DFA.** Stock tiktoken patterns (`gpt2`,
   `r50k_base`, `p50k_base`, `cl100k_base`, `o200k_base`) are compiled
   into fully-materialized dense DFAs at build time via `regex-automata`.
   All states are computed upfront and embedded in the binary — zero lazy
   building at search time, eliminating the ~55 ms cold-start penalty the
   `regex` crate's lazy DFA incurs on large Unicode patterns. The
   `precompiled-dfa` Cargo feature (on by default) controls this; disable
   it for smaller binaries at the cost of a first-call warm-up.
7. **Immutable shared regex.** The pre-compiled dense DFA has no mutable
   state, so a single instance is shared across all threads — no
   per-thread clones needed. For non-stock patterns and the `fancy-regex`
   fallback, per-thread clones are still used to avoid mutex contention
   on internal scratch buffers.
8. **Parallel batch API.** `encode_ordinary_batch` / `encode_batch` fan
   out to rayon's global thread pool, so a batch of independent documents
   encodes in parallel. The Python bindings release the GIL for the full
   batch.
9. **GIL release.** Every Python-facing encode/decode call is wrapped in
   `py.detach(|| ...)` so Python threads can make real forward progress.

## API

### Python (`riptoken.Encoding`)

`get_encoding` / `encoding_for_model` return a `riptoken.Encoding`.
Hot-path methods run in the Rust core and release the GIL; every other
attribute forwards to the underlying `tiktoken.Encoding` via
`__getattr__`, so the full `tiktoken.Encoding` API is available.

| Method                                         | Returns                    |
| ---------------------------------------------- | -------------------------- |
| `encode_ordinary(text)`                        | `list[int]`                |
| `encode(text, allowed_special=None)`           | `list[int]`                |
| `encode_ordinary_batch(texts)`                 | `list[list[int]]`          |
| `encode_batch(texts, allowed_special=None)`    | `list[list[int]]`          |
| `decode(tokens)`                               | `str`                      |
| `decode_bytes(tokens)`                         | `bytes`                    |
| `n_vocab`, `eot_token`, `special_tokens_set`, … | forwarded to `tiktoken`    |

`allowed_special` accepts a `set[str]` or the sentinel `"all"`.

You can also construct a `riptoken.CoreBPE` directly from a `.tiktoken`
file via `load_tiktoken_bpe` if you want to avoid the `tiktoken`
dependency. `CoreBPE` exposes the same hot-path methods as `Encoding`
plus `encode_single_token`, `decode_single_token_bytes`, `n_vocab()`,
and `token_byte_values()`.

### Rust (`riptoken::CoreBPE`)

See [docs.rs/riptoken](https://docs.rs/riptoken) for full Rust API
documentation. The same methods are available, returning `Vec<Rank>`, `Vec<u8>`,
etc.

## Compatibility

riptoken reads the same `.tiktoken` vocabulary files as `tiktoken` and produces
identical token sequences. We run a CI parity check against `tiktoken` on every
commit across multiple corpora (English, code, multilingual, emoji, random
bytes).

If you find a string where riptoken produces different output from tiktoken,
that is a bug — please open an issue with the input and both outputs.

## Development

```bash
# Rust tests
cargo test

# Rust linting
cargo clippy --all-targets -- -D warnings

# Python extension + test suite
python -m venv .venv && source .venv/bin/activate
pip install -e .[test]
maturin develop --features python --release
pytest

# Benchmark
python scripts/bench.py
```

The Python test suite and benchmark use `riptoken.get_encoding("o200k_base")`
under the hood, which reads the vocabulary through `tiktoken` and its on-disk
cache at `~/.cache/tiktoken/`. No local `.tiktoken` file is required — the
first run downloads it automatically.

## Contributing

Issues and PRs welcome. Please include a benchmark or test case demonstrating
any performance or behavior change.

## License

MIT — see [LICENSE](LICENSE).

## Credits

riptoken is a re-implementation of the ideas in OpenAI's
[tiktoken](https://github.com/openai/tiktoken). The core BPE algorithm is due
to them; riptoken reuses vocabulary files in the `.tiktoken` format.
