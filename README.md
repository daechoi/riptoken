# riptoken

**A fast BPE tokenizer for LLMs. Drop-in compatible with OpenAI's [`tiktoken`](https://github.com/openai/tiktoken), 2.4×–6.5× faster single-threaded and up to ~22× faster in parallel batch mode on wide machines.**

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

All benchmarks: `o200k_base` vocab, release builds, outputs verified
byte-identical.

### Apple Silicon (M-series), Python 3.13

#### Single-threaded

| Corpus                 |  Tokens | riptoken (tok/s) | tiktoken (tok/s) | Speedup   |
| ---------------------- | ------: | ---------------: | ---------------: | --------- |
| English prose          |  40,001 |       16,293,161 |        3,154,943 | **5.16×** |
| Python source code     |  72,501 |       16,294,984 |        2,702,277 | **6.03×** |
| Rust source code       |  88,001 |       18,431,909 |        3,050,831 | **6.04×** |
| Multilingual + emoji   |  85,600 |        9,467,460 |        3,634,679 | **2.60×** |
| Random-ish bytes       | 120,000 |       18,397,196 |        4,325,133 | **4.25×** |

#### Parallel batch (256 docs, rayon + GIL release)

| Corpus                 |     Tokens | riptoken (tok/s) | tiktoken (tok/s) | Speedup   |
| ---------------------- | ---------: | ---------------: | ---------------: | --------- |
| English prose          | 10,240,256 |       31,587,392 |       13,305,166 | **2.37×** |
| Python source code     | 18,560,256 |       40,671,916 |       11,512,182 | **3.53×** |
| Rust source code       | 22,528,256 |       47,642,701 |       13,573,170 | **3.51×** |
| Multilingual + emoji   | 21,913,600 |       29,356,351 |       15,049,425 | **1.95×** |
| Random-ish bytes       | 30,720,000 |       47,077,613 |       17,109,835 | **2.75×** |

### 32-core Sapphire Rapids (Linux), Python 3.12

#### Single-threaded

| Corpus                 |  Tokens | riptoken (tok/s) | tiktoken (tok/s) | Speedup   |
| ---------------------- | ------: | ---------------: | ---------------: | --------- |
| English prose          |  40,001 |       11,452,863 |        2,061,672 | **5.56×** |
| Python source code     |  72,501 |       11,287,635 |        1,796,196 | **6.28×** |
| Rust source code       |  88,001 |       13,252,883 |        2,037,556 | **6.50×** |
| Multilingual + emoji   |  85,600 |        5,544,698 |        2,309,829 | **2.40×** |
| Random-ish bytes       | 120,000 |       13,342,255 |        2,931,637 | **4.55×** |

#### Parallel batch (256 docs, rayon + GIL release)

| Corpus                 |     Tokens | riptoken (tok/s) | tiktoken (tok/s) | Speedup    |
| ---------------------- | ---------: | ---------------: | ---------------: | ---------- |
| English prose          | 10,240,256 |       27,306,398 |        7,451,179 | **3.66×**  |
| Python source code     | 18,560,256 |       31,016,281 |       10,601,406 | **2.93×**  |
| Rust source code       | 22,528,256 |       35,756,270 |        1,590,601 | **22.48×** |
| Multilingual + emoji   | 21,913,600 |       30,151,253 |        6,763,520 | **4.46×**  |
| Random-ish bytes       | 30,720,000 |       37,719,358 |       14,034,414 | **2.69×**  |

On the 32-core box, tiktoken's parallel batch path hits severe DFA cache pool
contention — most visible on Rust source code where its throughput drops to
1.6 M tok/s (below its own single-threaded speed). riptoken's pre-compiled
dense DFA has no mutable state, so it scales linearly with cores.

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
