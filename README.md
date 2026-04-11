# riptoken

**A fast BPE tokenizer for LLMs. Drop-in compatible with OpenAI's [`tiktoken`](https://github.com/openai/tiktoken), 1.8×–2.7× faster on realistic text.**

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
both libraries, outputs verified byte-identical.

| Corpus                 | Tokens  | riptoken (tok/s) | tiktoken (tok/s) | Speedup   |
| ---------------------- | ------- | ---------------: | ---------------: | --------- |
| English prose          |  40,001 |         8,557,965 |         3,158,249 | **2.71×** |
| Python source code     |  72,501 |         6,653,380 |         2,705,474 | **2.46×** |
| Rust source code       |  88,001 |         7,178,632 |         3,084,426 | **2.33×** |
| Multilingual + emoji   |  85,600 |         6,805,565 |         3,631,832 | **1.87×** |
| Random-ish bytes       | 120,000 |         8,840,042 |         4,365,504 | **2.02×** |

Reproduce with:

```bash
python scripts/bench.py
```

## Install

### Python

```bash
pip install riptoken
```

Pre-built wheels are published for CPython 3.9–3.13 on Linux (x86_64, aarch64),
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
import tiktoken  # only for the vocabulary loader path

# Load any tiktoken-format vocabulary.
ranks = riptoken.load_tiktoken_bpe("o200k_base.tiktoken")
special_tokens = {"<|endoftext|>": 199999, "<|endofprompt|>": 200018}
pat = (
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"""
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"""
    r"""\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)

enc = riptoken.CoreBPE(ranks, special_tokens, pat)

tokens = enc.encode_ordinary("Hello, world!")
assert enc.decode_bytes(tokens) == b"Hello, world!"

# With allowed special tokens
tokens = enc.encode("Hi <|endoftext|>", allowed_special={"<|endoftext|>"})
```

riptoken's `CoreBPE` has the same method surface as `tiktoken`'s core encoder.
In most existing codebases the migration is a single import change.

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
6. **Thread-local regex pool.** `fancy-regex` keeps mutable scratch state
   inside each `Regex`; concurrent `find_iter` calls contend on it. We
   pre-clone 128 regex instances and dispatch via a hashed thread id.
7. **GIL release.** Every Python-facing encode/decode call is wrapped in
   `py.detach(|| ...)` so Python threads can make real forward progress.

## API

### Python (`riptoken.CoreBPE`)

| Method                                         | Returns                    |
| ---------------------------------------------- | -------------------------- |
| `encode_ordinary(text)`                        | `list[int]`                |
| `encode(text, allowed_special)`                | `list[int]`                |
| `encode_single_token(piece: bytes)`            | `int`                      |
| `decode_bytes(tokens)`                         | `bytes`                    |
| `decode_single_token_bytes(token)`             | `bytes`                    |
| `n_vocab()`                                    | `int`                      |
| `token_byte_values()`                          | `list[bytes]`              |

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

You will need `o200k_base.tiktoken` in the project root to run benchmarks and
parity tests. Download it once from `tiktoken`'s public CDN or copy it out of
your local tiktoken cache.

## Contributing

Issues and PRs welcome. Please include a benchmark or test case demonstrating
any performance or behavior change.

## License

MIT — see [LICENSE](LICENSE).

## Credits

riptoken is a re-implementation of the ideas in OpenAI's
[tiktoken](https://github.com/openai/tiktoken). The core BPE algorithm is due
to them; riptoken reuses vocabulary files in the `.tiktoken` format.
