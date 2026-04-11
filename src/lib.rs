//! # riptoken
//!
//! Fast BPE tokenizer for LLMs — a drop-in compatible, faster reimplementation
//! of OpenAI's [`tiktoken`](https://github.com/openai/tiktoken).
//!
//! ## Design
//!
//! riptoken is structured as three layers:
//! 1. A pure-Rust core ([`CoreBPE`]) that can be used directly from Rust.
//! 2. An optional PyO3 binding (enabled with the `python` feature).
//! 3. A Python wrapper package shipped on PyPI.
//!
//! The core BPE algorithm is a Rust port of tiktoken's with several
//! optimizations applied — see `README.md` for benchmarks and details.
//!
//! ## Example
//!
//! ```no_run
//! use riptoken::{CoreBPE, Rank};
//! use rustc_hash::FxHashMap;
//!
//! // In practice you would load `encoder` from an o200k_base / cl100k_base
//! // vocabulary file via `riptoken::load_tiktoken_bpe`.
//! let mut encoder: FxHashMap<Vec<u8>, Rank> = FxHashMap::default();
//! encoder.insert(b"h".to_vec(), 0);
//! encoder.insert(b"i".to_vec(), 1);
//! encoder.insert(b"hi".to_vec(), 2);
//!
//! let specials = FxHashMap::default();
//! let bpe = CoreBPE::new(encoder, specials, r"\w+").unwrap();
//!
//! let tokens = bpe.encode_ordinary("hi");
//! assert_eq!(tokens, vec![2]);
//! ```

use fancy_regex::Regex as FancyRegex;
use regex::Regex as FastRegex;
use rustc_hash::{FxHashMap as HashMap, FxHasher};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Integer rank of a token in the BPE vocabulary.
///
/// Lower ranks are merged first. [`Rank::MAX`] is reserved as a sentinel meaning
/// "this byte span is not in the vocabulary".
pub type Rank = u32;

/// Number of thread-local regex clones. Must be a power of two for cheap
/// modulo via bitmask, but we use plain `%` since this is off the hot path.
const MAX_NUM_THREADS: usize = 128;

/// Pieces shorter than this use the `Vec`-based merge path with a linear-scan
/// min-find. Pieces at or above this length use a heap-based path.
///
/// Short pieces benefit from cache locality; long pieces avoid the `O(m·n)`
/// cliff of the linear scan. The threshold matches tiktoken's.
const LARGE_PIECE_THRESHOLD: usize = 500;

thread_local! {
    static THREAD_INDEX: usize = {
        let mut h = FxHasher::default();
        std::thread::current().id().hash(&mut h);
        (h.finish() as usize) % MAX_NUM_THREADS
    };
}

#[inline]
fn thread_index() -> usize {
    THREAD_INDEX.with(|&i| i)
}

/// Errors produced when constructing a [`CoreBPE`].
#[derive(Debug)]
pub enum BuildError {
    /// The regex pattern failed to compile in both the fast and the fallback
    /// engines. Contains the fallback engine's error message.
    InvalidRegex(String),
    /// The encoder and decoder had mismatched sizes (usually means duplicate
    /// ranks or bytes in the input vocabulary).
    VocabularyMismatch,
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::InvalidRegex(e) => write!(f, "invalid regex pattern: {e}"),
            BuildError::VocabularyMismatch => write!(
                f,
                "vocabulary has duplicate entries (encoder/decoder size mismatch)"
            ),
        }
    }
}

impl std::error::Error for BuildError {}

impl From<fancy_regex::Error> for BuildError {
    fn from(e: fancy_regex::Error) -> Self {
        BuildError::InvalidRegex(e.to_string())
    }
}

/// Errors produced during decoding.
#[derive(Debug)]
pub enum DecodeError {
    /// A token ID was not in the vocabulary.
    InvalidToken(Rank),
    /// The decoded bytes were not valid UTF-8 (only raised by [`CoreBPE::decode`]).
    InvalidUtf8,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::InvalidToken(t) => write!(f, "invalid token id: {t}"),
            DecodeError::InvalidUtf8 => write!(f, "decoded bytes are not valid UTF-8"),
        }
    }
}

impl std::error::Error for DecodeError {}

// --- Split engine dispatch ---------------------------------------------------

/// How to apply the whitespace-shrink rule in the fast-path engine.
///
/// The rule reproduces tiktoken's `\s+(?!\S)` alternative — "match a
/// whitespace run but don't swallow the last char if a non-whitespace char
/// follows, so it can attach to the following word via the word
/// alternative's optional non-word prefix".
#[derive(Clone, Copy, PartialEq, Debug)]
enum ShrinkMode {
    /// No `\s+(?!\S)` alternative in the original pattern (e.g. a toy
    /// pattern like `\w+|\s+`). No shrinking is applied.
    None,
    /// o200k / cl100k style: the pattern has a separate `\s*[\r\n]+`
    /// alternative which takes priority and matches any whitespace run
    /// containing a newline as its own piece. The shrink rule should only
    /// fire on plain whitespace runs — newline runs are emitted whole.
    PlainOnly,
    /// gpt2 / r50k / p50k style: no separate newline alternative, so
    /// `\s+(?!\S)` absorbs every whitespace run, newlines included. The
    /// shrink rule should fire on any whitespace run.
    Unified,
}

/// Try to rewrite a tiktoken-style pattern into a form that the SIMD-accelerated
/// `regex` crate can compile.
///
/// The only transformation applied is stripping the `\s+(?!\S)` alternative,
/// which appears in every stock tiktoken encoding. Its purpose — "leave the
/// last whitespace of a run for the next word's leading-space prefix" — is
/// replicated in Rust code by [`SplitEngine::find_pieces`] when running on
/// the fast path, using the returned [`ShrinkMode`] to select which
/// whitespace runs the shrink should fire on.
///
/// Returns `Some((regex, mode))` if the pattern compiles successfully with
/// the `regex` crate after transformation, or `None` if the pattern contains
/// other lookarounds (or features) that regex can't handle — in which case
/// the caller falls back to `fancy-regex`.
fn try_transform_for_fast_regex(pattern: &str) -> Option<(FastRegex, ShrinkMode)> {
    // 1. Collapse the whitespace-run tail and classify the pattern family.
    //    Every stock tiktoken pattern ends with one of two shapes:
    //       ...|\s*[\r\n]+|\s+(?!\S)|\s+   (o200k/cl100k-style)
    //       ...|\s+$|\s+(?!\S)|\s          (gpt2/r50k/p50k-style)
    //    We collapse the `\s+(?!\S)|...` pair into a single greedy `\s+`
    //    and reproduce the "leave last char" semantics in `find_pieces`
    //    below. The `ShrinkMode` distinguishes the two families so the
    //    shrink rule can be applied correctly: in the o200k family the
    //    separate `\s*[\r\n]+` alternative already handles runs that
    //    contain a newline, so shrink must only fire on plain whitespace
    //    runs; in the gpt2 family there is no such alternative and shrink
    //    must fire on any whitespace run.
    //
    //    Order matters: the more specific `\s+` form must be tried before
    //    the bare `\s` form, otherwise we'd leave a trailing `+` behind.
    let shrink_mode = if pattern.contains(r"\s+(?!\S)|\s+") {
        ShrinkMode::PlainOnly
    } else if pattern.contains(r"\s+(?!\S)|\s") {
        ShrinkMode::Unified
    } else {
        ShrinkMode::None
    };
    let mut stripped = pattern.replace(r"\s+(?!\S)|\s+", r"\s+");
    stripped = stripped.replace(r"\s+(?!\S)|\s", r"\s+");
    // 2. Reject anything that still contains a lookaround. We intentionally
    //    don't try to be clever — any residual `(?=`, `(?!`, `(?<=`, `(?<!`
    //    means we fall back to fancy-regex.
    if stripped.contains("(?=")
        || stripped.contains("(?!")
        || stripped.contains("(?<=")
        || stripped.contains("(?<!")
    {
        return None;
    }
    // 3. Convert possessive quantifiers to greedy. tiktoken's newer
    //    cl100k_base / p50k_base / o200k_base patterns use `?+`, `++`,
    //    `*+`, `{n,m}+` as backtracking-engine speed hints ("don't retry
    //    this match"). The regex crate's parser silently accepts the
    //    syntax but its DFA interprets it differently, producing wrong
    //    matches.
    //
    //    In a DFA engine possessive markers are semantically unnecessary:
    //    the DFA is already linear-time and has no backtracking to
    //    disable. And in every tiktoken pattern the possessive and greedy
    //    matches are identical by construction — the alternatives are
    //    disjoint enough that backtracking would never change the result
    //    — so converting possessive → greedy is safe.
    //
    //    Simple string replace handles `?+`, `++`, `*+` (tiktoken patterns
    //    never put these chars inside `[...]` or escape them). `{n,m}+`
    //    needs care: `\p{L}+` also contains the literal sequence `}+`,
    //    but the `+` there is a greedy quantifier on the class, not a
    //    possessive marker — so we use a precise regex to match only
    //    ranges `{digits[,digits]}+`.
    stripped = stripped
        .replace("?+", "?")
        .replace("++", "+")
        .replace("*+", "*");
    let range_possessive = FastRegex::new(r"(\{\d+(?:,\d*)?\})\+").ok()?;
    let stripped = range_possessive.replace_all(&stripped, "$1").into_owned();
    // 4. Try to compile with the regex crate. Unicode property classes like
    //    `\p{L}`, `\p{N}` are supported by default.
    let regex = FastRegex::new(&stripped).ok()?;
    Some((regex, shrink_mode))
}

/// True if `s` consists entirely of whitespace with no `\n` or `\r`.
///
/// Used under [`ShrinkMode::PlainOnly`] (o200k/cl100k family): the whitespace
/// shrink rule must NOT fire on matches that contain a newline, because those
/// come from the `\s*[\r\n]+` alternative in the pattern, which takes priority
/// in the alternation and is emitted whole. Checking "contains no newline" is
/// a reliable way to distinguish a `\s+` match from a `\s*[\r\n]+` match
/// without needing capture groups.
#[inline]
fn is_plain_whitespace_run(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_whitespace() && c != '\n' && c != '\r')
}

/// True if `s` consists entirely of whitespace (any kind, including `\n`/`\r`).
///
/// Used under [`ShrinkMode::Unified`] (gpt2/r50k/p50k family): there is no
/// separate newline alternative in those patterns, so `\s+(?!\S)` absorbs
/// whitespace runs containing newlines and the shrink rule must fire on
/// them too.
#[inline]
fn is_whitespace_run(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_whitespace())
}

/// True if the next char in `s` (starting at `pos`) is a non-whitespace char.
/// Returns `false` if `pos` is at end-of-string.
#[inline]
fn next_char_is_non_whitespace(text: &str, pos: usize) -> bool {
    match text[pos..].chars().next() {
        Some(c) => !c.is_whitespace(),
        None => false,
    }
}

/// The pattern-matching engine used to split text into pieces before BPE.
///
/// Both the fast and fancy engines hold per-thread-slot clones. The regex
/// crate's `Regex` is `Send + Sync` but internally uses a `Pool<Cache>` of
/// DFA scratch buffers guarded by a mutex — under high thread counts that
/// pool becomes a contention point, so we give each thread its own instance.
/// The clones are cheap (regex state is Arc-shared internally).
enum SplitEngine {
    /// SIMD-accelerated, finite-automaton regex. One clone per thread slot,
    /// plus a [`ShrinkMode`] selecting how to emulate `\s+(?!\S)` in Rust.
    Fast {
        clones: Vec<FastRegex>,
        shrink_mode: ShrinkMode,
    },
    /// Backtracking regex with lookaround support. One clone per thread slot.
    Fancy(Vec<FancyRegex>),
}

impl SplitEngine {
    /// Build an engine for `pattern`. Attempts the fast path first and falls
    /// back to fancy-regex if the pattern can't be transformed.
    fn new(pattern: &str) -> Result<Self, BuildError> {
        if let Some((fast, shrink_mode)) = try_transform_for_fast_regex(pattern) {
            let clones: Vec<FastRegex> = (0..MAX_NUM_THREADS).map(|_| fast.clone()).collect();
            return Ok(SplitEngine::Fast {
                clones,
                shrink_mode,
            });
        }
        let fancy = FancyRegex::new(pattern)?;
        let clones: Vec<FancyRegex> = (0..MAX_NUM_THREADS).map(|_| fancy.clone()).collect();
        Ok(SplitEngine::Fancy(clones))
    }

    /// True if this engine took the fast path. Useful for tests and stats.
    #[cfg(test)]
    fn is_fast(&self) -> bool {
        matches!(self, SplitEngine::Fast { .. })
    }

    /// Iterate the pieces of `text`, invoking `f` with each piece as a `&str`.
    ///
    /// On the fast path, applies the `\s+(?!\S)` whitespace-shrink rule in
    /// Rust so that output matches tiktoken exactly.
    #[inline]
    fn find_pieces<F: FnMut(&str)>(&self, text: &str, mut f: F) {
        match self {
            SplitEngine::Fast {
                clones,
                shrink_mode,
            } => {
                let regex = &clones[thread_index()];
                let mut pos = 0;
                while pos < text.len() {
                    let m = match regex.find_at(text, pos) {
                        Some(m) => m,
                        None => break,
                    };
                    // With well-formed tiktoken patterns every byte matches
                    // some alternative, so the match should start exactly at
                    // `pos`. If there's a gap, something unexpected happened
                    // — skip over it to make progress.
                    if m.start() > pos {
                        pos = m.start();
                    }
                    let start = m.start();
                    let mut end = m.end();
                    let piece = &text[start..end];

                    // Whitespace post-processing: `\s+(?!\S)` means "match
                    // whitespace run but don't swallow the last whitespace
                    // if a non-whitespace char follows". Replicate by
                    // shrinking by one char in exactly that situation.
                    //
                    // Under `PlainOnly` (o200k/cl100k), the shrink fires
                    // only on runs without newlines — matches containing
                    // `\n`/`\r` came from the separate `\s*[\r\n]+`
                    // alternative and are emitted whole. Under `Unified`
                    // (gpt2/r50k/p50k), there is no such alternative, so
                    // the shrink fires on every whitespace run.
                    let should_shrink = match shrink_mode {
                        ShrinkMode::None => false,
                        ShrinkMode::PlainOnly => is_plain_whitespace_run(piece),
                        ShrinkMode::Unified => is_whitespace_run(piece),
                    };
                    if should_shrink && end < text.len() && next_char_is_non_whitespace(text, end) {
                        // Find the byte offset of the last char in the piece.
                        if let Some((last_i, _)) = piece.char_indices().next_back() {
                            // Only shrink if the piece has more than one char.
                            if last_i > 0 {
                                end = start + last_i;
                            } else {
                                // Single-char whitespace followed by a word —
                                // this shouldn't normally happen with tiktoken
                                // patterns (the word alternative would have
                                // consumed this position first), but if it
                                // does, keep the full match to avoid an
                                // empty piece.
                            }
                        }
                    }

                    f(&text[start..end]);
                    // Guard against pathological zero-width matches.
                    if end == pos {
                        pos += 1;
                    } else {
                        pos = end;
                    }
                }
            }
            SplitEngine::Fancy(clones) => {
                let regex = &clones[thread_index()];
                for mat in regex.find_iter(text) {
                    match mat {
                        Ok(m) => f(m.as_str()),
                        Err(_) => continue,
                    }
                }
            }
        }
    }
}

/// The core BPE encoder/decoder.
///
/// A `CoreBPE` owns the vocabulary, the compiled regex used to split text into
/// pieces, and a pool of per-thread regex clones. It is `Send + Sync` and
/// designed to be constructed once and shared (e.g. behind an `Arc`).
#[cfg_attr(feature = "python", pyclass(module = "riptoken._riptoken"))]
pub struct CoreBPE {
    /// Byte-sequence → rank. Lookup key for encoding.
    encoder: HashMap<Vec<u8>, Rank>,
    /// Rank → byte-sequence. Lookup key for decoding.
    decoder: HashMap<Rank, Vec<u8>>,
    /// Special token string → rank.
    special_tokens_encoder: HashMap<String, Rank>,
    /// Rank → special token bytes.
    special_tokens_decoder: HashMap<Rank, Vec<u8>>,
    /// Fast or fancy engine for splitting text into pieces.
    split_engine: SplitEngine,
    /// Thread-local clones of the special-token regex. Empty if there are no
    /// special tokens. Special-token patterns are always literal alternations,
    /// so `fancy_regex` is fine here (and needed only for its `find_from_pos`
    /// API which `regex` crate also provides but we keep symmetric).
    special_regex_tls: Vec<FancyRegex>,
    /// Sorted vocabulary bytes, useful for prefix queries.
    sorted_token_bytes: Vec<Vec<u8>>,
}

// --- Core BPE merge algorithm -------------------------------------------------

/// Compute the rank of a byte pair directly from a slice — no allocation.
///
/// `HashMap<Vec<u8>, Rank>::get` accepts any `Q: ?Sized` where
/// `Vec<u8>: Borrow<Q>`, and `Vec<u8>` implements `Borrow<[u8]>`, so this
/// avoids the `.to_vec()` allocation the naive implementation does.
#[inline(always)]
fn rank_of(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Rank {
    ranks.get(piece).copied().unwrap_or(Rank::MAX)
}

/// `O(m·n)` linear-scan BPE merge for short pieces.
///
/// Returns a list of `(start_position, rank)` such that consecutive windows of
/// two elements give the start/end of each final token:
///
/// ```text
/// parts: [(0, _), (2, _), (5, _), (5, MAX)]
/// tokens: piece[0..2], piece[2..5]
/// ```
#[inline]
fn byte_pair_merge(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<(usize, Rank)> {
    // Fast path: trivially short pieces.
    if piece.len() < 2 {
        return vec![(0, Rank::MAX), (piece.len(), Rank::MAX)];
    }

    let mut parts: Vec<(usize, Rank)> = Vec::with_capacity(piece.len() + 1);

    // Populate initial byte pairs AND find the initial minimum in a single pass.
    let mut min_rank: (Rank, usize) = (Rank::MAX, usize::MAX);
    for i in 0..piece.len() - 1 {
        let rank = rank_of(ranks, &piece[i..i + 2]);
        if rank < min_rank.0 {
            min_rank = (rank, i);
        }
        parts.push((i, rank));
    }
    parts.push((piece.len() - 1, Rank::MAX));
    parts.push((piece.len(), Rank::MAX));

    // Returns the rank of the merge *starting* at `parts[i]`, using the
    // pre-remove parts vector — so it looks 3 ahead to see past the soon-to-be-
    // removed `parts[i+1]`.
    let get_rank = |parts: &[(usize, Rank)], i: usize| -> Rank {
        if i + 3 < parts.len() {
            rank_of(ranks, &piece[parts[i].0..parts[i + 3].0])
        } else {
            Rank::MAX
        }
    };

    while min_rank.0 != Rank::MAX {
        let i = min_rank.1;

        // Update parts[i-1] and parts[i] BEFORE the remove. `parts.remove`
        // shifts everything from `i+2` leftward, evicting cache lines. Reading
        // the hot neighbours first keeps the accesses on hot memory.
        if i > 0 {
            parts[i - 1].1 = get_rank(&parts, i - 1);
        }
        parts[i].1 = get_rank(&parts, i);
        parts.remove(i + 1);

        // Rescan for new minimum. Excludes the two trailing sentinels.
        min_rank = (Rank::MAX, usize::MAX);
        for (j, &(_, rank)) in parts[..parts.len() - 2].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, j);
            }
        }
    }

    parts
}

/// Apply BPE to a single piece that is NOT already a full vocabulary entry.
#[inline]
fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<Rank> {
    // Single byte fast path.
    if piece.len() == 1 {
        // Every standard BPE vocab includes all 256 single bytes.
        return vec![*ranks.get(piece).expect("byte fallback")];
    }

    if piece.len() < LARGE_PIECE_THRESHOLD {
        let positions = byte_pair_merge(ranks, piece);
        // `positions` has n+1 entries and yields n tokens (windows of 2).
        let mut out: Vec<Rank> = Vec::with_capacity(positions.len() - 1);
        out.extend(
            positions
                .windows(2)
                .map(|w| rank_of(ranks, &piece[w[0].0..w[1].0])),
        );
        out
    } else {
        byte_pair_merge_large(ranks, piece)
    }
}

// --- Heap-based merge for long pieces ----------------------------------------

/// `O(m log n)` heap-based BPE merge for long pieces, with an intrusive
/// doubly-linked list embedded in a flat `Vec<State>` to avoid the `O(n)`
/// shifts of `Vec::remove`.
///
/// Uses lazy invalidation: we never remove stale entries from the heap —
/// instead we bump a generation counter on the state and skip heap entries
/// whose stored rank no longer matches.
fn byte_pair_merge_large(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<Rank> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    // state[start].end is the exclusive end position of the current token
    // starting at `start`. state[start].prev is the start position of the
    // previous token (or usize::MAX if none). `cur_rank` tracks which rank
    // this state represents in the heap; stale heap entries are discarded.
    #[derive(Clone)]
    struct State {
        prev: usize,
        end: usize,
        cur_rank: Rank,
    }

    let n = piece.len();
    let mut state: Vec<State> = (0..n)
        .map(|i| State {
            prev: if i == 0 { usize::MAX } else { i - 1 },
            end: i + 1,
            cur_rank: 0,
        })
        .collect();

    // Heap entry: (Reverse(rank), start). Reverse so BinaryHeap is a min-heap.
    let mut heap: BinaryHeap<(Reverse<Rank>, usize)> = BinaryHeap::with_capacity(n);

    // Seed with all initial pair ranks.
    for i in 0..n.saturating_sub(1) {
        let rank = rank_of(ranks, &piece[i..state[i + 1].end]);
        state[i].cur_rank = rank;
        if rank != Rank::MAX {
            heap.push((Reverse(rank), i));
        }
    }

    while let Some((Reverse(rank), start)) = heap.pop() {
        // Lazy invalidation: skip if this entry is stale.
        if state[start].cur_rank != rank || rank == Rank::MAX {
            continue;
        }

        // Absorb the next token into [start]: extend `end`, unlink [right].
        let right = state[start].end;
        if right >= n {
            continue;
        }
        let new_end = state[right].end;
        state[start].end = new_end;

        // Patch the "prev" of whatever comes after the absorbed one.
        if new_end < n {
            state[new_end].prev = start;
        }

        // Invalidate the old right entry so future stale heap pops skip it.
        state[right].cur_rank = Rank::MAX;

        // Recompute rank of [start] (now a longer span).
        let next_end = state[start].end;
        if next_end < n {
            let new_rank = rank_of(ranks, &piece[start..state[next_end].end]);
            state[start].cur_rank = new_rank;
            if new_rank != Rank::MAX {
                heap.push((Reverse(new_rank), start));
            }
        } else {
            state[start].cur_rank = Rank::MAX;
        }

        // Recompute rank of [prev] — it now points to a longer span too.
        let prev = state[start].prev;
        if prev != usize::MAX {
            let prev_next_end = state[prev].end; // this is still `start` unchanged
            debug_assert_eq!(prev_next_end, start);
            let span_end = state[start].end;
            let new_rank = rank_of(ranks, &piece[prev..span_end]);
            state[prev].cur_rank = new_rank;
            if new_rank != Rank::MAX {
                heap.push((Reverse(new_rank), prev));
            }
        }
    }

    // Walk the linked list from start to collect final tokens.
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < n {
        let end = state[i].end;
        tokens.push(rank_of(ranks, &piece[i..end]));
        i = end;
    }
    tokens
}

// --- Special-token regex helpers ----------------------------------------------

/// Build a regex that matches any of the given special token strings.
///
/// Returns `None` if `specials` is empty — callers should then skip the
/// special-token scan entirely.
fn build_special_regex(specials: &HashMap<String, Rank>) -> Result<Option<FancyRegex>, BuildError> {
    if specials.is_empty() {
        return Ok(None);
    }
    // Escape each special token literally and join with `|`.
    let parts: Vec<String> = specials
        .keys()
        .map(|s| fancy_regex::escape(s).into_owned())
        .collect();
    let pattern = parts.join("|");
    Ok(Some(FancyRegex::new(&pattern)?))
}

// --- CoreBPE public API ------------------------------------------------------

impl CoreBPE {
    /// Construct a new `CoreBPE`.
    ///
    /// - `encoder`: byte-sequence → rank map (the vocabulary).
    /// - `special_tokens_encoder`: special-token string → rank.
    /// - `pattern`: a regex string used to split text into pieces before BPE.
    ///
    /// Returns a [`BuildError`] if the regex is invalid or the vocabulary has
    /// duplicate entries.
    pub fn new(
        encoder: HashMap<Vec<u8>, Rank>,
        special_tokens_encoder: HashMap<String, Rank>,
        pattern: &str,
    ) -> Result<Self, BuildError> {
        let split_engine = SplitEngine::new(pattern)?;
        let decoder: HashMap<Rank, Vec<u8>> =
            encoder.iter().map(|(k, v)| (*v, k.clone())).collect();
        if decoder.len() != encoder.len() {
            return Err(BuildError::VocabularyMismatch);
        }
        let special_tokens_decoder: HashMap<Rank, Vec<u8>> = special_tokens_encoder
            .iter()
            .map(|(k, v)| (*v, k.as_bytes().to_vec()))
            .collect();

        let special_regex = build_special_regex(&special_tokens_encoder)?;
        let special_regex_tls: Vec<FancyRegex> = match special_regex {
            Some(r) => (0..MAX_NUM_THREADS).map(|_| r.clone()).collect(),
            None => Vec::new(),
        };

        let mut sorted_token_bytes: Vec<Vec<u8>> = encoder.keys().cloned().collect();
        sorted_token_bytes.sort();

        Ok(CoreBPE {
            encoder,
            decoder,
            special_tokens_encoder,
            special_tokens_decoder,
            split_engine,
            special_regex_tls,
            sorted_token_bytes,
        })
    }

    /// Size of the vocabulary, defined as `max_rank + 1` across all tokens
    /// (ordinary + special). This matches tiktoken's `n_vocab` semantics, so
    /// vocabularies with reserved rank gaps (like `o200k_base`) report the
    /// "reach" of the id space rather than the count of live tokens.
    pub fn n_vocab(&self) -> usize {
        let max_ordinary = self.encoder.values().copied().max().unwrap_or(0);
        let max_special = self
            .special_tokens_encoder
            .values()
            .copied()
            .max()
            .unwrap_or(0);
        max_ordinary.max(max_special) as usize + 1
    }

    /// A sorted list of every (non-special) token's bytes.
    pub fn token_byte_values(&self) -> &[Vec<u8>] {
        &self.sorted_token_bytes
    }

    #[inline]
    fn tl_special_regex(&self) -> Option<&FancyRegex> {
        self.special_regex_tls.get(thread_index())
    }

    /// Emit the tokens produced by BPE-encoding one regex-split piece.
    #[inline]
    fn emit_piece(&self, piece: &[u8], out: &mut Vec<Rank>) {
        // Whole-piece fast path: most regex splits are already full tokens.
        if let Some(&token) = self.encoder.get(piece) {
            out.push(token);
            return;
        }
        out.extend(byte_pair_encode(piece, &self.encoder));
    }

    /// Encode ordinary text, ignoring special tokens entirely.
    ///
    /// Any special-token substrings in the input will be tokenized as regular
    /// text (this matches tiktoken's behavior).
    pub fn encode_ordinary(&self, text: &str) -> Vec<Rank> {
        // Rough pre-allocation: ~4 bytes per token is the average for English
        // + code at the o200k_base vocab size. Overshoot a little to avoid the
        // last realloc and undershoot hurts less than it helps (reallocs are
        // cheap for the final doubling).
        let mut ret = Vec::with_capacity(text.len() / 3 + 1);
        self.split_engine.find_pieces(text, |piece| {
            self.emit_piece(piece.as_bytes(), &mut ret);
        });
        ret
    }

    /// Encode many ordinary texts in parallel using rayon.
    ///
    /// Each text is tokenized on a rayon worker thread; the returned vector
    /// preserves input order. Uses the global rayon pool, so the level of
    /// parallelism is controlled by `RAYON_NUM_THREADS` or
    /// `rayon::ThreadPoolBuilder`.
    pub fn encode_ordinary_batch(&self, texts: &[&str]) -> Vec<Vec<Rank>> {
        use rayon::prelude::*;
        texts.par_iter().map(|t| self.encode_ordinary(t)).collect()
    }

    /// Encode many texts in parallel using rayon, honoring special tokens.
    ///
    /// See [`CoreBPE::encode`] for the special-token semantics.
    pub fn encode_batch(&self, texts: &[&str], allowed_special: &HashSet<&str>) -> Vec<Vec<Rank>> {
        use rayon::prelude::*;
        texts
            .par_iter()
            .map(|t| self.encode(t, allowed_special))
            .collect()
    }

    /// Encode text, allowing a specific set of special tokens.
    ///
    /// Special tokens in `allowed_special` are emitted as their assigned ranks.
    /// Special tokens NOT in `allowed_special` are tokenized as ordinary text.
    pub fn encode(&self, text: &str, allowed_special: &HashSet<&str>) -> Vec<Rank> {
        let special_regex = match self.tl_special_regex() {
            Some(r) => r,
            // No special tokens registered at all — just do ordinary encoding.
            None => return self.encode_ordinary(text),
        };

        let mut ret = Vec::new();
        let mut start = 0usize;
        loop {
            // Find the next *allowed* special token.
            let mut next_special: Option<(usize, usize)> = None;
            let mut search_from = start;
            while search_from <= text.len() {
                match special_regex.find_from_pos(text, search_from) {
                    Ok(Some(m)) => {
                        if allowed_special.contains(&text[m.start()..m.end()]) {
                            next_special = Some((m.start(), m.end()));
                            break;
                        }
                        // Skip this match — move one char forward.
                        search_from = m.start() + 1;
                    }
                    _ => break,
                }
            }

            let end = next_special.map_or(text.len(), |(s, _)| s);

            // Encode the ordinary text between [start, end).
            self.split_engine.find_pieces(&text[start..end], |piece| {
                self.emit_piece(piece.as_bytes(), &mut ret);
            });

            // Emit the special token (if any) and advance.
            match next_special {
                Some((s, e)) => {
                    let piece = &text[s..e];
                    if let Some(&tok) = self.special_tokens_encoder.get(piece) {
                        ret.push(tok);
                    }
                    start = e;
                }
                None => break,
            }
        }
        ret
    }

    /// Look up a single token by its byte sequence.
    pub fn encode_single_token(&self, piece: &[u8]) -> Option<Rank> {
        if let Some(&r) = self.encoder.get(piece) {
            return Some(r);
        }
        if let Ok(s) = std::str::from_utf8(piece) {
            if let Some(&r) = self.special_tokens_encoder.get(s) {
                return Some(r);
            }
        }
        None
    }

    /// Decode a sequence of tokens into the underlying bytes.
    ///
    /// Unknown token IDs are silently skipped — this matches tiktoken's
    /// `decode_bytes` behavior. Use [`CoreBPE::decode_single_token_bytes`] if
    /// you need strict validation.
    pub fn decode_bytes(&self, tokens: &[Rank]) -> Vec<u8> {
        let mut ret = Vec::with_capacity(tokens.len() * 2);
        for &token in tokens {
            if let Some(bytes) = self.decoder.get(&token) {
                ret.extend_from_slice(bytes);
            } else if let Some(bytes) = self.special_tokens_decoder.get(&token) {
                ret.extend_from_slice(bytes);
            }
        }
        ret
    }

    /// Decode tokens as a UTF-8 string.
    ///
    /// Returns [`DecodeError::InvalidUtf8`] if the concatenated bytes are not
    /// valid UTF-8. This can happen mid-stream when a multi-byte character
    /// spans a token boundary; prefer [`CoreBPE::decode_bytes`] for streaming.
    pub fn decode(&self, tokens: &[Rank]) -> Result<String, DecodeError> {
        String::from_utf8(self.decode_bytes(tokens)).map_err(|_| DecodeError::InvalidUtf8)
    }

    /// Look up the bytes of a single token. Returns an error if the token is
    /// not in the vocabulary.
    pub fn decode_single_token_bytes(&self, token: Rank) -> Result<Vec<u8>, DecodeError> {
        if let Some(bytes) = self.decoder.get(&token) {
            return Ok(bytes.clone());
        }
        if let Some(bytes) = self.special_tokens_decoder.get(&token) {
            return Ok(bytes.clone());
        }
        Err(DecodeError::InvalidToken(token))
    }
}

// --- PyO3 bindings ------------------------------------------------------------

#[cfg(feature = "python")]
#[pymethods]
impl CoreBPE {
    #[new]
    #[pyo3(signature = (encoder, special_tokens_encoder, pattern))]
    fn py_new(
        encoder: HashMap<Vec<u8>, Rank>,
        special_tokens_encoder: HashMap<String, Rank>,
        pattern: &str,
    ) -> PyResult<Self> {
        Self::new(encoder, special_tokens_encoder, pattern)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[pyo3(name = "encode_ordinary")]
    fn py_encode_ordinary(&self, py: Python<'_>, text: &str) -> Vec<Rank> {
        py.detach(|| self.encode_ordinary(text))
    }

    #[pyo3(name = "encode")]
    fn py_encode(&self, py: Python<'_>, text: &str, allowed_special: HashSet<String>) -> Vec<Rank> {
        py.detach(|| {
            let allowed_refs: HashSet<&str> = allowed_special.iter().map(|s| s.as_str()).collect();
            self.encode(text, &allowed_refs)
        })
    }

    #[pyo3(name = "encode_ordinary_batch")]
    fn py_encode_ordinary_batch(&self, py: Python<'_>, texts: Vec<String>) -> Vec<Vec<Rank>> {
        py.detach(|| {
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            self.encode_ordinary_batch(&refs)
        })
    }

    #[pyo3(name = "encode_batch")]
    fn py_encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        allowed_special: HashSet<String>,
    ) -> Vec<Vec<Rank>> {
        py.detach(|| {
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let allowed_refs: HashSet<&str> = allowed_special.iter().map(|s| s.as_str()).collect();
            self.encode_batch(&refs, &allowed_refs)
        })
    }

    #[pyo3(name = "encode_single_token")]
    fn py_encode_single_token(&self, piece: &[u8]) -> PyResult<Rank> {
        self.encode_single_token(piece)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("token not found"))
    }

    #[pyo3(name = "decode_bytes")]
    fn py_decode_bytes<'py>(
        &self,
        py: Python<'py>,
        tokens: Vec<Rank>,
    ) -> pyo3::Bound<'py, pyo3::types::PyBytes> {
        let bytes = py.detach(|| self.decode_bytes(&tokens));
        pyo3::types::PyBytes::new(py, &bytes)
    }

    #[pyo3(name = "decode_single_token_bytes")]
    fn py_decode_single_token_bytes<'py>(
        &self,
        py: Python<'py>,
        token: Rank,
    ) -> PyResult<pyo3::Bound<'py, pyo3::types::PyBytes>> {
        let bytes = self
            .decode_single_token_bytes(token)
            .map_err(|e| pyo3::exceptions::PyKeyError::new_err(e.to_string()))?;
        Ok(pyo3::types::PyBytes::new(py, &bytes))
    }

    #[pyo3(name = "n_vocab")]
    fn py_n_vocab(&self) -> usize {
        self.n_vocab()
    }

    #[pyo3(name = "token_byte_values")]
    fn py_token_byte_values<'py>(
        &self,
        py: Python<'py>,
    ) -> Vec<pyo3::Bound<'py, pyo3::types::PyBytes>> {
        self.sorted_token_bytes
            .iter()
            .map(|b| pyo3::types::PyBytes::new(py, b))
            .collect()
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn _riptoken(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<CoreBPE>()?;
    Ok(())
}

// --- Unit tests ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_bpe() -> CoreBPE {
        let mut encoder = HashMap::default();
        for (i, b) in b"helo ".iter().enumerate() {
            encoder.insert(vec![*b], i as Rank);
        }
        encoder.insert(b"he".to_vec(), 100);
        encoder.insert(b"ll".to_vec(), 101);
        CoreBPE::new(encoder, HashMap::default(), r"\w+| ").unwrap()
    }

    #[test]
    fn merge_empty_piece() {
        let ranks: HashMap<Vec<u8>, Rank> = HashMap::default();
        let result = byte_pair_merge(&ranks, b"");
        assert_eq!(result, vec![(0, Rank::MAX), (0, Rank::MAX)]);
    }

    #[test]
    fn merge_single_byte() {
        let ranks: HashMap<Vec<u8>, Rank> = HashMap::default();
        let result = byte_pair_merge(&ranks, b"a");
        assert_eq!(result, vec![(0, Rank::MAX), (1, Rank::MAX)]);
    }

    #[test]
    fn merge_two_byte_exact_match() {
        let mut ranks = HashMap::default();
        ranks.insert(b"ab".to_vec(), 5);
        let result = byte_pair_merge(&ranks, b"ab");
        let positions: Vec<usize> = result.iter().map(|&(p, _)| p).collect();
        assert_eq!(positions, vec![0, 2]);
    }

    #[test]
    fn merge_no_vocab_matches() {
        let ranks: HashMap<Vec<u8>, Rank> = HashMap::default();
        let result = byte_pair_merge(&ranks, b"abcd");
        let positions: Vec<usize> = result.iter().map(|&(p, _)| p).collect();
        // No merges possible — each byte is its own token.
        assert_eq!(positions, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn merge_cascade() {
        let mut ranks = HashMap::default();
        ranks.insert(b"ab".to_vec(), 0);
        ranks.insert(b"cd".to_vec(), 1);
        let result = byte_pair_merge(&ranks, b"abcd");
        let positions: Vec<usize> = result.iter().map(|&(p, _)| p).collect();
        assert_eq!(positions, vec![0, 2, 4]);
    }

    #[test]
    fn encode_toy() {
        let bpe = toy_bpe();
        let tokens = bpe.encode_ordinary("hello");
        // "he"=100, "ll"=101, "o"=3
        assert_eq!(tokens, vec![100, 101, 3]);
    }

    #[test]
    fn roundtrip_toy() {
        let bpe = toy_bpe();
        let text = "hello";
        let tokens = bpe.encode_ordinary(text);
        let decoded = bpe.decode_bytes(&tokens);
        assert_eq!(decoded, text.as_bytes());
        assert_eq!(bpe.decode(&tokens).unwrap(), text);
    }

    #[test]
    fn encode_single_token_and_lookup() {
        let bpe = toy_bpe();
        assert_eq!(bpe.encode_single_token(b"he"), Some(100));
        assert_eq!(bpe.encode_single_token(b"zz"), None);
        assert_eq!(bpe.decode_single_token_bytes(100).unwrap(), b"he".to_vec());
        assert!(bpe.decode_single_token_bytes(9999).is_err());
    }

    #[test]
    fn n_vocab_counts_everything() {
        let mut encoder = HashMap::default();
        encoder.insert(b"a".to_vec(), 0);
        encoder.insert(b"b".to_vec(), 1);
        let mut specials = HashMap::default();
        specials.insert("<|endoftext|>".to_string(), 2);
        let bpe = CoreBPE::new(encoder, specials, r"\w+").unwrap();
        assert_eq!(bpe.n_vocab(), 3);
    }

    #[test]
    fn encode_with_allowed_special() {
        let mut encoder = HashMap::default();
        for b in b"abcdefghijklmnopqrstuvwxyz <>|" {
            encoder.insert(vec![*b], *b as Rank);
        }
        let mut specials = HashMap::default();
        specials.insert("<|eot|>".to_string(), 999);
        let bpe = CoreBPE::new(encoder, specials, r"\w+|[<|>]").unwrap();

        let allowed: HashSet<&str> = std::iter::once("<|eot|>").collect();
        let tokens = bpe.encode("ab<|eot|>cd", &allowed);
        assert!(tokens.contains(&999));

        // When not allowed, the special string is tokenized as ordinary text
        // and the special rank does NOT appear.
        let empty: HashSet<&str> = HashSet::new();
        let tokens = bpe.encode("ab<|eot|>cd", &empty);
        assert!(!tokens.contains(&999));
    }

    #[test]
    fn fast_engine_kicks_in_on_tiktoken_patterns() {
        // The o200k_base pattern contains `\s+(?!\S)` which fancy-regex can
        // compile but regex crate cannot. The transformation should strip it
        // and produce a Fast engine.
        let o200k = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        let engine = SplitEngine::new(o200k).unwrap();
        assert!(engine.is_fast(), "o200k_base should use fast engine");

        // Simple pattern with no lookarounds should also use fast.
        let simple = SplitEngine::new(r"\w+|\s+").unwrap();
        assert!(simple.is_fast());
    }

    #[test]
    fn whitespace_shrink_matches_tiktoken_behavior() {
        // Build a toy BPE with a pattern that mimics the structure tiktoken
        // uses: word with optional leading non-word char, or whitespace.
        let mut encoder: HashMap<Vec<u8>, Rank> = HashMap::default();
        for b in 0u8..=255 {
            encoder.insert(vec![b], b as Rank);
        }
        // Inject a token for " hello" so we can detect the whitespace-attach
        // behavior via the whole-piece fast path.
        encoder.insert(b" hello".to_vec(), 1000);
        encoder.insert(b"hello".to_vec(), 1001);

        // Pattern mirrors tiktoken's shape: optional non-word + letters,
        // then the `\s+(?!\S)|\s+` whitespace tail that exercises the
        // fast-path shrink rule.
        let pattern = r"[^\r\n\p{L}\p{N}]?\p{L}+|\s+(?!\S)|\s+";
        let bpe = CoreBPE::new(encoder, HashMap::default(), pattern).unwrap();
        assert!(bpe.split_engine.is_fast());

        // "  hello" should tokenize as [" ", " hello"] — the first whitespace
        // is a standalone piece (one space) and the second attaches to the
        // following word as " hello".
        let tokens = bpe.encode_ordinary("  hello");
        assert_eq!(
            tokens,
            vec![b' ' as Rank, 1000],
            "fast path should replicate `\\s+(?!\\S)` whitespace-shrink behavior"
        );

        // "hello " (trailing whitespace) should tokenize as ["hello", " "]
        // — the trailing whitespace has no following word so it stays full.
        let tokens = bpe.encode_ordinary("hello ");
        assert_eq!(tokens, vec![1001, b' ' as Rank]);
    }

    #[test]
    fn whitespace_shrink_unified_mode_includes_newlines() {
        // gpt2 / r50k / p50k-style pattern: no separate `\s*[\r\n]+`
        // alternative, so `\s+(?!\S)` must absorb runs that contain a
        // newline and the shrink rule must fire on them. Regression test
        // for parity with tiktoken on gpt2-family encodings.
        let mut encoder: HashMap<Vec<u8>, Rank> = HashMap::default();
        for b in 0u8..=255 {
            encoder.insert(vec![b], b as Rank);
        }
        encoder.insert(b" hello".to_vec(), 1000);
        encoder.insert(b"hello".to_vec(), 1001);

        // Note the bare `\s` at the tail — this is the gpt2 shape.
        let pattern = r" ?\p{L}+|\s+$|\s+(?!\S)|\s";
        let bpe = CoreBPE::new(encoder, HashMap::default(), pattern).unwrap();
        assert!(bpe.split_engine.is_fast());

        // "\n  hello" — the whitespace run `\n  ` contains a newline but
        // must still be shrunk to `\n ` so the trailing space can attach
        // to `hello` as ` hello`. Expected pieces: ["\n ", " hello"].
        let tokens = bpe.encode_ordinary("\n  hello");
        assert_eq!(
            tokens,
            vec![b'\n' as Rank, b' ' as Rank, 1000],
            "unified shrink mode must fire on whitespace runs that include newlines"
        );

        // "\n" at end of input should stay whole — `\s+$` consumes it, and
        // with no following non-whitespace there's nothing to shrink for.
        let tokens = bpe.encode_ordinary("hi\n");
        assert_eq!(tokens, vec![b'h' as Rank, b'i' as Rank, b'\n' as Rank]);
    }

    #[test]
    fn batch_encode_matches_sequential() {
        let bpe = toy_bpe();
        let texts = vec!["hello", "hello world", "the lazy fox"];
        let batch = bpe.encode_ordinary_batch(&texts);
        let seq: Vec<Vec<Rank>> = texts.iter().map(|t| bpe.encode_ordinary(t)).collect();
        assert_eq!(batch, seq);

        // encode_batch with empty allowed set should equal encode_ordinary_batch
        let empty: HashSet<&str> = HashSet::new();
        let batch_sp = bpe.encode_batch(&texts, &empty);
        assert_eq!(batch_sp, seq);
    }

    #[test]
    fn large_piece_matches_small_piece() {
        // Cross-validation: the heap path should produce the same tokens
        // as the Vec path on pieces that exercise both.
        let mut ranks = HashMap::default();
        // Byte fallback
        for b in 0u8..=255 {
            ranks.insert(vec![b], b as Rank);
        }
        // A few merges
        ranks.insert(b"ab".to_vec(), 300);
        ranks.insert(b"cd".to_vec(), 301);
        ranks.insert(b"abcd".to_vec(), 302);

        let piece = b"abcdabcdabcdabcd";
        let small = {
            let pos = byte_pair_merge(&ranks, piece);
            pos.windows(2)
                .map(|w| rank_of(&ranks, &piece[w[0].0..w[1].0]))
                .collect::<Vec<_>>()
        };
        let large = byte_pair_merge_large(&ranks, piece);
        assert_eq!(small, large, "heap and vec paths disagree");
    }
}
