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
use regex::RegexBuilder as FastRegexBuilder;
use regex_automata::{
    Input,
    dfa::{dense, regex::Regex as DfaRegex},
    nfa::thompson,
    util::syntax,
};
use rustc_hash::{FxHashMap as HashMap, FxHasher};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

#[cfg(feature = "python")]
use pyo3::prelude::*;

// --- Build-time pre-compiled DFAs --------------------------------------------

#[cfg(feature = "precompiled-dfa")]
mod prebuilt {
    use super::*;

    // Raw patterns exactly as tiktoken provides them. gpt2, r50k_base,
    // p50k_base, and p50k_edit all share the same pattern.
    const GPT2_RAW: &str =
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";
    const CL100K_RAW: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s";
    const O200K_RAW: &str = concat!(
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"|\p{N}{1,3}",
        r"| ?[^\s\p{L}\p{N}]+[\r\n/]*",
        r"|\s*[\r\n]+",
        r"|\s+(?!\S)|\s+",
    );

    /// Force 4-byte alignment for `include_bytes!` data so that
    /// `dense::DFA::from_bytes` can reinterpret the slice as `&[u32]`.
    #[repr(C)]
    struct AlignAs<Align, Bytes: ?Sized> {
        _align: [Align; 0],
        bytes: Bytes,
    }

    macro_rules! include_dfa {
        ($path:expr) => {{
            const ALIGNED: &AlignAs<u32, [u8]> = &AlignAs {
                _align: [],
                bytes: *include_bytes!($path),
            };
            &ALIGNED.bytes
        }};
    }

    static GPT2_FWD: &[u8] = include_dfa!(concat!(env!("OUT_DIR"), "/gpt2_fwd.dfa"));
    static GPT2_REV: &[u8] = include_dfa!(concat!(env!("OUT_DIR"), "/gpt2_rev.dfa"));
    static CL100K_FWD: &[u8] = include_dfa!(concat!(env!("OUT_DIR"), "/cl100k_fwd.dfa"));
    static CL100K_REV: &[u8] = include_dfa!(concat!(env!("OUT_DIR"), "/cl100k_rev.dfa"));
    static O200K_FWD: &[u8] = include_dfa!(concat!(env!("OUT_DIR"), "/o200k_fwd.dfa"));
    static O200K_REV: &[u8] = include_dfa!(concat!(env!("OUT_DIR"), "/o200k_rev.dfa"));

    /// Try to load a pre-built DFA for a known stock tiktoken pattern.
    ///
    /// Returns `None` if the pattern doesn't match any stock pattern or if
    /// deserialization fails.
    pub(crate) fn try_load(pattern: &str) -> Option<(DfaRegex, ShrinkMode)> {
        let (fwd_bytes, rev_bytes, shrink_mode) = if pattern == GPT2_RAW {
            (GPT2_FWD, GPT2_REV, ShrinkMode::Unified)
        } else if pattern == CL100K_RAW {
            (CL100K_FWD, CL100K_REV, ShrinkMode::PlainOnly)
        } else if pattern == O200K_RAW {
            (O200K_FWD, O200K_REV, ShrinkMode::PlainOnly)
        } else {
            return None;
        };
        let (fwd, _) = dense::DFA::from_bytes(fwd_bytes).ok()?;
        let (rev, _) = dense::DFA::from_bytes(rev_bytes).ok()?;
        Some((
            DfaRegex::builder().build_from_dfas(fwd.to_owned(), rev.to_owned()),
            shrink_mode,
        ))
    }
}

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

/// Strip lookarounds and possessive quantifiers from a tiktoken-style pattern
/// so it can be compiled by a pure-DFA engine (`regex` crate or
/// `regex-automata` dense DFA).
///
/// Returns `Some((transformed, shrink_mode))` if the pattern is eligible for
/// the fast (non-backtracking) path, or `None` if it contains features that
/// require a full backtracking engine.
fn transform_pattern(pattern: &str) -> Option<(String, ShrinkMode)> {
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
    //    The shrink family is determined by the presence of a dedicated
    //    newline alternative like `\s*[\r\n]` or `\s*[\r\n]+`: patterns
    //    that have one (o200k/cl100k) use `PlainOnly` so that
    //    newline-containing matches from that alternative aren't shrunk;
    //    patterns without one (gpt2/r50k/p50k) use `Unified` so the
    //    shrink fires on all whitespace runs including newlines.
    let has_lookahead_ws = pattern.contains(r"\s+(?!\S)");
    let has_newline_alt = pattern.contains(r"\s*[\r\n]");
    let shrink_mode = if has_lookahead_ws {
        if has_newline_alt {
            ShrinkMode::PlainOnly
        } else {
            ShrinkMode::Unified
        }
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
    Some((stripped, shrink_mode))
}

/// Try to rewrite a tiktoken-style pattern into a form that the SIMD-accelerated
/// `regex` crate can compile (lazy DFA).
///
/// Returns `Some((regex, mode))` if the pattern compiles successfully with
/// the `regex` crate after transformation, or `None` if the pattern contains
/// other lookarounds (or features) that regex can't handle — in which case
/// the caller falls back to `fancy-regex`.
fn try_transform_for_fast_regex(pattern: &str) -> Option<(FastRegex, ShrinkMode)> {
    let (transformed, shrink_mode) = transform_pattern(pattern)?;
    let regex = FastRegexBuilder::new(&transformed)
        .dfa_size_limit(32 * (1 << 20))
        .build()
        .ok()?;
    Some((regex, shrink_mode))
}

/// Try to build a fully pre-compiled dense DFA for the pattern.
///
/// Unlike the lazy DFA in `try_transform_for_fast_regex`, this materializes
/// all states upfront — zero cold-start at search time, and the resulting
/// `DfaRegex` has no mutable state so it can be shared across threads without
/// cloning.
///
/// Returns `None` if the pattern can't be transformed or the DFA build fails
/// (e.g. the state table is too large).
fn try_build_precompiled_dfa(pattern: &str) -> Option<(DfaRegex, ShrinkMode)> {
    let (transformed, shrink_mode) = transform_pattern(pattern)?;
    let dfa = DfaRegex::builder()
        .syntax(syntax::Config::new().unicode(true).utf8(true))
        .thompson(thompson::Config::new())
        .dense(dense::Config::new().start_kind(regex_automata::dfa::StartKind::Unanchored))
        .build(&transformed)
        .ok()?;
    Some((dfa, shrink_mode))
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

/// Apply the `\s+(?!\S)` whitespace-shrink rule, returning the adjusted end.
///
/// If the matched piece is a whitespace run (per `shrink_mode`) and the
/// character immediately after it is non-whitespace, shrink by one character
/// so that trailing whitespace can attach to the following word.
#[inline]
fn apply_shrink(text: &str, start: usize, end: usize, shrink_mode: ShrinkMode) -> usize {
    let piece = &text[start..end];
    let should_shrink = match shrink_mode {
        ShrinkMode::None => false,
        ShrinkMode::PlainOnly => is_plain_whitespace_run(piece),
        ShrinkMode::Unified => is_whitespace_run(piece),
    };
    if should_shrink && end < text.len() && next_char_is_non_whitespace(text, end) {
        if let Some((last_i, _)) = piece.char_indices().next_back() {
            if last_i > 0 {
                return start + last_i;
            }
        }
    }
    end
}

/// The pattern-matching engine used to split text into pieces before BPE.
enum SplitEngine {
    /// Pre-compiled dense DFA — all states materialized upfront, zero
    /// cold-start penalty. The `DfaRegex` has no mutable state, so one
    /// instance is shared across all threads without cloning.
    PrecompiledDfa {
        dfa_regex: Box<DfaRegex>,
        shrink_mode: ShrinkMode,
    },
    /// SIMD-accelerated lazy DFA. One clone per thread slot, plus a
    /// [`ShrinkMode`] selecting how to emulate `\s+(?!\S)` in Rust.
    Fast {
        clones: Vec<FastRegex>,
        shrink_mode: ShrinkMode,
    },
    /// Backtracking regex with lookaround support. One clone per thread slot.
    Fancy(Vec<FancyRegex>),
}

impl SplitEngine {
    /// Build an engine for `pattern`.
    ///
    /// Priority: pre-built DFA (stock patterns) → eager DFA build →
    /// lazy DFA → fancy-regex.
    fn new(pattern: &str) -> Result<Self, BuildError> {
        // 1. Try pre-built DFA from build.rs (zero cost, stock patterns only).
        #[cfg(feature = "precompiled-dfa")]
        if let Some((dfa_regex, shrink_mode)) = prebuilt::try_load(pattern) {
            return Ok(SplitEngine::PrecompiledDfa {
                dfa_regex: Box::new(dfa_regex),
                shrink_mode,
            });
        }
        // 2. Try eager dense DFA build (non-stock patterns).
        if let Some((dfa_regex, shrink_mode)) = try_build_precompiled_dfa(pattern) {
            return Ok(SplitEngine::PrecompiledDfa {
                dfa_regex: Box::new(dfa_regex),
                shrink_mode,
            });
        }
        // 3. Try lazy DFA.
        if let Some((fast, shrink_mode)) = try_transform_for_fast_regex(pattern) {
            let clones: Vec<FastRegex> = (0..MAX_NUM_THREADS).map(|_| fast.clone()).collect();
            return Ok(SplitEngine::Fast {
                clones,
                shrink_mode,
            });
        }
        // 4. Fallback: fancy-regex with lookaround support.
        let fancy = FancyRegex::new(pattern)?;
        let clones: Vec<FancyRegex> = (0..MAX_NUM_THREADS).map(|_| fancy.clone()).collect();
        Ok(SplitEngine::Fancy(clones))
    }

    /// True if this engine uses a non-backtracking path (precompiled or lazy DFA).
    #[cfg(test)]
    fn is_fast(&self) -> bool {
        matches!(
            self,
            SplitEngine::PrecompiledDfa { .. } | SplitEngine::Fast { .. }
        )
    }

    /// True if this engine uses a pre-compiled dense DFA (zero cold-start).
    #[cfg(all(test, feature = "precompiled-dfa"))]
    fn is_precompiled(&self) -> bool {
        matches!(self, SplitEngine::PrecompiledDfa { .. })
    }

    /// Iterate the pieces of `text`, invoking `f` with each piece as a `&str`.
    ///
    /// On the fast paths, applies the `\s+(?!\S)` whitespace-shrink rule in
    /// Rust so that output matches tiktoken exactly.
    #[inline]
    fn find_pieces<F: FnMut(&str)>(&self, text: &str, mut f: F) {
        match self {
            SplitEngine::PrecompiledDfa {
                dfa_regex,
                shrink_mode,
            } => {
                let haystack = text.as_bytes();
                let mut pos = 0;
                while pos < haystack.len() {
                    let input = Input::new(haystack).range(pos..);
                    let m = match dfa_regex.find(input) {
                        Some(m) => m,
                        None => break,
                    };
                    if m.start() > pos {
                        pos = m.start();
                    }
                    let start = m.start();
                    let end = apply_shrink(text, start, m.end(), *shrink_mode);
                    f(&text[start..end]);
                    if end == pos {
                        pos += 1;
                    } else {
                        pos = end;
                    }
                }
            }
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
                    if m.start() > pos {
                        pos = m.start();
                    }
                    let start = m.start();
                    let end = apply_shrink(text, start, m.end(), *shrink_mode);
                    f(&text[start..end]);
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

#[derive(Clone, Copy)]
enum SpecialPolicy<'set, 'token> {
    AllowSet(&'set HashSet<&'token str>),
    AllowAll,
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

    // Heap entry: Reverse((rank, start)), a min-heap on both, so rank ties pop
    // the leftmost pair first, as in the Vec path and tiktoken.
    let mut heap: BinaryHeap<Reverse<(Rank, usize)>> = BinaryHeap::with_capacity(n);

    // Seed with all initial pair ranks.
    for i in 0..n.saturating_sub(1) {
        let rank = rank_of(ranks, &piece[i..state[i + 1].end]);
        state[i].cur_rank = rank;
        if rank != Rank::MAX {
            heap.push(Reverse((rank, i)));
        }
    }

    while let Some(Reverse((rank, start))) = heap.pop() {
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
                heap.push(Reverse((new_rank, start)));
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
                heap.push(Reverse((new_rank, prev)));
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

    /// Encode many texts in parallel using rayon, allowing every registered
    /// special token.
    ///
    /// See [`CoreBPE::encode_with_special_tokens`] for the special-token
    /// semantics.
    pub fn encode_with_special_tokens_batch(&self, texts: &[&str]) -> Vec<Vec<Rank>> {
        use rayon::prelude::*;
        texts
            .par_iter()
            .map(|t| self.encode_with_special_tokens(t))
            .collect()
    }

    fn encode_impl(&self, text: &str, special_policy: SpecialPolicy<'_, '_>) -> Vec<Rank> {
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
                        let piece = &text[m.start()..m.end()];
                        let is_allowed = match special_policy {
                            SpecialPolicy::AllowSet(allowed_special) => {
                                allowed_special.contains(piece)
                            }
                            SpecialPolicy::AllowAll => true,
                        };
                        if is_allowed {
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

    /// Encode text, allowing a specific set of special tokens.
    ///
    /// Special tokens in `allowed_special` are emitted as their assigned ranks.
    /// Special tokens NOT in `allowed_special` are tokenized as ordinary text.
    pub fn encode(&self, text: &str, allowed_special: &HashSet<&str>) -> Vec<Rank> {
        self.encode_impl(text, SpecialPolicy::AllowSet(allowed_special))
    }

    /// Encode text, allowing every special token registered in this encoder.
    ///
    /// This is equivalent to calling [`CoreBPE::encode`] with an allow set
    /// containing all keys from `special_tokens_encoder`, but avoids requiring
    /// callers to materialize that set on every encode call.
    pub fn encode_with_special_tokens(&self, text: &str) -> Vec<Rank> {
        self.encode_impl(text, SpecialPolicy::AllowAll)
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

    #[pyo3(name = "encode", signature = (text, allowed_special = None))]
    fn py_encode(
        &self,
        py: Python<'_>,
        text: &str,
        allowed_special: Option<HashSet<String>>,
    ) -> Vec<Rank> {
        py.detach(|| {
            let allowed = allowed_special.unwrap_or_default();
            let allowed_refs: HashSet<&str> = allowed.iter().map(|s| s.as_str()).collect();
            self.encode(text, &allowed_refs)
        })
    }

    #[pyo3(name = "encode_with_special_tokens")]
    fn py_encode_with_special_tokens(&self, py: Python<'_>, text: &str) -> Vec<Rank> {
        py.detach(|| self.encode_with_special_tokens(text))
    }

    #[pyo3(name = "encode_ordinary_batch")]
    fn py_encode_ordinary_batch(&self, py: Python<'_>, texts: Vec<String>) -> Vec<Vec<Rank>> {
        py.detach(|| {
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            self.encode_ordinary_batch(&refs)
        })
    }

    #[pyo3(name = "encode_batch", signature = (texts, allowed_special = None))]
    fn py_encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        allowed_special: Option<HashSet<String>>,
    ) -> Vec<Vec<Rank>> {
        py.detach(|| {
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let allowed = allowed_special.unwrap_or_default();
            let allowed_refs: HashSet<&str> = allowed.iter().map(|s| s.as_str()).collect();
            self.encode_batch(&refs, &allowed_refs)
        })
    }

    #[pyo3(name = "encode_with_special_tokens_batch")]
    fn py_encode_with_special_tokens_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
    ) -> Vec<Vec<Rank>> {
        py.detach(|| {
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            self.encode_with_special_tokens_batch(&refs)
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

    /// Decode tokens into a Python `str`, matching `tiktoken.Encoding.decode`.
    ///
    /// Invalid UTF-8 sequences (which can occur mid-stream when a multi-byte
    /// character spans a token boundary) are replaced with U+FFFD, matching
    /// tiktoken's default `errors="replace"` behavior. For strict decoding or
    /// raw bytes, use [`decode_bytes`].
    #[pyo3(name = "decode")]
    fn py_decode(&self, py: Python<'_>, tokens: Vec<Rank>) -> String {
        py.detach(|| {
            let bytes = self.decode_bytes(&tokens);
            String::from_utf8_lossy(&bytes).into_owned()
        })
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
    fn encode_with_special_tokens_allows_every_registered_special() {
        let mut encoder = HashMap::default();
        for b in b"abcdefghijklmnopqrstuvwxyz <>|_" {
            encoder.insert(vec![*b], *b as Rank);
        }
        let mut specials = HashMap::default();
        specials.insert("<|eot|>".to_string(), 999);
        specials.insert("<|fim_prefix|>".to_string(), 1000);
        let bpe = CoreBPE::new(encoder, specials, r"\w+|[<|>]").unwrap();

        let text = "ab<|eot|>cd<|fim_prefix|>ef";
        let allowed: HashSet<&str> = ["<|eot|>", "<|fim_prefix|>"].into_iter().collect();

        let tokens = bpe.encode_with_special_tokens(text);
        assert_eq!(tokens, bpe.encode(text, &allowed));
        assert!(tokens.contains(&999));
        assert!(tokens.contains(&1000));

        let texts = ["ab<|eot|>", "cd<|fim_prefix|>"];
        assert_eq!(
            bpe.encode_with_special_tokens_batch(&texts),
            bpe.encode_batch(&texts, &allowed)
        );
    }

    #[test]
    fn fast_engine_kicks_in_on_tiktoken_patterns() {
        // The o200k_base pattern contains `\s+(?!\S)` which fancy-regex can
        // compile but regex crate cannot. The transformation should strip it
        // and produce a Fast engine (precompiled or lazy DFA).
        let o200k = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        let engine = SplitEngine::new(o200k).unwrap();
        assert!(engine.is_fast(), "o200k_base should use fast engine");

        // Simple pattern with no lookarounds should also use fast.
        let simple = SplitEngine::new(r"\w+|\s+").unwrap();
        assert!(simple.is_fast());
    }

    #[test]
    #[cfg(feature = "precompiled-dfa")]
    fn prebuilt_dfa_used_for_stock_patterns() {
        // Stock tiktoken patterns should hit the prebuilt DFA path.
        let o200k_raw = concat!(
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
            r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
            r"|\p{N}{1,3}",
            r"| ?[^\s\p{L}\p{N}]+[\r\n/]*",
            r"|\s*[\r\n]+",
            r"|\s+(?!\S)|\s+",
        );
        let engine = SplitEngine::new(o200k_raw).unwrap();
        assert!(
            engine.is_precompiled(),
            "o200k_base stock pattern should use prebuilt DFA"
        );

        let gpt2_raw =
            r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";
        let engine = SplitEngine::new(gpt2_raw).unwrap();
        assert!(
            engine.is_precompiled(),
            "gpt2 stock pattern should use prebuilt DFA"
        );

        let cl100k_raw = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s";
        let engine = SplitEngine::new(cl100k_raw).unwrap();
        assert!(
            engine.is_precompiled(),
            "cl100k_base stock pattern should use prebuilt DFA"
        );

        // Non-stock pattern still uses the fast path (eager DFA build).
        let custom = SplitEngine::new(r"\w+|\s+").unwrap();
        assert!(custom.is_fast(), "custom pattern should use fast engine");
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

    #[test]
    fn large_piece_breaks_rank_ties_from_the_left() {
        // In a run of one byte every pair has the same rank. tiktoken and the
        // Vec path merge the leftmost pair first, so the heap path must too.
        let mut ranks = HashMap::default();
        for b in 0u8..=255 {
            ranks.insert(vec![b], b as Rank);
        }
        ranks.insert(b"aa".to_vec(), 300);
        for n in [3, 5, 7] {
            let piece = vec![b'a'; n];
            let small = {
                let pos = byte_pair_merge(&ranks, &piece);
                pos.windows(2)
                    .map(|w| rank_of(&ranks, &piece[w[0].0..w[1].0]))
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                small.last(),
                Some(&(b'a' as Rank)),
                "odd run ends in a lone byte"
            );
            assert_eq!(byte_pair_merge_large(&ranks, &piece), small, "run of {n}");
        }
    }
}
