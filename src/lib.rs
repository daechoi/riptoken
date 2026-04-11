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

use fancy_regex::Regex;
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
    /// The regex pattern failed to compile.
    InvalidRegex(fancy_regex::Error),
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
        BuildError::InvalidRegex(e)
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
    /// Pre-cloned regex instances, one per "thread slot". Indexed via
    /// [`thread_index`]. Avoids contention on `fancy_regex`'s internal mutable
    /// scratch state when the tokenizer is used from multiple threads.
    regex_tls: Vec<Regex>,
    /// Thread-local clones of the special-token regex. Empty if there are no
    /// special tokens.
    special_regex_tls: Vec<Regex>,
    /// Sorted vocabulary bytes, useful for prefix queries.
    sorted_token_bytes: Vec<Vec<u8>>,
}

// --- Core BPE merge algorithm -------------------------------------------------

/// Compute the rank of a byte pair directly from a slice — no allocation.
///
/// `HashMap<Vec<u8>, Rank>::get` accepts any `Q: ?Sized` where
/// `Vec<u8>: Borrow<Q>`, and `Vec<u8>` implements `Borrow<[u8]>`, so this
/// avoids the `.to_vec()` allocation the naive implementation does.
#[inline]
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
        positions
            .windows(2)
            .map(|w| rank_of(ranks, &piece[w[0].0..w[1].0]))
            .collect()
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
fn build_special_regex(specials: &HashMap<String, Rank>) -> Result<Option<Regex>, BuildError> {
    if specials.is_empty() {
        return Ok(None);
    }
    // Escape each special token literally and join with `|`.
    let parts: Vec<String> = specials
        .keys()
        .map(|s| fancy_regex::escape(s).into_owned())
        .collect();
    let pattern = parts.join("|");
    Ok(Some(Regex::new(&pattern)?))
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
        let regex = Regex::new(pattern)?;
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

        let regex_tls: Vec<Regex> = (0..MAX_NUM_THREADS).map(|_| regex.clone()).collect();
        let special_regex_tls: Vec<Regex> = match special_regex {
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
            regex_tls,
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
    fn tl_regex(&self) -> &Regex {
        &self.regex_tls[thread_index()]
    }

    #[inline]
    fn tl_special_regex(&self) -> Option<&Regex> {
        self.special_regex_tls.get(thread_index())
    }

    /// Encode ordinary text, ignoring special tokens entirely.
    ///
    /// Any special-token substrings in the input will be tokenized as regular
    /// text (this matches tiktoken's behavior).
    pub fn encode_ordinary(&self, text: &str) -> Vec<Rank> {
        let regex = self.tl_regex();
        let mut ret = Vec::new();
        for mat in regex.find_iter(text) {
            let m = match mat {
                Ok(m) => m,
                Err(_) => continue,
            };
            let piece = m.as_str().as_bytes();
            // Whole-piece fast path: most regex splits are already full tokens.
            if let Some(&token) = self.encoder.get(piece) {
                ret.push(token);
                continue;
            }
            ret.extend(byte_pair_encode(piece, &self.encoder));
        }
        ret
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
        let regex = self.tl_regex();

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
            for mat in regex.find_iter(&text[start..end]) {
                let m = match mat {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                let piece = m.as_str().as_bytes();
                if let Some(&token) = self.encoder.get(piece) {
                    ret.push(token);
                    continue;
                }
                ret.extend(byte_pair_encode(piece, &self.encoder));
            }

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
