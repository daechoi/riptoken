//! Integration tests for riptoken. These exercise the public API from an
//! external-crate perspective (only `pub` items are visible here).

use riptoken::{CoreBPE, Rank};
use rustc_hash::FxHashMap;

/// A small toy vocabulary useful for end-to-end testing.
fn toy() -> CoreBPE {
    let mut encoder: FxHashMap<Vec<u8>, Rank> = FxHashMap::default();
    // Single bytes for everything we'll use
    let chars = b"abcdefghijklmnopqrstuvwxyz ";
    for (i, &b) in chars.iter().enumerate() {
        encoder.insert(vec![b], i as Rank);
    }
    // A few merges
    encoder.insert(b"he".to_vec(), 100);
    encoder.insert(b"ll".to_vec(), 101);
    encoder.insert(b"hello".to_vec(), 102);
    encoder.insert(b" world".to_vec(), 103);

    // Use a pattern that keeps leading whitespace attached to the next word,
    // matching how tiktoken patterns actually work. This produces pieces like
    // " world" as a single match, which lets us verify the vocab-whole-piece
    // fast path in `encode_ordinary`.
    CoreBPE::new(encoder, FxHashMap::default(), r" ?\w+").unwrap()
}

#[test]
fn encode_decode_roundtrip() {
    let bpe = toy();
    for text in ["", "a", "hello", "hello world", "abc def ghi"] {
        let tokens = bpe.encode_ordinary(text);
        let decoded = bpe.decode_bytes(&tokens);
        assert_eq!(decoded, text.as_bytes(), "roundtrip failed for {text:?}");
    }
}

#[test]
fn whole_piece_fast_path_is_used() {
    let bpe = toy();
    // " world" is a single vocab entry — should emit just one token.
    let tokens = bpe.encode_ordinary("hello world");
    // "hello" is a single vocab entry too.
    assert_eq!(tokens, vec![102, 103]);
}

#[test]
fn invalid_regex_errors() {
    let encoder: FxHashMap<Vec<u8>, Rank> = FxHashMap::default();
    let specials: FxHashMap<String, Rank> = FxHashMap::default();
    let result = CoreBPE::new(encoder, specials, "([unclosed");
    assert!(result.is_err());
}

#[test]
fn decode_skips_unknown_tokens() {
    let bpe = toy();
    // 9999 is not in the vocab — should be silently skipped.
    let bytes = bpe.decode_bytes(&[100, 9999, 101]);
    assert_eq!(bytes, b"hell".to_vec());
}

#[test]
fn decode_single_token_errors_on_unknown() {
    let bpe = toy();
    assert!(bpe.decode_single_token_bytes(9999).is_err());
}

#[test]
fn sends_across_threads() {
    use std::sync::Arc;
    use std::thread;

    let bpe = Arc::new(toy());
    let text = "hello world hello world";

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let bpe = Arc::clone(&bpe);
            thread::spawn(move || {
                for _ in 0..100 {
                    let t = bpe.encode_ordinary(text);
                    let b = bpe.decode_bytes(&t);
                    assert_eq!(b, text.as_bytes());
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn long_piece_takes_heap_path_and_matches() {
    // Build a vocab with byte fallback plus some merges, then feed a long
    // input that exceeds the heap threshold.
    let mut encoder: FxHashMap<Vec<u8>, Rank> = FxHashMap::default();
    for b in 0u8..=255 {
        encoder.insert(vec![b], b as Rank);
    }
    encoder.insert(b"ab".to_vec(), 300);
    encoder.insert(b"cd".to_vec(), 301);
    encoder.insert(b"abcd".to_vec(), 302);

    let bpe = CoreBPE::new(encoder, FxHashMap::default(), r"[a-z]+").unwrap();

    // A piece >= 500 bytes to trigger the heap path.
    let long = "abcd".repeat(200);
    let tokens = bpe.encode_ordinary(&long);
    let decoded = bpe.decode_bytes(&tokens);
    assert_eq!(decoded, long.as_bytes());
    // Every "abcd" should merge into the single token 302.
    assert!(tokens.iter().all(|&t| t == 302));
}
