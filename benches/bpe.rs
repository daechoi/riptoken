//! Criterion micro-benchmarks for the core BPE algorithm.
//!
//! Run with `cargo bench --bench bpe`. These benchmarks use a small synthetic
//! vocabulary so they run without any external data; the realistic
//! `o200k_base`-based comparison against tiktoken lives in
//! `scripts/bench.py`.

use std::hint::black_box;
use std::time::Instant;

use riptoken::{CoreBPE, Rank};
use rustc_hash::FxHashMap;

fn build_toy() -> CoreBPE {
    let mut encoder: FxHashMap<Vec<u8>, Rank> = FxHashMap::default();
    // All single bytes as byte fallback.
    for b in 0u8..=255 {
        encoder.insert(vec![b], b as Rank);
    }
    // Common English bigrams and trigrams.
    for (i, pair) in [
        &b"th"[..],
        &b"he"[..],
        &b"in"[..],
        &b"er"[..],
        &b"an"[..],
        &b"re"[..],
        &b"on"[..],
        &b"at"[..],
        &b"en"[..],
        &b"nd"[..],
        &b"the"[..],
        &b"and"[..],
        &b"ing"[..],
        &b"ion"[..],
        &b"ent"[..],
        &b"for"[..],
        &b"tion"[..],
        &b" the"[..],
        &b" and"[..],
        &b" of"[..],
    ]
    .iter()
    .enumerate()
    {
        encoder.insert(pair.to_vec(), 256 + i as Rank);
    }
    CoreBPE::new(encoder, FxHashMap::default(), r" ?\w+|[^\w\s]+|\s+").unwrap()
}

fn bench_fn<F: FnMut()>(label: &str, iters: u64, mut f: F) {
    // Warm-up
    for _ in 0..3 {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let ns_per_iter = elapsed.as_nanos() as f64 / iters as f64;
    println!("{label:<40} {ns_per_iter:>12.0} ns/iter  ({iters} iters)");
}

fn main() {
    let bpe = build_toy();
    println!("riptoken micro-benchmarks\n");

    let short = "hello world and the fox";
    bench_fn("encode_ordinary short", 100_000, || {
        black_box(bpe.encode_ordinary(black_box(short)));
    });

    let medium = "the quick brown fox jumps over the lazy dog ".repeat(20);
    bench_fn("encode_ordinary medium", 10_000, || {
        black_box(bpe.encode_ordinary(black_box(&medium)));
    });

    let long = "abcdefghijklmnopqrstuvwxyz ".repeat(200);
    bench_fn("encode_ordinary long", 1_000, || {
        black_box(bpe.encode_ordinary(black_box(&long)));
    });

    let tokens = bpe.encode_ordinary(&medium);
    bench_fn("decode_bytes medium", 100_000, || {
        black_box(bpe.decode_bytes(black_box(&tokens)));
    });
}
