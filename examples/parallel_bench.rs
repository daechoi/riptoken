//! Rust-only parallel batch benchmark — eliminates pyo3 as a variable.
//!
//! Usage:
//!
//! ```bash
//! cargo run --release --example parallel_bench -- o200k_base.tiktoken
//! ```

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use riptoken::{CoreBPE, Rank};
use rustc_hash::FxHashMap;
use std::env;
use std::fs;
use std::time::Instant;

fn load(path: &str) -> FxHashMap<Vec<u8>, Rank> {
    let raw = fs::read_to_string(path).expect("read tiktoken file");
    let mut ranks: FxHashMap<Vec<u8>, Rank> = FxHashMap::default();
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let b64 = parts.next().expect("b64");
        let rank_str = parts.next().expect("rank");
        ranks.insert(
            B64.decode(b64).expect("b64 decode"),
            rank_str.parse().expect("rank parse"),
        );
    }
    ranks
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("o200k_base.tiktoken");

    let pat = concat!(
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"|\p{N}{1,3}",
        r"| ?[^\s\p{L}\p{N}]+[\r\n/]*",
        r"|\s*[\r\n]+",
        r"|\s+(?!\S)",
        r"|\s+",
    );

    eprintln!("loading vocab from {path}...");
    let encoder = load(path);
    let specials: FxHashMap<String, Rank> = [("<|endoftext|>".to_string(), 199999_u32)]
        .into_iter()
        .collect();
    let bpe = CoreBPE::new(encoder, specials, pat).expect("build bpe");

    let text: String = "The quick brown fox jumps over the lazy dog. ".repeat(1000);
    let n_tok_per_doc = bpe.encode_ordinary(&text).len();
    eprintln!("rayon pool size: {}", rayon::current_num_threads());
    eprintln!("tokens per doc: {}", n_tok_per_doc);

    // Sequential baseline.
    let iters_seq: usize = 100;
    let t0 = Instant::now();
    for _ in 0..iters_seq {
        bpe.encode_ordinary(&text);
    }
    let seq = t0.elapsed();
    let seq_tps = (n_tok_per_doc * iters_seq) as f64 / seq.as_secs_f64();
    println!(
        "sequential {:4} iters: {:6.1}ms  {:>5.1}M tps",
        iters_seq,
        seq.as_secs_f64() * 1000.0,
        seq_tps / 1e6
    );

    // Parallel batches.
    for &n in &[8_usize, 16, 32, 64, 128, 256, 512, 1024] {
        let docs: Vec<&str> = (0..n).map(|_| text.as_str()).collect();
        bpe.encode_ordinary_batch(&docs); // warm up
        let t0 = Instant::now();
        bpe.encode_ordinary_batch(&docs);
        let dt = t0.elapsed();
        let tps = (n_tok_per_doc * n) as f64 / dt.as_secs_f64();
        let speedup = tps / seq_tps;
        println!(
            "batch n={:4}: {:7.1}ms  {:>5.1}M tps  {:5.1}x vs seq",
            n,
            dt.as_secs_f64() * 1000.0,
            tps / 1e6,
            speedup
        );
    }
}
