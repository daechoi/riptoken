//! Measurement spike: pre-compiled dense DFA sizes for tiktoken patterns.
//!
//! Builds a `regex_automata::dfa::regex::Regex` (forward + reverse DFAs) for
//! each stock tiktoken pattern after applying the same transformation that
//! `try_transform_for_fast_regex` uses (strip `\s+(?!\S)`, convert possessive
//! quantifiers).
//!
//! Reports:
//! - Dense DFA memory (forward + reverse)
//! - Sparse DFA memory (forward + reverse)
//! - Build time (with and without minimization)
//! - Correctness check vs lazy DFA on CJK + mixed-script text
//!
//! Usage:
//!
//! ```bash
//! cargo run --release --example dfa_spike
//! ```

use regex::Regex as LazyRegex;
use regex::RegexBuilder as LazyRegexBuilder;
use regex_automata::{
    Anchored, Input,
    dfa::{dense, regex::Regex as DfaRegex},
    nfa::thompson,
    util::syntax,
};
use std::time::Instant;

/// Stock tiktoken patterns, keyed by encoding name.
const PATTERNS: &[(&str, &str)] = &[
    (
        "gpt2 / r50k_base",
        concat!(
            r"'(?:[sdmt]|ll|ve|re)",
            r"| ?\p{L}+",
            r"| ?\p{N}+",
            r"| ?[^\s\p{L}\p{N}]+",
            r"|\s+(?!\S)",
            r"|\s+",
        ),
    ),
    (
        "p50k_base / p50k_edit",
        concat!(
            r"'(?:[sdmt]|ll|ve|re)",
            r"| ?\p{L}+",
            r"| ?\p{N}+",
            r"| ?[^\s\p{L}\p{N}]+",
            r"|\s+(?!\S)",
            r"|\s+",
        ),
    ),
    (
        "cl100k_base",
        concat!(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
            r"|[^\r\n\p{L}\p{N}]?\p{L}+",
            r"|\p{N}{1,3}",
            r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
            r"|\s*[\r\n]+",
            r"|\s+(?!\S)",
            r"|\s+",
        ),
    ),
    (
        "o200k_base",
        concat!(
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
            r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
            r"|\p{N}{1,3}",
            r"| ?[^\s\p{L}\p{N}]+[\r\n/]*",
            r"|\s*[\r\n]+",
            r"|\s+(?!\S)",
            r"|\s+",
        ),
    ),
];

/// Apply the same transformations as `try_transform_for_fast_regex` in lib.rs:
/// strip `\s+(?!\S)` lookaround, convert possessive quantifiers.
fn transform_pattern(pattern: &str) -> String {
    let mut s = pattern.replace(r"\s+(?!\S)|\s+", r"\s+");
    s = s.replace(r"\s+(?!\S)|\s", r"\s+");

    // Convert possessive quantifiers to greedy.
    s = s.replace("?+", "?").replace("++", "+").replace("*+", "*");
    let range_poss = LazyRegex::new(r"(\{\d+(?:,\d*)?\})\+").unwrap();
    range_poss.replace_all(&s, "$1").into_owned()
}

/// Build a dense DFA regex from a transformed pattern.
fn build_dense(pattern: &str) -> Result<DfaRegex, Box<dyn std::error::Error>> {
    let re = DfaRegex::builder()
        .syntax(syntax::Config::new().unicode(true).utf8(true))
        .thompson(thompson::Config::new())
        .dense(dense::Config::new().start_kind(regex_automata::dfa::StartKind::Unanchored))
        .build(pattern)?;
    Ok(re)
}

/// Build a dense DFA regex with minimization enabled.
fn build_dense_minimized(pattern: &str) -> Result<DfaRegex, Box<dyn std::error::Error>> {
    let re = DfaRegex::builder()
        .syntax(syntax::Config::new().unicode(true).utf8(true))
        .thompson(thompson::Config::new())
        .dense(
            dense::Config::new()
                .start_kind(regex_automata::dfa::StartKind::Unanchored)
                .minimize(true),
        )
        .build(pattern)?;
    Ok(re)
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1 << 20 {
        format!("{:.1} MB", bytes as f64 / (1 << 20) as f64)
    } else if bytes >= 1 << 10 {
        format!("{:.1} KB", bytes as f64 / (1 << 10) as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Collect all match spans from the lazy DFA regex.
fn lazy_matches(re: &LazyRegex, text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut pos = 0;
    while pos < text.len() {
        match re.find_at(text, pos) {
            Some(m) => {
                if m.start() > pos {
                    pos = m.start();
                }
                spans.push((m.start(), m.end()));
                if m.end() == pos {
                    pos += 1;
                } else {
                    pos = m.end();
                }
            }
            None => break,
        }
    }
    spans
}

/// Collect all match spans from the pre-compiled dense DFA regex.
fn dense_matches(re: &DfaRegex, text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut pos = 0;
    let haystack = text.as_bytes();
    while pos < haystack.len() {
        let input = Input::new(&haystack[pos..]).anchored(Anchored::No);
        match re.find(input) {
            Some(m) => {
                let start = pos + m.start();
                let end = pos + m.end();
                if start > pos {
                    pos = start;
                    continue;
                }
                spans.push((start, end));
                if end == pos {
                    pos += 1;
                } else {
                    pos = end;
                }
            }
            None => break,
        }
    }
    spans
}

fn main() {
    // Test text: mixed CJK, Latin, whitespace, numbers, punctuation.
    let cjk_text = concat!(
        "The quick brown fox jumps over the lazy dog. ",
        "これは日本語のテストです。",
        "中文测试文本，包含标点符号。",
        "한국어 테스트 텍스트입니다.",
        "   \t  multiple   spaces   and\ttabs\n\n",
        "Numbers: 123 456 789 0.123 1,234,567\n",
        "Mixed: Hello世界こんにちは123!@#$%\n",
        "Unicode: café résumé naïve über straße\n",
        "CJK repeated: ",
        "日本語テスト文章を繰り返し。",
        "日本語テスト文章を繰り返し。",
        "日本語テスト文章を繰り返し。",
        "日本語テスト文章を繰り返し。",
    );

    println!("=== DFA Spike: Pre-compiled Dense DFA Sizes ===\n");
    println!("Test text length: {} bytes\n", cjk_text.len());

    for (name, raw_pattern) in PATTERNS {
        println!("--- {name} ---");

        let pattern = transform_pattern(raw_pattern);
        println!("  Transformed pattern ({} chars)", pattern.len());

        // Build lazy DFA (current approach) for correctness baseline.
        let lazy_re = LazyRegexBuilder::new(&pattern)
            .dfa_size_limit(32 * (1 << 20))
            .build();
        let lazy_re = match lazy_re {
            Ok(re) => re,
            Err(e) => {
                println!("  SKIP: lazy regex build failed: {e}");
                continue;
            }
        };
        let lazy_spans = lazy_matches(&lazy_re, cjk_text);
        println!("  Lazy DFA: {} matches on test text", lazy_spans.len());

        // Dense DFA (no minimization).
        print!("  Dense DFA (no minimize): ");
        let t0 = Instant::now();
        match build_dense(&pattern) {
            Ok(dfa_re) => {
                let build_time = t0.elapsed();
                let fwd_size = dfa_re.forward().memory_usage();
                let rev_size = dfa_re.reverse().memory_usage();
                let total = fwd_size + rev_size;
                println!(
                    "fwd={} + rev={} = {} (built in {:.1}ms)",
                    format_size(fwd_size),
                    format_size(rev_size),
                    format_size(total),
                    build_time.as_secs_f64() * 1000.0
                );

                // Sparse DFA.
                let sparse_fwd = dfa_re.forward().to_sparse().unwrap();
                let sparse_rev = dfa_re.reverse().to_sparse().unwrap();
                let sparse_total = sparse_fwd.memory_usage() + sparse_rev.memory_usage();
                println!(
                    "  Sparse DFA:             fwd={} + rev={} = {}",
                    format_size(sparse_fwd.memory_usage()),
                    format_size(sparse_rev.memory_usage()),
                    format_size(sparse_total),
                );

                // Correctness check.
                let dense_spans = dense_matches(&dfa_re, cjk_text);
                if dense_spans == lazy_spans {
                    println!(
                        "  Correctness: PASS ({} matches identical)",
                        dense_spans.len()
                    );
                } else {
                    println!(
                        "  Correctness: FAIL (dense={} vs lazy={} matches)",
                        dense_spans.len(),
                        lazy_spans.len()
                    );
                    // Show first few differences.
                    let max_diff = 5;
                    let mut shown = 0;
                    for (i, (d, l)) in dense_spans.iter().zip(lazy_spans.iter()).enumerate() {
                        if d != l && shown < max_diff {
                            println!("    match[{i}]: dense={d:?} vs lazy={l:?}");
                            shown += 1;
                        }
                    }
                    if dense_spans.len() != lazy_spans.len() {
                        println!(
                            "    count mismatch: dense={} vs lazy={}",
                            dense_spans.len(),
                            lazy_spans.len()
                        );
                    }
                }
            }
            Err(e) => {
                println!("FAILED: {e}");
            }
        }

        // Dense DFA with minimization.
        print!("  Dense DFA (minimized):  ");
        let t0 = Instant::now();
        match build_dense_minimized(&pattern) {
            Ok(dfa_re) => {
                let build_time = t0.elapsed();
                let fwd_size = dfa_re.forward().memory_usage();
                let rev_size = dfa_re.reverse().memory_usage();
                let total = fwd_size + rev_size;
                println!(
                    "fwd={} + rev={} = {} (built in {:.1}ms)",
                    format_size(fwd_size),
                    format_size(rev_size),
                    format_size(total),
                    build_time.as_secs_f64() * 1000.0
                );
            }
            Err(e) => {
                println!("FAILED: {e}");
            }
        }

        println!();
    }

    // Decision gate summary.
    println!("=== Decision Gate ===");
    println!("  < 20 MB per pattern  → Proceed to Phase 2");
    println!("  20-50 MB             → Try sparse DFA; proceed only if < 20 MB sparse");
    println!("  > 50 MB              → Abort, accept lazy DFA performance");
}
