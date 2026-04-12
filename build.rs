//! Build script: pre-compile dense DFAs for stock tiktoken patterns.
//!
//! When the `precompiled-dfa` feature is enabled, this script builds a
//! fully-materialized dense DFA for each stock tiktoken pattern at compile
//! time. The serialized bytes are embedded in the binary via `include_bytes!`
//! and deserialized at near-zero cost at runtime — eliminating both the ~1.5s
//! eager DFA build and the ~55ms lazy-DFA cold-start.

fn main() {
    // Only rerun when this file changes — the patterns are hard-coded.
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "precompiled-dfa")]
    precompile::run();
}

#[cfg(feature = "precompiled-dfa")]
mod precompile {
    use regex::Regex;
    use regex_automata::{
        dfa::{dense, regex::Regex as DfaRegex},
        nfa::thompson,
        util::syntax,
    };
    use std::env;
    use std::fs;
    use std::path::PathBuf;

    /// Stock tiktoken patterns (raw, exactly as tiktoken provides them).
    ///
    /// gpt2, r50k_base, p50k_base, and p50k_edit all share the same pattern.
    const STOCK_PATTERNS: &[(&str, &str)] = &[
        (
            "gpt2",
            r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s",
        ),
        (
            "cl100k",
            r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s",
        ),
        (
            "o200k",
            concat!(
                r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                r"|\p{N}{1,3}",
                r"| ?[^\s\p{L}\p{N}]+[\r\n/]*",
                r"|\s*[\r\n]+",
                r"|\s+(?!\S)|\s+",
            ),
        ),
    ];

    /// Same transformation as `transform_pattern` in lib.rs: strip the
    /// `\s+(?!\S)` lookaround and convert possessive quantifiers to greedy.
    fn transform_pattern(pattern: &str) -> Option<String> {
        let mut stripped = pattern.replace(r"\s+(?!\S)|\s+", r"\s+");
        stripped = stripped.replace(r"\s+(?!\S)|\s", r"\s+");
        if stripped.contains("(?=")
            || stripped.contains("(?!")
            || stripped.contains("(?<=")
            || stripped.contains("(?<!")
        {
            return None;
        }
        stripped = stripped
            .replace("?+", "?")
            .replace("++", "+")
            .replace("*+", "*");
        let range_possessive = Regex::new(r"(\{\d+(?:,\d*)?\})\+").ok()?;
        let stripped = range_possessive.replace_all(&stripped, "$1").into_owned();
        Some(stripped)
    }

    pub fn run() {
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let target_endian =
            env::var("CARGO_CFG_TARGET_ENDIAN").unwrap_or_else(|_| "little".to_string());

        for (name, raw_pattern) in STOCK_PATTERNS {
            let transformed = match transform_pattern(raw_pattern) {
                Some(t) => t,
                None => {
                    println!("cargo:warning=precompiled-dfa: failed to transform {name}");
                    continue;
                }
            };

            let dfa = match DfaRegex::builder()
                .syntax(syntax::Config::new().unicode(true).utf8(true))
                .thompson(thompson::Config::new())
                .dense(dense::Config::new().start_kind(regex_automata::dfa::StartKind::Unanchored))
                .build(&transformed)
            {
                Ok(dfa) => dfa,
                Err(e) => {
                    println!("cargo:warning=precompiled-dfa: DFA build failed for {name}: {e}");
                    continue;
                }
            };

            let (fwd_bytes, rev_bytes) = if target_endian == "big" {
                (
                    dfa.forward().to_bytes_big_endian().0,
                    dfa.reverse().to_bytes_big_endian().0,
                )
            } else {
                (
                    dfa.forward().to_bytes_little_endian().0,
                    dfa.reverse().to_bytes_little_endian().0,
                )
            };

            fs::write(out_dir.join(format!("{name}_fwd.dfa")), &fwd_bytes).unwrap();
            fs::write(out_dir.join(format!("{name}_rev.dfa")), &rev_bytes).unwrap();
        }
    }
}
