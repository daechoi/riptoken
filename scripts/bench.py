"""Benchmark riptoken vs tiktoken on a variety of realistic workloads.

Runs four different corpora that exercise different parts of the encoder:

1. English pangram prose — friendly to the vocab-fast-path
2. Source code (Python + Rust) — lots of punctuation, identifiers, whitespace
3. Multilingual prose — CJK, emoji, accented latin
4. Dense random-ish bytes — exercises the BPE merge loop hard

For each corpus both libraries must produce byte-identical output, otherwise
the benchmark aborts with a diff.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import tiktoken

from riptoken import CoreBPE, load_tiktoken_bpe

# o200k_base regex pattern (copied from tiktoken's openai_public.py)
PAT_STR = "|".join(
    [
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)

SPECIAL_TOKENS = {"<|endoftext|>": 199999, "<|endofprompt|>": 200018}


# --- Corpora -----------------------------------------------------------------

PANGRAMS = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "Sphinx of black quartz, judge my vow. "
)

PYTHON_CODE = '''
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

@dataclass(frozen=True, slots=True)
class Request:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes = b""

async def handle(req: Request) -> AsyncIterator[bytes]:
    """Stream an HTTP response body back to the caller."""
    async with connect() as conn:
        cur = await conn.cursor("SELECT id, payload FROM events WHERE ts > $1")
        while rec := await cur.fetchmany(512):
            for r in rec:
                yield serialize(r.id, r.payload).encode("utf-8")
'''

RUST_CODE = r'''
use std::collections::BTreeMap;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct Metric<'a> {
    pub name: &'a str,
    pub labels: BTreeMap<&'a str, &'a str>,
    pub value: f64,
}

impl<'a> Metric<'a> {
    pub fn new(name: &'a str) -> Self {
        Self { name, labels: BTreeMap::new(), value: 0.0 }
    }

    pub fn observe(&mut self, v: f64) {
        self.value += v;
    }
}

async fn gather(tx: mpsc::Sender<Metric<'_>>) -> Result<(), BoxErr> {
    let m = Metric::new("requests_total");
    tx.send(m).await.map_err(Into::into)
}
'''

MULTILINGUAL = (
    "こんにちは、世界。今日は良い天気ですね。"
    "这是一段中文测试文本，用来检验分词器对 CJK 字符的处理。"
    "안녕하세요, 여러분. 이 문장은 한국어로 작성되었습니다."
    "Привет мир! Это тестовый текст на русском языке. "
    "Γειά σας, κόσμε! Αυτό είναι ένα δείγμα κειμένου. "
    "مرحبا بالعالم. هذا نص تجريبي. "
    "🌍🚀💡📚🔥 ¿Hablas español? Oui, un peu. Ciao!\n"
)

RANDOM_BYTES = (
    "xzqv jklm wrpb dfgn hstc yeia oulv "
    "aXx9 Bz8Q wPnR tLm2 kJv5 cGh7 fDs3 "
    "qqq www eee rrr ttt yyy uuu iii ooo ppp "
    "asd fgh jkl zxc vbn mqq wwe rty uio pas "
    "123 456 789 0 12 345 67890 11223344556677 "
)


@dataclass
class Corpus:
    name: str
    text: str


CORPORA = [
    Corpus("pangrams (English prose)", PANGRAMS * 1000),
    Corpus("python source code", PYTHON_CODE * 500),
    Corpus("rust source code", RUST_CODE * 500),
    Corpus("multilingual", MULTILINGUAL * 800),
    Corpus("random-ish bytes", RANDOM_BYTES * 1200),
]


# --- Runner ------------------------------------------------------------------


def main() -> None:
    print("Loading o200k_base vocabulary...")
    ranks = load_tiktoken_bpe("o200k_base.tiktoken")
    rip = CoreBPE(ranks, SPECIAL_TOKENS, PAT_STR)
    tik = tiktoken.get_encoding("o200k_base")

    N = 10
    print(f"\nSingle-threaded benchmark: {N} runs per corpus. All outputs verified byte-identical.\n")

    header = f"{'corpus':<28} {'tokens':>10} {'rip (ms)':>10} {'tik (ms)':>10} {'rip tps':>14} {'tik tps':>14} {'speedup':>10}"
    print(header)
    print("-" * len(header))

    for c in CORPORA:
        # warm up
        rip_tokens = rip.encode_ordinary(c.text)
        tik_tokens = tik.encode_ordinary(c.text)
        if rip_tokens != tik_tokens:
            first_diff = next(
                (i for i, (a, b) in enumerate(zip(rip_tokens, tik_tokens)) if a != b),
                min(len(rip_tokens), len(tik_tokens)),
            )
            raise SystemExit(
                f"FAIL {c.name}: rip={len(rip_tokens)} tik={len(tik_tokens)} "
                f"first diff at token {first_diff}"
            )

        start = time.perf_counter()
        for _ in range(N):
            rip.encode_ordinary(c.text)
        rip_s = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(N):
            tik.encode_ordinary(c.text)
        tik_s = time.perf_counter() - start

        n_tok = len(rip_tokens)
        rip_tps = (n_tok * N) / rip_s
        tik_tps = (n_tok * N) / tik_s
        print(
            f"{c.name:<28} {n_tok:>10,} {rip_s * 100:>9.2f} {tik_s * 100:>9.2f} "
            f"{rip_tps:>14,.0f} {tik_tps:>14,.0f} {tik_s / rip_s:>9.2f}x"
        )

    # --- Parallel batch benchmark --------------------------------------------
    #
    # Simulates a bulk workload: many independent documents encoded at once.
    # riptoken uses its built-in `encode_ordinary_batch` which releases the
    # GIL and fans out to rayon's global thread pool. tiktoken's built-in
    # parallel path is `encode_ordinary_batch` too (threaded over a small
    # pool). Identical input, identical output.

    BATCH_SIZE = 256
    print(f"\nParallel batch benchmark: batch of {BATCH_SIZE} documents per corpus.\n")

    header = f"{'corpus':<28} {'tokens':>10} {'rip (ms)':>10} {'tik (ms)':>10} {'rip tps':>14} {'tik tps':>14} {'speedup':>10}"
    print(header)
    print("-" * len(header))

    for c in CORPORA:
        docs = [c.text] * BATCH_SIZE

        # Warm up + parity check on first doc.
        rip_batch = rip.encode_ordinary_batch(docs)
        assert all(d == rip_batch[0] for d in rip_batch)
        assert rip_batch[0] == tik.encode_ordinary(c.text)

        start = time.perf_counter()
        rip.encode_ordinary_batch(docs)
        rip_s = time.perf_counter() - start

        start = time.perf_counter()
        tik.encode_ordinary_batch(docs)
        tik_s = time.perf_counter() - start

        n_tok = len(rip_batch[0]) * BATCH_SIZE
        rip_tps = n_tok / rip_s
        tik_tps = n_tok / tik_s
        print(
            f"{c.name:<28} {n_tok:>10,} {rip_s * 1000:>9.2f} {tik_s * 1000:>9.2f} "
            f"{rip_tps:>14,.0f} {tik_tps:>14,.0f} {tik_s / rip_s:>9.2f}x"
        )


if __name__ == "__main__":
    main()
