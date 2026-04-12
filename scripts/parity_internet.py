"""Byte-identical parity test: riptoken vs tiktoken on real-world internet text.

Downloads large, diverse corpora from the public internet and verifies that
riptoken produces exactly the same token sequence as tiktoken on every one.
"""

from __future__ import annotations

import sys
import time
import urllib.request

import tiktoken

import riptoken

SOURCES: list[tuple[str, str]] = [
    # --- Literature (Project Gutenberg, plain UTF-8) ---
    (
        "Moby Dick (1.2 MB)",
        "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
    ),
    (
        "War and Peace (3.3 MB)",
        "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
    ),
    (
        "Complete Shakespeare (5.5 MB)",
        "https://www.gutenberg.org/cache/epub/100/pg100.txt",
    ),
    (
        "Les Misérables — French (3.3 MB)",
        "https://www.gutenberg.org/cache/epub/17489/pg17489.txt",
    ),
    # --- Source code ---
    (
        "Linux kernel — sched/core.c (330 KB)",
        "https://raw.githubusercontent.com/torvalds/linux/master/kernel/sched/core.c",
    ),
    (
        "CPython — compile.c (450 KB)",
        "https://raw.githubusercontent.com/python/cpython/main/Python/compile.c",
    ),
    (
        "Rust compiler — parser.rs",
        "https://raw.githubusercontent.com/rust-lang/rust/master/compiler/rustc_parse/src/parser/mod.rs",
    ),
    # --- Multilingual (Wikipedia lead sections, HTML stripped) ---
    (
        "Wikipedia — Japan (日本語)",
        "https://ja.wikipedia.org/w/index.php?title=%E6%97%A5%E6%9C%AC&action=raw",
    ),
    (
        "Wikipedia — Russia (Русский)",
        "https://ru.wikipedia.org/w/index.php?title=%D0%A0%D0%BE%D1%81%D1%81%D0%B8%D1%8F&action=raw",
    ),
    (
        "Wikipedia — China (中文)",
        "https://zh.wikipedia.org/w/index.php?title=%E4%B8%AD%E5%9B%BD&action=raw",
    ),
    (
        "Wikipedia — Korea (한국어)",
        "https://ko.wikipedia.org/w/index.php?title=%EB%8C%80%ED%95%9C%EB%AF%BC%EA%B5%AD&action=raw",
    ),
    (
        "Wikipedia — Arabic (العربية)",
        "https://ar.wikipedia.org/w/index.php?title=%D8%A7%D9%84%D8%B9%D8%B1%D8%A8%D9%8A%D8%A9&action=raw",
    ),
]

ENCODINGS = ["o200k_base", "cl100k_base"]


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "riptoken-parity-test/0.2"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def main() -> None:
    total_tokens = 0
    total_bytes = 0
    failures: list[str] = []

    for enc_name in ENCODINGS:
        print(f"\n{'=' * 72}")
        print(f"  Encoding: {enc_name}")
        print(f"{'=' * 72}\n")

        rip = riptoken.get_encoding(enc_name)
        tik = tiktoken.get_encoding(enc_name)

        for label, url in SOURCES:
            sys.stdout.write(f"  {label:<45} ")
            sys.stdout.flush()

            try:
                text = fetch(url)
            except Exception as e:
                print(f"SKIP ({e})")
                continue

            size_kb = len(text.encode("utf-8")) / 1024

            t0 = time.perf_counter()
            rip_tokens = rip.encode_ordinary(text)
            rip_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            tik_tokens = tik.encode_ordinary(text)
            tik_ms = (time.perf_counter() - t0) * 1000

            n = len(rip_tokens)
            total_tokens += n
            total_bytes += len(text.encode("utf-8"))

            if rip_tokens == tik_tokens:
                speedup = tik_ms / rip_ms if rip_ms > 0 else float("inf")
                print(
                    f"OK  {size_kb:>7.0f} KB  {n:>8,} tok  "
                    f"rip {rip_ms:>7.1f} ms  tik {tik_ms:>7.1f} ms  "
                    f"{speedup:.1f}x"
                )
            else:
                # Find first divergence
                first = next(
                    (
                        i
                        for i, (a, b) in enumerate(zip(rip_tokens, tik_tokens))
                        if a != b
                    ),
                    min(len(rip_tokens), len(tik_tokens)),
                )
                print(
                    f"FAIL  rip={len(rip_tokens)} tik={len(tik_tokens)} "
                    f"first diff at token {first}"
                )
                failures.append(f"{enc_name}/{label}")

    print(f"\n{'=' * 72}")
    print(
        f"  Total: {total_bytes / 1024 / 1024:.1f} MB input, "
        f"{total_tokens:,} tokens across {len(ENCODINGS)} encodings"
    )

    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    - {f}")
        print(f"{'=' * 72}")
        sys.exit(1)
    else:
        print(f"  All parity checks passed.")
        print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
