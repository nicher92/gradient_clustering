#!/usr/bin/env python3
"""Convert OpenEuroLLM-Excellent-SFT into the {id, prompt, completion, source}
jsonl that the G-Vendi gradient collector expects.

Streams from the Hub so nothing large is materialised locally. Excellent-SFT is
ordered in blocks by source (nemotron, smoltalk2, dolci, perfectblend, orca), so
a naive head-of-file sample is badly biased -- use --per-source to take a fixed
number from each instead.

    python prepare_data.py --per-source 20000 --out data/excellent.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

REPO = "openeurollm/OpenEuroLLM-Excellent-SFT"

# Row counts per source, from the dataset card. Used only to report coverage.
CARD_COUNTS = {
    "nemotron": 3_755_227,
    "smoltalk2": 904_866,
    "dolci": 564_958,
    "open-perfectblend": 533_311,
    "orca": 429_354,
}


def to_prompt_completion(row: dict) -> dict | None:
    """First user turn -> prompt, first assistant turn -> completion.

    Excellent-SFT rows are single-turn (messages listlength is always 2), so
    this is a straight split. Returns None for anything that does not match
    that shape rather than guessing.
    """
    messages = row.get("messages") or []
    if len(messages) < 2:
        return None
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        return None

    prompt = (messages[0].get("content") or "").strip()
    completion = (messages[1].get("content") or "").strip()
    if not prompt or not completion:
        return None

    return {
        "id": row["id"],
        "prompt": prompt,
        "completion": completion,
        "source": row.get("source") or "unknown",
    }


def source_family(source: str) -> str:
    """'nemotron/chat' -> 'nemotron'. The card reports counts per family."""
    return source.split("/")[0] if source else "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=20_000,
                    help="rows to keep per source family")
    ap.add_argument("--total", type=int, default=None,
                    help="alternative to --per-source: take this many rows from "
                         "the head of the file (biased, for quick tests only)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle-buffer", type=int, default=10_000,
                    help="streaming shuffle buffer; 0 disables shuffling")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(REPO, split="train", streaming=True)
    if args.shuffle_buffer:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    kept = Counter()
    skipped = 0
    written = 0

    with open(args.out, "w") as f:
        for row in ds:
            if args.total is not None and written >= args.total:
                break

            converted = to_prompt_completion(row)
            if converted is None:
                skipped += 1
                continue

            family = source_family(converted["source"])

            if args.total is None:
                if kept[family] >= args.per_source:
                    # Stop once every known family is full. Unknown families are
                    # capped too but do not hold the loop open.
                    if all(kept[s] >= args.per_source for s in CARD_COUNTS):
                        break
                    continue

            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            kept[family] += 1
            written += 1

            if written % 10_000 == 0:
                print(f"{written} written: {dict(kept)}", flush=True)

    print(f"\nwrote {written} rows to {args.out} ({skipped} skipped as malformed)")
    print(f"{'source':<20} {'kept':>8} {'of total':>12} {'coverage':>10}")
    for family, n in sorted(kept.items(), key=lambda kv: -kv[1]):
        total = CARD_COUNTS.get(family)
        if total:
            print(f"{family:<20} {n:>8} {total:>12,} {n / total:>9.2%}")
        else:
            print(f"{family:<20} {n:>8} {'?':>12} {'?':>10}")


if __name__ == "__main__":
    main()
