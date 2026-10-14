#!/usr/bin/env python3
"""Convert the Tulu-3-based mix into the jsonl the gradient collector expects.

This is the mixture behind ckpt-4000 (IFEval 68.8), as opposed to
OpenEuroLLM-Excellent-SFT which is behind ckpt-3626 (IFEval 58.4). Sampling both
the same way and measuring G-Vendi on each is the cheap test of whether that gap
shows up as a coverage difference.

The mix, as trained:

    tulu-3-sft-mixture (commercial subset)    920,552   69.10%
    EuroBlocks-SFT-Synthetic-1124 (subset)    161,644   12.13%
    Nemotron-Post-Training-v2 math (head)     250,000   18.77%

Three schemas, so three converters:

    tulu        id / messages[{content, role}] / source
    euroblocks  conversations[{from, value}]; its own `messages` column is a
                float turn count and is discarded
    nemotron    messages[{role, content}]; a local arrow directory, split=math

SAMPLING. tulu-3-sft-mixture is stored grouped by source, so a streaming
shuffle with a bounded buffer never reaches the later ones: a first attempt at
this with --shuffle-buffer 10000 returned five sources out of ~18 and missed
`personahub_if` entirely. This version instead walks the whole stream and keeps
a per-source quota, which is slower but is the only way to see a source that is
3% of the data. Pass --probe to count sources without writing anything.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

from datasets import load_dataset

TULU_REPO = "allenai/tulu-3-sft-mixture"
EUROBLOCKS_REPO = "utter-project/EuroBlocks-SFT-Synthetic-1124"

# tulu-3-sft-mixture is partly non-commercial. The run this reproduces used the
# commercial subset (920,552 of 939,344 rows). Drop the non-commercial parts
# unless --keep-noncommercial is passed.
NONCOMMERCIAL_SOURCE_MARKERS = ("wildchat",)


def from_messages(row: dict, source: str | None = None) -> dict | None:
    """First user turn -> prompt, first assistant turn -> completion.

    Tulu rows run to 294 messages, so this takes the opening exchange rather
    than the whole conversation. Same rule as the Excellent-SFT converter, which
    keeps the two measurements comparable; it does mean long conversations are
    represented by their first turn only.
    """
    messages = row.get("messages") or []
    user = next((m for m in messages if m.get("role") == "user"), None)
    assistant = next((m for m in messages if m.get("role") == "assistant"), None)
    if user is None or assistant is None:
        return None

    prompt = (user.get("content") or "").strip()
    completion = (assistant.get("content") or "").strip()
    if not prompt or not completion:
        return None

    return {
        "id": str(row.get("id") or ""),
        "prompt": prompt,
        "completion": completion,
        "source": source or row.get("source") or "unknown",
    }


def from_conversations(row: dict, source: str) -> dict | None:
    """EuroBlocks: conversations[{from, value}] -> prompt/completion."""
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    prompt = completion = None
    for turn in row.get("conversations") or []:
        role = role_map.get(turn.get("from"))
        if role == "user" and prompt is None:
            prompt = (turn.get("value") or "").strip()
        elif role == "assistant" and prompt is not None:
            completion = (turn.get("value") or "").strip()
            break

    if not prompt or not completion:
        return None
    return {"id": str(row.get("id") or ""), "prompt": prompt,
            "completion": completion, "source": source}


def is_commercial(source: str) -> bool:
    lowered = (source or "").lower()
    return not any(marker in lowered for marker in NONCOMMERCIAL_SOURCE_MARKERS)


def probe_sources(limit: int | None) -> Counter:
    """Count rows per source by streaming, without converting or writing.

    Worth running first: it tells you how many sources there are and how rare
    the smallest is, which is what sets a sensible --per-source quota.
    """
    counts: Counter = Counter()
    stream = load_dataset(TULU_REPO, split="train", streaming=True)
    for i, row in enumerate(stream):
        if limit is not None and i >= limit:
            break
        counts[row.get("source") or "unknown"] += 1
        if (i + 1) % 100_000 == 0:
            print(f"  {i + 1:,} rows, {len(counts)} sources so far", flush=True)
    return counts


def round_robin(rows: list[dict]) -> list[dict]:
    """Reorder so sources alternate: one from each, then the next from each.

    Any prefix of the result is then roughly proportional to the whole, so a
    gradient run that is stopped early still has every source represented.
    """
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)

    ordered: list[dict] = []
    for group in zip_longest(*by_source.values()):
        ordered.extend(row for row in group if row is not None)
    return ordered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--per-source", type=int, default=1500,
                    help="rows to keep per Tulu source; the whole stream is "
                         "walked, so rare sources are reached")
    ap.add_argument("--euroblocks", type=int, default=10_000,
                    help="rows from EuroBlocks (a single source)")
    ap.add_argument("--nemotron", type=int, default=0,
                    help="rows from the Nemotron math split; needs --nemotron-path")
    ap.add_argument("--nemotron-path", default=None)
    ap.add_argument("--keep-noncommercial", action="store_true")
    ap.add_argument("--probe", action="store_true",
                    help="just count sources and exit")
    ap.add_argument("--probe-limit", type=int, default=None,
                    help="stop the probe after this many rows")
    ap.add_argument("--max-scan", type=int, default=None,
                    help="give up after scanning this many Tulu rows")
    args = ap.parse_args()

    if args.probe:
        counts = probe_sources(args.probe_limit)
        total = sum(counts.values())
        print(f"\n{total:,} rows, {len(counts)} sources\n")
        print(f"{'source':<70} {'rows':>9} {'share':>8}")
        for source, n in counts.most_common():
            print(f"{source[:70]:<70} {n:>9,} {n / total:>7.2%}")
        return

    if args.out is None:
        ap.error("--out is required unless --probe is given")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    kept: Counter = Counter()
    skipped = 0

    # ---- Tulu: per-source quotas over the full stream ---------------------
    print(f"tulu-3-sft-mixture (quota {args.per_source}/source, full scan)...")
    stream = load_dataset(TULU_REPO, split="train", streaming=True)
    for i, row in enumerate(stream):
        if args.max_scan is not None and i >= args.max_scan:
            print(f"  stopped at --max-scan {args.max_scan:,}")
            break

        source = row.get("source") or "unknown"
        if kept[source] >= args.per_source:
            continue
        if not args.keep_noncommercial and not is_commercial(source):
            continue

        converted = from_messages(row)
        if converted is None:
            skipped += 1
            continue

        converted["id"] = f"tulu:{converted['id'] or i}"
        converted["component"] = "tulu"
        rows.append(converted)
        kept[source] += 1

        if (i + 1) % 100_000 == 0:
            print(f"  scanned {i + 1:,}, kept {len(rows)}, "
                  f"{len(kept)} sources", flush=True)

    print(f"  kept {len(rows)} rows across {len(kept)} sources "
          f"({skipped} malformed)")

    # ---- EuroBlocks -------------------------------------------------------
    if args.euroblocks:
        print(f"EuroBlocks-SFT-Synthetic-1124 ({args.euroblocks})...")
        euro = load_dataset(EUROBLOCKS_REPO, split="train", streaming=True)
        n_euro = 0
        for i, row in enumerate(euro):
            if n_euro >= args.euroblocks:
                break
            converted = from_conversations(row, "euroblocks")
            if converted is None:
                continue
            converted["id"] = f"euroblocks:{converted['id'] or i}"
            converted["component"] = "euroblocks"
            rows.append(converted)
            n_euro += 1
        print(f"  kept {n_euro}")

    # ---- Nemotron math ----------------------------------------------------
    if args.nemotron and args.nemotron_path:
        print(f"Nemotron math ({args.nemotron})...")
        nemo = load_dataset(args.nemotron_path, split="math", streaming=True)
        n_nemo = 0
        for i, row in enumerate(nemo):
            if n_nemo >= args.nemotron:
                break
            converted = from_messages(row, "nemotron/math")
            if converted is None:
                continue
            converted["id"] = f"nemotron:{converted['id'] or i}"
            converted["component"] = "nemotron"
            rows.append(converted)
            n_nemo += 1
        print(f"  kept {n_nemo}")

    # Interleave so a partial gradient run still covers every source. The
    # collector processes the file in order, and a first attempt here wrote
    # all Tulu then all EuroBlocks -- the first 2,500 gradients were Tulu only,
    # which made the comparison meaningless until the run finished.
    rows = round_robin(rows)

    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(rows)} rows to {args.out}")
    by_component = Counter(r["component"] for r in rows)
    print(f"\n{'component':<14} {'rows':>8} {'share':>8}")
    for component, n in by_component.most_common():
        print(f"{component:<14} {n:>8} {n / len(rows):>7.1%}")

    print(f"\n{'source':<70} {'rows':>8}")
    for source, n in Counter(r["source"] for r in rows).most_common():
        print(f"{source[:70]:<70} {n:>8}")



if __name__ == "__main__":
    main()
