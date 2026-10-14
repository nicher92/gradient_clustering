#!/usr/bin/env python3
"""Look at the actual examples behind the gradient clusters.

The numbers in analyze.py tell you a source owns 32 clusters. They do not tell
you what is in them. Three views, because they answer different questions:

  exclusive  clusters that are >=90% one source -- what that source uniquely
             contributes, i.e. what disappears if you drop it
  largest    the dominant modes -- what the mix is mostly made of
  outliers   singletons and near-singletons -- where malformed data tends to
             sit, since degenerate text produces unusual gradients

Clustering is reproducible: same gradients, same --seed, same labels as any
other run with those settings.

    python inspect_clusters.py \
        --dataset_filename data/excellent.jsonl \
        --gradient_storage_dir data/gradients/excellent \
        --view exclusive --source orca
"""
from __future__ import annotations

import argparse
import json
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm


# ------------------------------------------------------------------ loading

def load_gradients(storage_dir: Path, device: str) -> tuple[list[str], torch.Tensor]:
    files = sorted(storage_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files under {storage_dir}")

    merged: dict[str, torch.Tensor] = {}
    for path in tqdm(files, desc="loading gradients", leave=False):
        merged.update(load_file(str(path), device=device))

    ids = list(merged)
    return ids, torch.stack([merged[i] for i in ids], dim=0)


def load_rows(dataset_filename: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(dataset_filename) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


# ----------------------------------------------------------------- k-means

def cluster_kmeans(data: torch.Tensor, k: int, num_iter: int = 20,
                   seed: int = 0) -> torch.Tensor:
    data = F.normalize(data.float(), dim=1)
    k = min(k, data.shape[0])

    generator = torch.Generator(device=data.device).manual_seed(seed)
    perm = torch.randperm(data.shape[0], device=data.device, generator=generator)[:k]
    centroids = data[perm].clone()
    labels = torch.zeros(data.shape[0], dtype=torch.long, device=data.device)

    for _ in tqdm(range(num_iter), desc="k-means", leave=False):
        for start in range(0, data.shape[0], 4096):
            block = data[start : start + 4096]
            labels[start : start + 4096] = (block @ centroids.T).argmax(dim=1)
        for c in range(k):
            members = data[labels == c]
            if members.shape[0]:
                centroids[c] = members.sum(dim=0)
        centroids = F.normalize(centroids, dim=1)

    return labels


# ---------------------------------------------------------------- printing

def show(row: dict, width: int, chars: int) -> None:
    src = row.get("source", "?")
    prompt = " ".join((row.get("prompt") or "").split())
    completion = " ".join((row.get("completion") or "").split())

    print(f"    [{src}]")
    for label, text in (("prompt", prompt), ("reply ", completion)):
        body = text[:chars] + ("..." if len(text) > chars else "")
        wrapped = textwrap.wrap(body, width=width - 12) or [""]
        print(f"      {label}: {wrapped[0]}")
        for line in wrapped[1:]:
            print(f"              {line}")
    print()


def cluster_header(label: int, members: list[int], ids: list[str],
                   rows: dict[str, dict]) -> None:
    sources = Counter(
        (rows.get(ids[i], {}).get("source") or "?") for i in members
    )
    composition = ", ".join(f"{s} {n}" for s, n in sources.most_common())
    print(f"\n{'=' * 78}")
    print(f"cluster {label}  ({len(members)} members: {composition})")
    print("=" * 78)


# -------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_filename", type=Path, required=True)
    ap.add_argument("--gradient_storage_dir", type=Path, required=True)
    ap.add_argument("--view", default="all",
                    choices=["all", "exclusive", "largest", "outliers"])
    ap.add_argument("--source", default=None,
                    help="for --view exclusive: restrict to this source")
    ap.add_argument("--clusters", type=int, default=None, help="k; default n/10")
    ap.add_argument("--n-clusters-shown", type=int, default=5)
    ap.add_argument("--n-examples", type=int, default=3,
                    help="examples printed per cluster")
    ap.add_argument("--chars", type=int, default=400, help="truncate text at")
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--purity", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ids, gradients = load_gradients(args.gradient_storage_dir, args.device)
    rows = load_rows(args.dataset_filename)
    print(f"{len(ids)} gradients, {len(rows)} dataset rows")

    k = args.clusters or max(2, len(ids) // 10)
    labels = cluster_kmeans(gradients, k, seed=args.seed).cpu().tolist()

    members: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        members[label].append(idx)
    
    def source_of(i: int) -> str:
        return rows.get(ids[i], {}).get("source") or "?"
    
    # ---- exclusive ----------------------------------------------------
    if args.view in ("all", "exclusive"):
        owned: dict[str, list[tuple[int, list[int]]]] = defaultdict(list)
        for label, idxs in members.items():
            counts = Counter(source_of(i) for i in idxs)
            dominant, n_dom = counts.most_common(1)[0]
            if n_dom / len(idxs) >= args.purity and len(idxs) >= 3:
                owned[dominant].append((label, idxs))

        targets = [args.source] if args.source else sorted(
            owned, key=lambda s: -len(owned[s])
        )
        for src in targets:
            if src not in owned:
                print(f"\nno clusters >= {args.purity:.0%} {src}")
                continue
            print(f"\n\n{'#' * 78}")
            print(f"# EXCLUSIVE TO {src.upper()} -- {len(owned[src])} clusters "
                  f">= {args.purity:.0%} this source")
            print(f"{'#' * 78}")
            biggest = sorted(owned[src], key=lambda kv: -len(kv[1]))
            for label, idxs in biggest[: args.n_clusters_shown]:
                cluster_header(label, idxs, ids, rows)
                for i in idxs[: args.n_examples]:
                    show(rows.get(ids[i], {}), args.width, args.chars)

    # ---- largest ------------------------------------------------------
    if args.view in ("all", "largest"):
        print(f"\n\n{'#' * 78}")
        print("# LARGEST CLUSTERS -- the dominant modes of the mix")
        print(f"{'#' * 78}")
        for label, idxs in sorted(members.items(), key=lambda kv: -len(kv[1]))[
            : args.n_clusters_shown
        ]:
            cluster_header(label, idxs, ids, rows)
            for i in idxs[: args.n_examples]:
                show(rows.get(ids[i], {}), args.width, args.chars)

    # ---- outliers -----------------------------------------------------
    if args.view in ("all", "outliers"):
        print(f"\n\n{'#' * 78}")
        print("# SMALLEST CLUSTERS -- outliers; check these for malformed data")
        print(f"{'#' * 78}")
        smallest = sorted(members.items(), key=lambda kv: len(kv[1]))
        for label, idxs in smallest[: args.n_clusters_shown]:
            cluster_header(label, idxs, ids, rows)
            for i in idxs[: args.n_examples]:
                show(rows.get(ids[i], {}), args.width, args.chars)

        sizes = Counter(len(v) for v in members.values())
        singletons = sizes.get(1, 0)
        print(f"\n{singletons}/{len(members)} clusters are singletons "
              f"({singletons / len(members):.1%})")


if __name__ == "__main__":
    main()
