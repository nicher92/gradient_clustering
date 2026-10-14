#!/usr/bin/env python3
"""G-Vendi diversity analysis over collected gradients.

Three things, in increasing order of usefulness:

1. Overall G-Vendi: exponentiated entropy of the gradient-space eigenspectrum.
   A single number, only meaningful compared against another mix measured the
   same way and at the same sample count.

2. Per-source G-Vendi: which component of the mix carries the variety.

3. Cluster composition: k-means in gradient space, then the source makeup of
   each cluster. This is the part that answers "what kind of update does mix A
   produce that mix B does not" -- clusters dominated by one source are the
   coverage that source uniquely provides.

    python analyze.py --dataset_filename data/excellent.jsonl \
                      --gradient_storage_dir data/gradients/excellent
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm


# ---------------------------------------------------------------- Vendi score

def entropy_q(p: np.ndarray, q: float = 1) -> float:
    p_ = p[p > 0]
    if q == 1:
        return float(-(p_ * np.log(p_)).sum())
    return float(np.log((p_ ** q).sum()) / (1 - q))


def vendi_score(gradients: torch.Tensor) -> float:
    """exp(entropy) of the eigenvalues of the normalised Gram matrix.

    Uses the (d x d) covariance rather than the (n x n) kernel: the nonzero
    eigenvalues coincide, and d = proj_dim is small and fixed while n is not.
    """
    g = F.normalize(gradients.float(), dim=1)
    n = g.shape[0]
    cov = (g.T @ g) / n                      # (proj_dim, proj_dim)
    eigenvalues = scipy.linalg.eigvalsh(cov.cpu().numpy())
    eigenvalues = np.clip(eigenvalues, 0, None)
    total = eigenvalues.sum()
    if total <= 0:
        return 0.0
    return float(np.exp(entropy_q(eigenvalues / total)))


# ------------------------------------------------------------------- k-means

def cluster_kmeans(data: torch.Tensor, k: int, num_iter: int = 20,
                   seed: int = 0) -> torch.Tensor:
    """Spherical k-means. Returns a label per row."""
    data = F.normalize(data.float(), dim=1)
    k = min(k, data.shape[0])

    generator = torch.Generator(device=data.device).manual_seed(seed)
    perm = torch.randperm(data.shape[0], device=data.device, generator=generator)[:k]
    centroids = data[perm].clone()
    labels = torch.zeros(data.shape[0], dtype=torch.long, device=data.device)

    for _ in tqdm(range(num_iter), desc="k-means", leave=False):
        labels = assign_labels(data, centroids)
        for c in range(k):
            members = data[labels == c]
            if members.shape[0]:
                centroids[c] = members.sum(dim=0)
        centroids = F.normalize(centroids, dim=1)

    return labels


def assign_labels(data: torch.Tensor, centroids: torch.Tensor,
                  chunk: int = 4096) -> torch.Tensor:
    """Nearest centroid by cosine similarity, chunked to bound memory."""
    out = torch.empty(data.shape[0], dtype=torch.long, device=data.device)
    for start in range(0, data.shape[0], chunk):
        block = data[start : start + chunk]
        out[start : start + chunk] = (block @ centroids.T).argmax(dim=1)
    return out


# ---------------------------------------------------------------------- I/O

def load_gradients(storage_dir: Path, device: str) -> tuple[list[str], torch.Tensor]:
    files = sorted(storage_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files under {storage_dir}")

    merged: dict[str, torch.Tensor] = {}
    for path in tqdm(files, desc="loading gradients"):
        merged.update(load_file(str(path), device=device))

    ids = list(merged)
    return ids, torch.stack([merged[i] for i in ids], dim=0)


def load_sources(dataset_filename: Path) -> dict[str, str]:
    sources: dict[str, str] = {}
    with open(dataset_filename) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            src = row.get("source") or "unknown"
            sources[row["id"]] = src
    return sources


# ---------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_filename", type=Path, required=True)
    ap.add_argument("--gradient_storage_dir", type=Path, required=True)
    ap.add_argument("--clusters", type=int, default=None,
                    help="k for k-means; defaults to n/10 as in the paper")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-source-n", type=int, default=200,
                    help="skip per-source scores below this many samples")
    ap.add_argument("--limit", type=int, default=None,
                    help="subsample to exactly this many gradients before scoring. "
                         "G-Vendi grows with n, so two mixes are only comparable "
                         "at equal n -- set this to the smaller run's count on both.")
    ap.add_argument("--per-source-n", type=int, default=None,
                    help="also report each source scored at exactly this many "
                         "samples, so per-source numbers are comparable to each "
                         "other and across mixes. Sources with fewer are skipped.")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for --limit subsampling and k-means init")
    args = ap.parse_args()

    ids, gradients = load_gradients(args.gradient_storage_dir, args.device)
    print(f"\n{len(ids)} gradients, dim {gradients.shape[1]}")

    if args.limit is not None and args.limit < len(ids):
        # Random rather than head-of-file: shards are written in collection
        # order, which follows the jsonl, which prepare_data.py filled one
        # source at a time. Truncating would skew the source mix.
        generator = torch.Generator().manual_seed(args.seed)
        pick = torch.randperm(len(ids), generator=generator)[: args.limit]
        ids = [ids[i] for i in pick.tolist()]
        gradients = gradients[pick.to(gradients.device)]
        print(f"subsampled to {len(ids)} (seed {args.seed})")
    elif args.limit is not None:
        print(f"--limit {args.limit} >= {len(ids)} available; using all")

    sources = load_sources(args.dataset_filename)
    missing = sum(1 for i in ids if i not in sources)
    if missing:
        print(f"warning: {missing} gradient ids absent from the dataset file")

    # -- 1. overall ------------------------------------------------------
    overall = vendi_score(gradients)
    print(f"\nG-Vendi (all): {overall:.2f}   "
          f"[1 = all identical, {gradients.shape[1]} = maximally spread]")

    # -- 2. per source ---------------------------------------------------
    by_source: dict[str, list[int]] = defaultdict(list)
    for idx, sample_id in enumerate(ids):
        by_source[sources.get(sample_id, "unknown")].append(idx)

    print(f"\n{'source':<20} {'n':>8} {'G-Vendi':>10}")
    for src, idxs in sorted(by_source.items(), key=lambda kv: -len(kv[1])):
        if len(idxs) < args.min_source_n:
            print(f"{src:<20} {len(idxs):>8} {'(too few)':>10}")
            continue
        score = vendi_score(gradients[torch.tensor(idxs, device=gradients.device)])
        print(f"{src:<20} {len(idxs):>8} {score:>10.2f}")

    # -- 2b. per source at matched n -------------------------------------
    # The scores above are at each source's own n, so a source with more rows
    # scores higher partly for that reason alone. Fixing n makes them
    # comparable to each other, and to the same source in another mix.
    if args.per_source_n:
        print(f"\nper-source at matched n={args.per_source_n}")
        print(f"{'source':<20} {'available':>10} {'G-Vendi':>10}")
        generator = torch.Generator().manual_seed(args.seed)
        for src, idxs in sorted(by_source.items(), key=lambda kv: -len(kv[1])):
            if len(idxs) < args.per_source_n:
                print(f"{src:<20} {len(idxs):>10} {'(too few)':>10}")
                continue
            pick = torch.randperm(len(idxs), generator=generator)[: args.per_source_n]
            chosen = torch.tensor([idxs[i] for i in pick.tolist()],
                                  device=gradients.device)
            print(f"{src:<20} {len(idxs):>10} {vendi_score(gradients[chosen]):>10.2f}")

    # -- 3. cluster composition -----------------------------------------
    k = args.clusters or max(2, len(ids) // 10)
    print(f"\nclustering into k={k}")
    labels = cluster_kmeans(gradients, k, seed=args.seed).cpu().numpy()

    composition: dict[int, Counter] = defaultdict(Counter)
    for sample_id, label in zip(ids, labels):
        composition[int(label)][sources.get(sample_id, "unknown")] += 1

    sizes = Counter(labels.tolist())
    print(f"cluster sizes: median {np.median(list(sizes.values())):.0f}, "
          f"max {max(sizes.values())}, empty {k - len(sizes)}")

    # Clusters owned by a single source are that source's unique contribution.
    purity_by_source: Counter = Counter()
    pure_clusters = 0
    for label, counts in composition.items():
        dominant, n_dominant = counts.most_common(1)[0]
        if n_dominant / sum(counts.values()) >= 0.9:
            pure_clusters += 1
            purity_by_source[dominant] += 1

    print(f"\n{pure_clusters}/{len(composition)} clusters are >=90% one source")
    print(f"{'source':<20} {'owned clusters':>16}")
    for src, n in purity_by_source.most_common():
        print(f"{src:<20} {n:>16}")

    print("\nTo compare two mixes: run this on each with the same --limit "
          "(the smaller run's gradient count) and the same --per-source-n. "
          "G-Vendi grows with n, and k defaults to n/10, so unequal samples "
          "make both the scores and the cluster counts incomparable.")


if __name__ == "__main__":
    main()
