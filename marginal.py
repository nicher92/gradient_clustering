#!/usr/bin/env python3
"""Does adding a candidate dataset buy coverage the mix does not already have?

analyze.py scores one mix. The decision question is marginal: not "what does X
score" but "what does the mix score with X versus without". A candidate can be
diverse in isolation and still be redundant, if everything it covers is already
covered.

Four numbers, in decreasing order of how much they should drive the decision:

  new clusters      regions of gradient space the base mix never reaches. The
                    strongest signal -- these vanish if you drop the candidate
  absorbed          candidate rows landing in clusters the base already owns.
                    High absorption means redundancy, whatever its own score
  delta G-Vendi     mix vs mix+candidate at matched n. A small delta on a
                    candidate that is large relative to the mix means it is
                    adding volume rather than variety
  isolation         mean cosine distance from each candidate row to its nearest
                    base-mix neighbour. Scale-free, so comparable across runs

Every gradient store compared here must have been collected with the same proxy
model, --max_length, --dtype, projector and seed. Different settings give
different projections and the numbers are then meaningless rather than wrong.

    python marginal.py \
        --base data/gradients/excellent --base-jsonl data/excellent.jsonl \
        --candidate data/gradients/tulu --candidate-jsonl data/tulu_mix.jsonl
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


# ---------------------------------------------------------------- vendi

def vendi_score(gradients: torch.Tensor) -> float:
    """exp(entropy) of the normalised Gram eigenspectrum: an effective count
    of the directions the gradients span."""
    g = F.normalize(gradients.float(), dim=1)
    cov = (g.T @ g) / g.shape[0]
    eigenvalues = np.clip(scipy.linalg.eigvalsh(cov.cpu().numpy()), 0, None)
    total = eigenvalues.sum()
    if total <= 0:
        return 0.0
    p = eigenvalues / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


# ----------------------------------------------------------------- i/o

def load_store(storage_dir: Path, device: str, label: str) -> tuple[list[str], torch.Tensor]:
    files = sorted(storage_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors under {storage_dir}")

    merged: dict[str, torch.Tensor] = {}
    for path in tqdm(files, desc=f"loading {label}", leave=False):
        merged.update(load_file(str(path), device=device))

    ids = list(merged)
    return ids, torch.stack([merged[i] for i in ids], dim=0)


def load_sources(dataset_filename: Path) -> dict[str, str]:
    sources: dict[str, str] = {}
    with open(dataset_filename) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                sources[row["id"]] = row.get("source") or "unknown"
    return sources


def subsample(ids: list[str], gradients: torch.Tensor, n: int,
              seed: int) -> tuple[list[str], torch.Tensor]:
    if n >= len(ids):
        return ids, gradients
    generator = torch.Generator().manual_seed(seed)
    pick = torch.randperm(len(ids), generator=generator)[:n]
    return [ids[i] for i in pick.tolist()], gradients[pick.to(gradients.device)]


# -------------------------------------------------------------- k-means

def fit_centroids(data: torch.Tensor, k: int, num_iter: int = 20,
                  seed: int = 0) -> torch.Tensor:
    """Spherical k-means on the base mix. Returns unit-norm centroids."""
    data = F.normalize(data.float(), dim=1)
    k = min(k, data.shape[0])

    generator = torch.Generator(device=data.device).manual_seed(seed)
    perm = torch.randperm(data.shape[0], device=data.device, generator=generator)[:k]
    centroids = data[perm].clone()

    for _ in tqdm(range(num_iter), desc="k-means", leave=False):
        labels = assign(data, centroids)
        for c in range(k):
            members = data[labels == c]
            if members.shape[0]:
                centroids[c] = members.sum(dim=0)
        centroids = F.normalize(centroids, dim=1)

    return centroids


def assign(data: torch.Tensor, centroids: torch.Tensor,
           chunk: int = 4096) -> torch.Tensor:
    out = torch.empty(data.shape[0], dtype=torch.long, device=data.device)
    for start in range(0, data.shape[0], chunk):
        block = data[start : start + chunk]
        out[start : start + chunk] = (block @ centroids.T).argmax(dim=1)
    return out


def nearest_similarity(query: torch.Tensor, reference: torch.Tensor,
                       chunk: int = 2048) -> torch.Tensor:
    """Cosine similarity from each query row to its closest reference row."""
    query = F.normalize(query.float(), dim=1)
    reference = F.normalize(reference.float(), dim=1)
    best = torch.empty(query.shape[0], device=query.device)
    for start in range(0, query.shape[0], chunk):
        block = query[start : start + chunk]
        best[start : start + chunk] = (block @ reference.T).max(dim=1).values
    return best


# ----------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True,
                    help="gradient store of the mix you already have")
    ap.add_argument("--base-jsonl", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True,
                    help="gradient store of the dataset you are considering")
    ap.add_argument("--candidate-jsonl", type=Path, required=True)
    ap.add_argument("--n", type=int, default=None,
                    help="subsample both to this many rows; defaults to the "
                         "smaller store, since G-Vendi grows with n")
    ap.add_argument("--clusters", type=int, default=None,
                    help="k for the base-mix clustering; defaults to n/10")
    ap.add_argument("--new-cluster-threshold", type=float, default=None,
                    help="cosine below which a candidate row counts as outside "
                         "the base mix. Default: the base mix's own 10th "
                         "percentile, i.e. as far out as its most unusual tenth. "
                         "A fixed 0.5 flagged 89%% of rows, because centroids "
                         "sit further from points than points do from each other")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    base_ids, base = load_store(args.base, args.device, "base")
    cand_ids, cand = load_store(args.candidate, args.device, "candidate")
    base_sources = load_sources(args.base_jsonl)
    cand_sources = load_sources(args.candidate_jsonl)

    n = args.n or min(len(base_ids), len(cand_ids))
    base_ids, base = subsample(base_ids, base, n, args.seed)
    cand_ids, cand = subsample(cand_ids, cand, n, args.seed + 1)

    print(f"\nbase      {args.base.name:<20} {len(base_ids):>7} gradients")
    print(f"candidate {args.candidate.name:<20} {len(cand_ids):>7} gradients")
    print(f"compared at n={n}, dim {base.shape[1]}")

    # -- 1. delta G-Vendi -------------------------------------------------
    # The union is subsampled back to n so the comparison is not confounded by
    # simply having twice the rows.
    union = torch.cat([base, cand], dim=0)
    union_ids = base_ids + cand_ids
    union_ids, union = subsample(union_ids, union, n, args.seed + 2)

    v_base = vendi_score(base)
    v_cand = vendi_score(cand)
    v_union = vendi_score(union)

    print(f"\n{'':<24} {'G-Vendi':>9}")
    print(f"{'base alone':<24} {v_base:>9.2f}")
    print(f"{'candidate alone':<24} {v_cand:>9.2f}")
    print(f"{'base + candidate':<24} {v_union:>9.2f}   "
          f"({v_union - v_base:+.2f} vs base)")

    # -- 2. coverage ------------------------------------------------------
    k = args.clusters or max(2, n // 10)
    print(f"\nclustering the base mix into k={k}")
    centroids = fit_centroids(base, k, seed=args.seed)

    # How close is a base row to its own centroid? That sets the scale: a
    # candidate row is only "outside" if it is further out than the base mix's
    # own tail. Without this the threshold is arbitrary and uninterpretable.
    base_best = (F.normalize(base.float(), dim=1) @ centroids.T).max(dim=1).values
    print(f"base rows to own centroid: median {base_best.median():.3f}, "
          f"10th pct {base_best.quantile(0.10):.3f}")

    threshold = args.new_cluster_threshold
    if threshold is None:
        threshold = float(base_best.quantile(0.10))
        print(f"threshold (base 10th percentile): {threshold:.3f}")

    base_labels = assign(F.normalize(base.float(), dim=1), centroids)
    cand_norm = F.normalize(cand.float(), dim=1)
    similarity = (cand_norm @ centroids.T)
    best_sim, cand_labels = similarity.max(dim=1)

    outside = best_sim < threshold
    n_outside = int(outside.sum())

    print(f"\n{n_outside}/{len(cand_ids)} candidate rows "
          f"({n_outside / len(cand_ids):.1%}) sit outside every base cluster "
          f"(cosine < {threshold:.3f})")

    touched = set(cand_labels.cpu().tolist())
    base_occupied = set(base_labels.cpu().tolist())
    print(f"{len(touched)}/{len(base_occupied)} base clusters receive "
          f"candidate rows; {len(base_occupied - touched)} are untouched")

    # -- 3. which candidate sources are new -------------------------------
    by_source_outside: Counter = Counter()
    by_source_total: Counter = Counter()
    for sample_id, is_outside in zip(cand_ids, outside.cpu().tolist()):
        source = cand_sources.get(sample_id, "unknown")
        by_source_total[source] += 1
        if is_outside:
            by_source_outside[source] += 1

    print(f"\ncandidate sources by share of rows outside the base mix")
    print(f"{'source':<62} {'n':>6} {'outside':>9}")
    rows = sorted(
        by_source_total.items(),
        key=lambda kv: -(by_source_outside[kv[0]] / kv[1]),
    )
    for source, total in rows:
        share = by_source_outside[source] / total
        print(f"{source[:62]:<62} {total:>6} {share:>8.1%}")

    # -- 4. isolation -----------------------------------------------------
    # Nearest-neighbour rather than nearest-centroid: a row can be far from
    # every centroid yet sit right beside an individual base example.
    nn_sim = nearest_similarity(cand, base)
    print(f"\nnearest base neighbour, cosine similarity:")
    print(f"  median {nn_sim.median():.3f}   "
          f"mean {nn_sim.mean():.3f}   "
          f"10th pct {nn_sim.quantile(0.10):.3f}")
    print("  (lower means the candidate sits further from anything in the base "
          "mix; compare across candidates rather than reading absolutely)")

    print("\nReading this: a candidate worth adding has rows outside the base "
          "clusters and a low nearest-neighbour similarity. A high G-VendI "
          "alone is not enough -- a dataset can be internally varied and still "
          "duplicate ground the mix already covers.")


if __name__ == "__main__":
    main()
