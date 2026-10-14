#!/usr/bin/env python3
"""Rank training data by gradient similarity to a target task.

The idea is from LESS (Xia et al., ICML 2024): training examples whose gradients
point the same way as a target task's are the ones that move the weights toward
doing that task well. Here the target is IFEval-style instruction following, and
the question is whether the sources you suspect matter actually rank highest.

Two scores per training example:

  max    cosine to the single closest target. Asks "is this example near ANY
         target", which is the robust question when the targets are varied
  mean   cosine to the averaged target direction. Only meaningful when the
         targets resemble each other: averaging 500 mutually dissimilar
         gradients gives a near-arbitrary vector, and cosine against it is
         noise. Observed in practice -- on one mix every source but one scored
         negative on mean and the outlier was unrelated maths, while max
         correctly showed nothing in that mix was close to the targets

Which column leads is chosen from the target set's internal cosine, not by the
reader, because the failure above is easy to miss: the ordering looks decisive
either way.

Both are reported per source, since a source-level ranking is what informs a
mixing decision.

WHAT THIS IS NOT. LESS uses Adam-adjusted gradients taken from warmup
checkpoints of the actual training run, because influence depends on how the
optimiser will move the weights, not on the raw gradient. These are raw
gradients from an off-the-shelf proxy model. That makes this a similarity
ranking rather than an influence estimate -- directionally useful, but it does
not license claims about what training on a given example would do.

    python target_similarity.py \
        --target data/gradients/target_if --target-jsonl data/target_if.jsonl \
        --train data/gradients/tulu --train-jsonl data/tulu_mix.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm


def load_store(storage_dir: Path, device: str, label: str) -> tuple[list[str], torch.Tensor]:
    files = sorted(storage_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors under {storage_dir}")

    merged: dict[str, torch.Tensor] = {}
    for path in tqdm(files, desc=f"loading {label}", leave=False):
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


def similarities(train: torch.Tensor, target: torch.Tensor,
                 chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine of each training row against the target set: (mean-direction, max)."""
    train = F.normalize(train.float(), dim=1)
    target = F.normalize(target.float(), dim=1)
    direction = F.normalize(target.mean(dim=0, keepdim=True), dim=1)

    to_mean = torch.empty(train.shape[0], device=train.device)
    to_max = torch.empty(train.shape[0], device=train.device)
    for start in range(0, train.shape[0], chunk):
        block = train[start : start + chunk]
        to_mean[start : start + chunk] = (block @ direction.T).squeeze(1)
        to_max[start : start + chunk] = (block @ target.T).max(dim=1).values
    return to_mean, to_max


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, required=True,
                    help="gradient store of the target task")
    ap.add_argument("--target-jsonl", type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True,
                    help="gradient store of the candidate training data")
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--top", type=int, default=15,
                    help="example training rows to print from the top of the ranking")
    ap.add_argument("--chars", type=int, default=160)
    ap.add_argument("--cohesion-threshold", type=float, default=0.1,
                    help="median internal cosine below which the target set is "
                         "treated as diffuse and ranking switches to max")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-scores", type=Path, default=None,
                    help="write per-example scores to this jsonl")
    args = ap.parse_args()

    target_ids, target = load_store(args.target, args.device, "target")
    train_ids, train = load_store(args.train, args.device, "train")
    train_rows = load_rows(args.train_jsonl)

    print(f"\ntarget {len(target_ids)} gradients, train {len(train_ids)}, "
          f"dim {train.shape[1]}")

    # How tight is the target set? If its examples barely agree with each other,
    # the averaged direction is not meaningful and only `max` should be read.
    target_norm = F.normalize(target.float(), dim=1)
    internal = (target_norm @ target_norm.T)
    off_diagonal = internal[~torch.eye(len(target_ids), dtype=torch.bool,
                                       device=internal.device)]
    print(f"target internal cosine: median {off_diagonal.median():.3f}, "
          f"mean {off_diagonal.mean():.3f}")
    diffuse = float(off_diagonal.median()) < args.cohesion_threshold
    if diffuse:
        print(f"  below the {args.cohesion_threshold} cohesion threshold: the "
              f"targets do not resemble each other, so their average is close "
              f"to nothing in particular. Ranking by max; mean is shown but "
              f"should not be read.")
    else:
        print(f"  at or above the {args.cohesion_threshold} cohesion threshold: "
              f"the averaged direction is meaningful. Ranking by mean.")

    to_mean, to_max = similarities(train, target)
    primary, primary_name = (to_max, "max") if diffuse else (to_mean, "mean")

    # -- per source ------------------------------------------------------
    by_source: dict[str, list[int]] = defaultdict(list)
    for idx, sample_id in enumerate(train_ids):
        by_source[train_rows.get(sample_id, {}).get("source", "unknown")].append(idx)

    mean_header = "mean" if not diffuse else "mean*"
    max_header = "max" if diffuse else "max*"
    print(f"\n{'source':<62} {'n':>6} {mean_header:>8} {max_header:>8}"
          f"   (* not the ranking column)")
    ranked = sorted(
        by_source.items(),
        key=lambda kv: -primary[torch.tensor(kv[1], device=train.device)].mean(),
    )
    for source, idxs in ranked:
        sel = torch.tensor(idxs, device=train.device)
        print(f"{source[:62]:<62} {len(idxs):>6} "
              f"{to_mean[sel].mean():>8.4f} {to_max[sel].mean():>8.4f}")

    # -- top individual examples ----------------------------------------
    order = torch.argsort(primary, descending=True)[: args.top]
    print(f"\ntop {args.top} training examples by {primary_name} similarity "
          f"to the target")
    for rank, idx in enumerate(order.cpu().tolist(), 1):
        row = train_rows.get(train_ids[idx], {})
        prompt = " ".join((row.get("prompt") or "").split())[: args.chars]
        print(f"\n{rank:>3}. {primary[idx]:.4f}  [{row.get('source', '?')[:50]}]")
        print(f"     {prompt}")

    if args.save_scores:
        with open(args.save_scores, "w") as f:
            for idx, sample_id in enumerate(train_ids):
                f.write(json.dumps({
                    "id": sample_id,
                    "source": train_rows.get(sample_id, {}).get("source", "unknown"),
                    "cos_mean": float(to_mean[idx]),
                    "cos_max": float(to_max[idx]),
                }) + "\n")
        print(f"\nwrote per-example scores to {args.save_scores}")

    print(f"\nRanked by {primary_name} (target internal cosine "
          f"{off_diagonal.median():.3f}).")
    print("Raw-gradient similarity, not influence: LESS uses Adam-adjusted "
          "gradients from warmup checkpoints of the real run. Read this as a "
          "ranking to investigate, not an estimate of training effect.")


if __name__ == "__main__":
    main()
