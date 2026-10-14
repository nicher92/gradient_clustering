#!/usr/bin/env python3
"""Compare two sources within a surface-homogeneous stratum.

Across whole mixes, gradient space partitions first on surface form -- language,
markdown style, answer format, programming language. That makes a lot of the
novelty signal reducible to things the dataset cards already tell you: EuroBlocks
looks novel against an English mix partly because it is in other languages.

The signal that is not available any other way is what remains once surface form
is held constant. Two English grade-school maths sources with the same `\\boxed{}`
answer format look identical by any metadata you have. If gradient space still
separates them, it is separating them on problem structure.

So: filter both sides to one stratum, then ask the same coverage questions.

    python stratified.py \\
        --a data/gradients/excellent --a-jsonl data/excellent.jsonl \\
        --a-source nemotron/math \\
        --b data/gradients/excellent --b-jsonl data/excellent.jsonl \\
        --b-source opb/train

Both sides may come from the same store, as above -- the point is the slice, not
the file. Interpreting the result:

  separation near zero   gradient space sees only surface form here, and the
                         tool adds nothing over metadata for same-surface data
  clear separation       it is picking up structure, and stratified comparison
                         is the setting where it earns its cost
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm


def vendi_score(gradients: torch.Tensor) -> float:
    g = F.normalize(gradients.float(), dim=1)
    cov = (g.T @ g) / g.shape[0]
    eigenvalues = np.clip(scipy.linalg.eigvalsh(cov.cpu().numpy()), 0, None)
    total = eigenvalues.sum()
    if total <= 0:
        return 0.0
    p = eigenvalues / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def load_slice(storage_dir: Path, dataset_filename: Path, source: str,
               device: str, label: str) -> tuple[list[str], torch.Tensor]:
    """Load only the gradients whose source matches (prefix match)."""
    wanted: set[str] = set()
    with open(dataset_filename) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if (row.get("source") or "").startswith(source):
                wanted.add(row["id"])
    if not wanted:
        raise SystemExit(f"no rows in {dataset_filename} with source prefix '{source}'")

    merged: dict[str, torch.Tensor] = {}
    for path in tqdm(sorted(storage_dir.glob("*.safetensors")),
                     desc=f"loading {label}", leave=False):
        for key, value in load_file(str(path), device=device).items():
            if key in wanted:
                merged[key] = value

    if not merged:
        raise SystemExit(f"'{source}' matched {len(wanted)} dataset rows but no "
                         f"gradients in {storage_dir} -- has it been collected?")

    ids = list(merged)
    return ids, torch.stack([merged[i] for i in ids], dim=0)


def subsample(ids: list[str], gradients: torch.Tensor, n: int,
              seed: int) -> tuple[list[str], torch.Tensor]:
    if n >= len(ids):
        return ids, gradients
    generator = torch.Generator().manual_seed(seed)
    pick = torch.randperm(len(ids), generator=generator)[:n]
    return [ids[i] for i in pick.tolist()], gradients[pick.to(gradients.device)]


def nearest(query: torch.Tensor, reference: torch.Tensor,
            chunk: int = 2048) -> torch.Tensor:
    query = F.normalize(query.float(), dim=1)
    reference = F.normalize(reference.float(), dim=1)
    best = torch.empty(query.shape[0], device=query.device)
    for start in range(0, query.shape[0], chunk):
        block = query[start : start + chunk]
        best[start : start + chunk] = (block @ reference.T).max(dim=1).values
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--a-jsonl", type=Path, required=True)
    ap.add_argument("--a-source", required=True, help="source prefix for side A")
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--b-jsonl", type=Path, required=True)
    ap.add_argument("--b-source", required=True, help="source prefix for side B")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    a_ids, a = load_slice(args.a, args.a_jsonl, args.a_source, args.device, "A")
    b_ids, b = load_slice(args.b, args.b_jsonl, args.b_source, args.device, "B")

    n = args.n or min(len(a_ids), len(b_ids))
    a_ids, a = subsample(a_ids, a, n, args.seed)
    b_ids, b = subsample(b_ids, b, n, args.seed + 1)

    print(f"\nA  {args.a_source:<40} {len(a_ids):>6} gradients")
    print(f"B  {args.b_source:<40} {len(b_ids):>6} gradients")
    print(f"compared at n={n}")

    # -- diversity ---------------------------------------------------------
    union_ids = a_ids + b_ids
    union = torch.cat([a, b], dim=0)
    union_ids, union = subsample(union_ids, union, n, args.seed + 2)

    v_a, v_b, v_union = vendi_score(a), vendi_score(b), vendi_score(union)
    print(f"\n{'':<22} {'G-Vendi':>9}")
    print(f"{'A alone':<22} {v_a:>9.2f}")
    print(f"{'B alone':<22} {v_b:>9.2f}")
    print(f"{'A + B':<22} {v_union:>9.2f}   ({v_union - max(v_a, v_b):+.2f} "
          f"vs the better of the two)")

    # -- separation --------------------------------------------------------
    # Within-side nearest-neighbour similarity is the control: it says how close
    # a row sits to its own kind. If cross-side similarity matches it, the two
    # slices are interleaved and gradient space does not distinguish them.
    a_within = nearest(a, a[torch.randperm(len(a_ids), device=a.device)][: len(a_ids) // 2])
    b_within = nearest(b, b[torch.randperm(len(b_ids), device=b.device)][: len(b_ids) // 2])
    a_to_b = nearest(a, b)
    b_to_a = nearest(b, a)

    print(f"\nnearest-neighbour cosine")
    print(f"{'':<22} {'median':>8} {'mean':>8}")
    print(f"{'A -> A (control)':<22} {a_within.median():>8.3f} {a_within.mean():>8.3f}")
    print(f"{'B -> B (control)':<22} {b_within.median():>8.3f} {b_within.mean():>8.3f}")
    print(f"{'A -> B':<22} {a_to_b.median():>8.3f} {a_to_b.mean():>8.3f}")
    print(f"{'B -> A':<22} {b_to_a.median():>8.3f} {b_to_a.mean():>8.3f}")

    within = float((a_within.median() + b_within.median()) / 2)
    across = float((a_to_b.median() + b_to_a.median()) / 2)
    gap = within - across
    print(f"\nwithin {within:.3f} vs across {across:.3f}   gap {gap:+.3f}")

    # -- can a linear probe tell them apart? -------------------------------
    # The decisive test. Centroids are fitted on half of each slice and scored
    # on the held-out half: fitting and scoring on the same rows reports ~78%
    # even for two samples from an identical distribution, which would look
    # like separation where there is none.
    generator = torch.Generator(device=a.device).manual_seed(args.seed)
    perm_a = torch.randperm(len(a_ids), device=a.device, generator=generator)
    perm_b = torch.randperm(len(b_ids), device=b.device, generator=generator)
    half_a, half_b = len(a_ids) // 2, len(b_ids) // 2

    fit_a, test_a = a[perm_a[:half_a]], a[perm_a[half_a:]]
    fit_b, test_b = b[perm_b[:half_b]], b[perm_b[half_b:]]

    centroid_a = F.normalize(F.normalize(fit_a.float(), dim=1).mean(0, keepdim=True), dim=1)
    centroid_b = F.normalize(F.normalize(fit_b.float(), dim=1).mean(0, keepdim=True), dim=1)
    centroids = torch.cat([centroid_a, centroid_b])

    a_pred = (F.normalize(test_a.float(), dim=1) @ centroids.T).argmax(1)
    b_pred = (F.normalize(test_b.float(), dim=1) @ centroids.T).argmax(1)
    accuracy = float((a_pred == 0).float().mean() + (b_pred == 1).float().mean()) / 2

    print(f"nearest-centroid accuracy (held out): {accuracy:.1%}  "
          f"(50% = indistinguishable)")
    print(f"centroid cosine A.B: {float(centroid_a @ centroid_b.T):.3f}")

    if accuracy < 0.6:
        print("\nThe two slices are not separable in gradient space. For data "
              "this similar on the surface, the tool adds nothing over the "
              "metadata you already have.")
    elif accuracy < 0.8:
        print("\nPartially separable: some structural signal, but the slices "
              "overlap substantially.")
    else:
        print("\nClearly separable despite matched surface form -- gradient "
              "space is picking up structure, not just formatting.")


if __name__ == "__main__":
    main()
