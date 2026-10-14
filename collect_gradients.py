#!/usr/bin/env python3
"""Collect projected per-example gradients with a small proxy model.

Single-GPU by default. Resumes by skipping sample ids already present in the
gradient store, so an interrupted run can be restarted with the same command.

    python collect_gradients.py \
        --dataset_filename data/excellent.jsonl \
        --save_dir data/gradients/excellent \
        --max_samples 50000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gradient_computer import GradientComputer

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    ap.add_argument("--dataset_filename", type=Path, required=True)
    ap.add_argument("--save_dir", type=Path, required=True)
    ap.add_argument("--max_samples", type=int, default=None,
                    help="cap on samples processed this run (after resume skip)")
    ap.add_argument("--max_length", type=int, default=2048,
                    help="truncate sequences; cost scales with this")
    ap.add_argument("--proj_dim", type=int, default=1024)
    ap.add_argument("--save_interval", type=int, default=500)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                    help="float32 is safest for gradient fidelity on a 0.5B model")
    ap.add_argument("--projector", default="auto", choices=["auto", "trak", "sketch"],
                    help="auto tries fast_jl then falls back to the count sketch")
    ap.add_argument("--shard", type=int, default=0, help="this shard index")
    ap.add_argument("--num_shards", type=int, default=1)
    return ap.parse_args()


def already_done(save_dir: Path) -> set[str]:
    """Sample ids recorded in the sidecar .txt files next to each shard."""
    done: set[str] = set()
    for txt in save_dir.glob("*.txt"):
        with open(txt) as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["id"])
    return done


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset_filename) as f:
        samples = [json.loads(line) for line in f if line.strip()]
    print(f"loaded {len(samples)} samples from {args.dataset_filename}")

    if args.num_shards > 1:
        samples = samples[args.shard :: args.num_shards]
        print(f"shard {args.shard}/{args.num_shards}: {len(samples)} samples")

    done = already_done(args.save_dir)
    if done:
        before = len(samples)
        samples = [s for s in samples if s["id"] not in done]
        print(f"resuming: {before - len(samples)} already computed, {len(samples)} remain")

    if args.max_samples is not None:
        samples = samples[: args.max_samples]
        print(f"capped to {len(samples)} samples this run")

    if not samples:
        print("nothing to do")
        return

    missing = [k for k in ("id", "prompt", "completion") if k not in samples[0]]
    if missing:
        raise KeyError(f"samples are missing required keys: {missing}")

    torch_dtype = {"float32": torch.float32,
                   "bfloat16": torch.bfloat16,
                   "float16": torch.float16}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, dtype=torch_dtype,
    ).cuda()

    computer = GradientComputer(
        model_name=args.model_name_or_path,
        model=model,
        tokenizer=tokenizer,
        proj_dim=args.proj_dim,
        save_interval=args.save_interval,
        max_length=args.max_length,
        projector=args.projector,
    )

    n = computer.compute_project_store_gradients(samples, args.save_dir)
    print(f"done: {n} gradients written to {args.save_dir}")


if __name__ == "__main__":
    main()
