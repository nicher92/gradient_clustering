# G-Vendi diversity analysis for OpenEuroLLM-Excellent-SFT

Gradient-space diversity measurement, after [Prismatic
Synthesis](https://github.com/jaehunjung1/prismatic-synthesis). A small proxy
model (Qwen2.5-0.5B-Instruct) produces one gradient per training example; those
are randomly projected to 1024 dimensions, and diversity is the exponentiated
entropy of the resulting eigenspectrum.

The paper's finding that makes this practical: diversity measured with a small
proxy transfers to much larger models, even across families.

## Install

```bash
python3 -m pip install torch transformers datasets safetensors scipy scikit-learn tqdm
```

No `trak` or `fast_jl` needed. The upstream code projects gradients with TRAK's
`CudaProjector`, which needs the `fast_jl` CUDA extension; its prebuilt PTX
fails on any driver older than the toolchain it was compiled against:

```
RuntimeError: CUDA error: the provided PTX was compiled with an
unsupported toolchain
```

(torch cu124 against a 12.3 driver, for instance — check with
`python -c "import torch; print(torch.version.cuda)"` against `nvidia-smi`.)

TRAK's `BasicProjector` fallback is not usable here either: it materialises the
projection matrix, and 494M x 1024 is about a terabyte. It fails trying to
allocate 118 GiB.

`projectors.py` therefore implements a count sketch instead — hash each gradient
coordinate to one of 1024 buckets with a random sign and sum. O(d) per sample
rather than O(d * proj_dim), no matrix stored, no CUDA extension. It preserves
dot products in expectation, which is what the Vendi score needs. Measured on
Gaussian vectors at 2M dims: ~5% relative error on dot products, 0.05 absolute
on cosine similarity. Good enough for clustering and ranking mixes; not for
tight per-pair distance guarantees.

If you do have a working `fast_jl`, `--projector trak` uses it. Default `auto`
tries TRAK and falls back. The projector actually in use is printed at startup.

## Run

```bash
# 1. sample the dataset, stratified by source
python prepare_data.py --per-source 10000 --out data/excellent.jsonl

# 2. collect gradients (resumable; rerun the same command to continue)
python collect_gradients.py \
    --dataset_filename data/excellent.jsonl \
    --save_dir data/gradients/excellent \
    --max_length 2048

# 3. analyse
python analyze.py \
    --dataset_filename data/excellent.jsonl \
    --gradient_storage_dir data/gradients/excellent
```

## Before committing to a long run

Time 200 examples first:

```bash
python collect_gradients.py --dataset_filename data/excellent.jsonl \
    --save_dir /tmp/grad-test --max_samples 200
```

Cost is one forward+backward per example, strictly sequential, so throughput
scales roughly linearly with `--max_length`. At 2048 tokens on a 16 GB consumer
card, expect single-digit examples/second — enough for tens of thousands
overnight, not for millions.

Gradients are projected straight out of the `.grad` tensors, one parameter at
a time, so the full 494M-element vector is never materialised. Peak memory is
roughly model + grads + the sign vector, about 5 GB at 0.5B, and is flat over
the run. If `nvidia-smi` shows memory climbing toward the ceiling and
throughput falling, something has regressed to the flattening path.

## Why stratified sampling

Excellent-SFT is stored in blocks by source: nemotron (3.76M rows), smoltalk2
(905k), dolci (565k), open-perfectblend (533k), orca (429k). Reading from the
head of the file gives you nemotron and nothing else. `prepare_data.py` takes a
fixed count per source and also shuffles within a streaming buffer.

This costs you proportionality — the sample is not representative of the mix as
trained. That is the right trade for coverage analysis (you want to see each
source's structure) and the wrong one for estimating the mix's overall
diversity as trained. For the latter, sample proportionally instead.

## Reading the output

**Overall G-Vendi** is bounded between 1 (every gradient identical) and
`proj_dim` (orthogonal). The absolute value means little; it is for comparing
mixes. It also grows with sample count, so only compare runs at matched `n`.

**Per-source G-Vendi** shows which component carries the variety. A large source
with a low score is adding volume but not coverage.

**Cluster composition** is the most actionable part. Clusters that are ≥90% one
source are the regions of gradient space only that source reaches — remove it
and that coverage disappears.

## Differences from the upstream implementation

- The projector is a count sketch rather than TRAK's Rademacher projection
  (see Install). Projected vectors are therefore not comparable with any
  produced by the upstream code.
- `DataCollatorForCompletionOnlyLM` was removed from TRL, so prompt masking is
  done directly: render the prompt with `add_generation_prompt=True`, mask that
  many leading tokens, supervise the rest.
- The projection accumulates per parameter rather than over one flattened
  494M-element vector. Identical arithmetic, but it avoids ~3 GB of transient
  allocation per sample, which on a 16 GB card was the difference between
  2 it/s and 0.08 it/s.
- `compute_project_store_gradients` takes three arguments here; upstream's
  `collect_gradients.py` passes four to a three-argument method, which does not
  run as published.
- Single-GPU with resume, rather than the upstream `CUDA_VISIBLE_DEVICES` fan-out
  (whose loop is also off by one: `seq 0 $NUM_GPUS` yields N+1 shards).
  `--shard`/`--num_shards` are available if you want to parallelise.

## Caveats worth keeping in mind

The projection is random and seeded at 0. Two runs are comparable only if the
projector saw the same `grad_dim` — so the same proxy model, unchanged.

Truncation at `--max_length` is not neutral. Your multilingual reasoning traces
average ~8000 tokens; at 2048 you are measuring the gradient of a truncated
example, which is a different example. Keep `max_length` fixed across any mixes
you intend to compare, and be aware the measurement is of the truncated
distribution.
