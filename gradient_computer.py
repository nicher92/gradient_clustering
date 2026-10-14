"""Per-example gradient collection and random projection, after Prismatic
Synthesis (Jung et al.), https://github.com/jaehunjung1/prismatic-synthesis

Two changes from the original:

1. `DataCollatorForCompletionOnlyLM` has been removed from current TRL, so the
   prompt masking is done here directly: render the prompt alone with
   `add_generation_prompt=True`, and mask that many leading tokens. Same
   semantics, no TRL dependency.

2. The gradient is projected straight out of the .grad tensors, one parameter
   at a time, instead of being flattened into a single 494M-element vector
   first. The upstream path allocates ~3 GB per sample for that flattening; on
   a 16 GB card the allocator fills within a few hundred samples and throughput
   drops by an order of magnitude.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import PreTrainedModel

from projectors import get_projector

IGNORE_INDEX = -100


class GradientComputer:
    def __init__(
        self,
        model_name: str,
        model: PreTrainedModel,
        tokenizer,
        proj_dim: int = 1024,
        save_interval: int = 500,
        max_length: int = 2048,
        projector: str = "auto",
    ):
        self.proj_dim = proj_dim

        self.save_interval = save_interval
        self.max_length = max_length

        self.model_name = model_name
        self.model = model
        self.model.eval()
        self.device = self.model.device
        self.dtype = self.model.dtype
        self.grad_dim = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.tokenizer = tokenizer

        self.projector = get_projector(
            grad_dim=self.grad_dim,
            proj_dim=self.proj_dim,
            device=self.device,
            prefer=projector,
        )
        print(f"projector: {type(self.projector).__name__}, "
              f"grad_dim={self.grad_dim:,} -> proj_dim={self.proj_dim}")

    def prepare_model_input(self, prompt: str, completion: str) -> dict | None:
        """Tokenise one example with the prompt masked out of the loss.

        Rendering twice -- prompt alone with the generation prompt appended,
        then prompt plus completion -- gives the boundary by length difference.
        This is the same approach TRL's prompt-completion path uses and needs
        nothing from the chat template beyond being able to render.
        """
        prompt_msgs = [{"role": "user", "content": prompt}]
        full_msgs = prompt_msgs + [{"role": "assistant", "content": completion}]

        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_msgs, add_generation_prompt=True, tokenize=True,
        )
        full_ids = self.tokenizer.apply_chat_template(full_msgs, tokenize=True)

        # apply_chat_template returns a BatchEncoding when return_dict defaults
        # to True (transformers 5.x). Normalise both shapes to a plain list.
        if not isinstance(prompt_ids, list):
            prompt_ids = prompt_ids["input_ids"]
        if not isinstance(full_ids, list):
            full_ids = full_ids["input_ids"]

        n_prompt = len(prompt_ids)
        if len(full_ids) <= n_prompt:
            return None  # completion contributed no tokens

        full_ids = full_ids[: self.max_length]
        if len(full_ids) <= n_prompt:
            return None  # truncation removed the whole completion

        labels = list(full_ids)
        labels[:n_prompt] = [IGNORE_INDEX] * n_prompt

        return {
            "input_ids": torch.tensor([full_ids], device=self.device),
            "attention_mask": torch.ones(1, len(full_ids), dtype=torch.long, device=self.device),
            "labels": torch.tensor([labels], device=self.device),
        }

    def obtain_projected_gradient(self, batch: dict) -> torch.Tensor:
        """Backward pass, then project straight out of the .grad tensors.

        The full 494M-element gradient is never concatenated: that allocation
        (2 GB, plus an fp16 copy) is what filled the card and collapsed
        throughput. See CountSketchProjector.project_model_grads.
        """
        self.model.zero_grad(set_to_none=True)
        loss = self.model(**batch).loss
        loss.backward()

        if hasattr(self.projector, "project_model_grads"):
            return self.projector.project_model_grads(self.model)

        # TRAK projectors only take a flat tensor, so fall back to the
        # concatenating path when fast_jl is actually available.
        grad = torch.cat(
            [p.grad.view(-1) for p in self.model.parameters() if p.grad is not None],
            dim=0,
        ).to(torch.float16)
        projected = self.projector.project(grad.unsqueeze(0), model_id=0)[0]
        del grad
        return projected

    @staticmethod
    def save_projected_gradients(projected: Dict[str, torch.Tensor], save_filename: Path) -> None:
        if not projected:
            return
        if save_filename.exists():
            raise FileExistsError(f"{save_filename} already exists; refusing to overwrite.")

        save_file({k: v.contiguous().cpu() for k, v in projected.items()}, str(save_filename))

        with open(save_filename.with_suffix(".txt"), "w") as f:
            f.write("\n".join(json.dumps({"id": sid}) for sid in projected) + "\n")

    def compute_project_store_gradients(
        self, samples: List[Dict], save_directory: Path, global_start_idx: int = 0,
    ) -> int:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        projected_grads: Dict[str, torch.Tensor] = {}
        n_done = 0
        n_skipped = 0

        def flush(idx: int) -> None:
            nonlocal projected_grads
            if not projected_grads:
                return
            name = save_directory / f"{global_start_idx + idx - len(projected_grads) + 1}.safetensors"
            self.save_projected_gradients(projected_grads, name)
            projected_grads = {}

        for sample_idx, sample in enumerate(tqdm(samples, desc="gradients")):
            encoding = self.prepare_model_input(sample["prompt"], sample["completion"])
            if encoding is None:
                n_skipped += 1
                continue

            projected_grads[sample["id"]] = self.obtain_projected_gradient(encoding)
            n_done += 1

            if n_done % self.save_interval == 0 and projected_grads:
                flush(sample_idx)

        flush(len(samples) - 1)

        if n_skipped:
            print(f"skipped {n_skipped} samples (empty or fully truncated completion)")
        return n_done
