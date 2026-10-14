"""Random projection for 500M-dimensional gradients, without fast_jl.

TRAK's CudaProjector fuses Rademacher generation into a CUDA kernel, so it never
materialises the (grad_dim x proj_dim) matrix. That kernel ships as prebuilt PTX
and fails on any driver older than the toolchain it was compiled against.
BasicProjector, the fallback, *does* materialise the matrix -- 494M x 1024 is
about a terabyte -- so it is not a real option either.

What this uses instead is a sign-randomised interleaved sum:

    out[j] = sum_i  s[i * proj_dim + j] * g[i * proj_dim + j]

Coordinate c lands in bucket c mod proj_dim, carrying a fixed random sign. Since
E[s_a s_b] = 0 for a != b and s^2 = 1, dot products are preserved in
expectation:  E[<Px, Py>] = <x, y>.

Why bucket by position rather than by a random hash: a random hash needs
scatter_add, which for 494M values landing in 1024 buckets means roughly half a
million atomic adds contending per bucket. That was the first implementation
here and it ran at ~12 s/sample. Interleaving instead makes the reduction a
contiguous strided sum with no atomics, and the stride spreads each layer's
parameters across all buckets rather than piling them into a few.

The cost is that coordinates congruent mod proj_dim always collide, where a
random hash would collide them only by chance. The random signs keep the
estimator unbiased either way; only the variance structure differs.
"""
from __future__ import annotations

import torch


class CountSketchProjector:
    """Drop-in replacement for trak.projectors.*Projector.

    Only `project(features, model_id)` is used by the gradient collector.
    """

    def __init__(
        self,
        grad_dim: int,
        proj_dim: int = 1024,
        seed: int = 0,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
        cache_signs: bool = True,
        **_ignored,
    ):
        self.grad_dim = grad_dim
        self.proj_dim = proj_dim
        self.seed = seed
        self.device = torch.device(device)
        self.dtype = dtype

        # Pad up to a whole number of buckets so the tensor reshapes cleanly.
        self.padded_dim = ((grad_dim + proj_dim - 1) // proj_dim) * proj_dim
        self.n_rows = self.padded_dim // proj_dim

        # One int8 per parameter: 494 MB at 0.5B. Worth it -- regenerating this
        # many randoms per sample costs more than the backward pass does.
        self.signs = None
        if cache_signs:
            self.signs = self._make_signs()

    def _make_signs(self) -> torch.Tensor:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        signs = torch.randint(
            0, 2, (self.padded_dim,),
            generator=generator, device=self.device, dtype=torch.int8,
        )
        return signs.mul_(2).sub_(1)

    def project(self, features: torch.Tensor, model_id: int = 0) -> torch.Tensor:
        """features: (batch, grad_dim) -> (batch, proj_dim)."""
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.shape[1] != self.grad_dim:
            raise ValueError(
                f"expected features of width {self.grad_dim}, got {features.shape[1]}"
            )

        batch = features.shape[0]
        signs = self.signs if self.signs is not None else self._make_signs()

        if self.padded_dim != self.grad_dim:
            pad = self.padded_dim - self.grad_dim
            features = torch.nn.functional.pad(features, (0, pad))

        # (batch, n_rows * proj_dim) -> (batch, n_rows, proj_dim) -> sum rows.
        # fp16 accumulation over ~480k terms overflows, so accumulate in fp32.
        signed = features * signs
        out = signed.view(batch, self.n_rows, self.proj_dim).sum(dim=1, dtype=torch.float32)

        return out.to(self.dtype)

    def project_model_grads(self, model, free_as_we_go: bool = True) -> torch.Tensor:
        """Project a model's .grad tensors without ever concatenating them.

        Mathematically identical to flattening every gradient into one vector
        and calling `project` on it -- the projection is a sum, so it can be
        accumulated one parameter at a time.

        The point is memory. Concatenating 494M elements costs 2 GB, plus
        another GB for an fp16 copy, per sample. On a 16 GB card that fills the
        allocator within a few hundred samples and throughput collapses. Here
        the largest transient is one parameter (the embedding, ~0.5 GB).

        Each parameter's slice is summed into buckets by position within the
        slice, then rolled by `offset mod proj_dim` so it lands on the same
        buckets the whole-vector path would have used.
        """
        signs = self.signs if self.signs is not None else self._make_signs()
        out = torch.zeros(self.proj_dim, dtype=torch.float32, device=self.device)

        offset = 0
        for param in model.parameters():
            if param.grad is None:
                continue

            n = param.numel()
            signed = param.grad.view(-1) * signs[offset : offset + n]

            pad = (-n) % self.proj_dim
            if pad:
                signed = torch.nn.functional.pad(signed, (0, pad))

            partial = signed.view(-1, self.proj_dim).sum(dim=0, dtype=torch.float32)
            out += torch.roll(partial, shifts=offset % self.proj_dim, dims=0)

            offset += n
            del signed, partial
            if free_as_we_go:
                param.grad = None

        return out.to(self.dtype)


def get_projector(grad_dim: int, proj_dim: int, device: torch.device,
                  prefer: str = "auto", **kwargs):
    """Return a projector, preferring TRAK's CUDA kernel when it actually runs.

    `prefer` is one of "auto", "trak", "sketch".
    """
    if prefer == "sketch":
        return CountSketchProjector(grad_dim, proj_dim, device=device, **kwargs)

    try:
        import fast_jl
        from trak.projectors import CudaProjector, ProjectionType

        num_sms = torch.cuda.get_device_properties(device.index).multi_processor_count
        fast_jl.project_rademacher_8(torch.zeros(8, 1_000, device=device), 512, 0, num_sms)

        return CudaProjector(
            grad_dim=grad_dim,
            proj_dim=proj_dim,
            seed=0,
            proj_type=ProjectionType.rademacher,
            device=device,
            dtype=torch.float16,
            block_size=128,
            max_batch_size=16,
        )
    except Exception as exc:  # noqa: BLE001 - any failure means fall back
        if prefer == "trak":
            raise
        print(f"fast_jl unusable ({type(exc).__name__}); using CountSketchProjector")
        return CountSketchProjector(grad_dim, proj_dim, device=device, **kwargs)
