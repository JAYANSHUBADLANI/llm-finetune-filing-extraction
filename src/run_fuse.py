"""Runs mlx_lm's fuse CLI with the same MPI probe patch as run_lora_training.py.

Fusing merges the trained LoRA adapter permanently into a copy of the base
model's weights, so evaluation can load one plain model with no
adapter_path argument. Tried loading with adapter_path directly for
evaluation first and it reproducibly consumed memory rapidly during
loading with no generation progress, twice, on this machine; fusing once
and loading the fused checkpoint normally avoids whatever that path does.
See PROGRESS.md.
"""

import mlx.core as mx

_original_init = mx.distributed.init


def _init_ring_only(*args, **kwargs):
    kwargs["backend"] = "ring"
    return _original_init(*args, **kwargs)


mx.distributed.init = _init_ring_only
mx.distributed.all_sum = lambda x, *args, **kwargs: x

from mlx_lm.fuse import main  # noqa: E402

if __name__ == "__main__":
    main()
