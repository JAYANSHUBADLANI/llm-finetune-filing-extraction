"""
Runs mlx_lm's LoRA CLI programmatically, after patching mx.distributed.init
to force the "ring" backend. Without this, mlx_lm.tuner.trainer calls
mx.distributed.init() with its default backend="any", which on this
machine finds this venv's conda-linked MPICH installation, tries to treat
it as Open MPI (which MLX requires and MPICH is not), and hard crashes
with SIGABRT before a single training step runs, exit code 134, no
Python exception to catch. Confirmed the crash is specifically the MPI
probe by calling mx.distributed.init(backend="ring") directly, which
returns a working size 1 group with no crash. This patch does not change
the meaning of the run, since this is single machine, single process
training either way. See PROGRESS.md.
"""

import sys

import mlx.core as mx

_original_init = mx.distributed.init


def _init_ring_only(*args, **kwargs):
    kwargs["backend"] = "ring"
    return _original_init(*args, **kwargs)


mx.distributed.init = _init_ring_only

# mlx_lm's evaluate() calls mx.distributed.all_sum(x, stream=...) directly,
# with no group argument, unlike average_gradients which resolves a group
# and short circuits when its size is 1. all_sum with no explicit group
# resolves the process's default global group at the C++ level, which is
# a second, independent trigger of the same MPI probe that crashed
# mx.distributed.init before it was patched above, so patching init alone
# was not sufficient, discovered when training got through the first
# validation pass and then crashed. Summing across a single process's
# values is the identity, so this is exact, not an approximation for the
# single machine case this project runs under.
mx.distributed.all_sum = lambda x, *args, **kwargs: x

from mlx_lm.lora import main  # noqa: E402

if __name__ == "__main__":
    main()
