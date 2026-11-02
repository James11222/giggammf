"""Small host-side exchanges between the processes of a multi-process (multi-node) job.

A multi-node job runs one controller process per node or per GPU (see
``jaxweb.initialize``). Fields only ever move between devices through XLA
collectives inside the jitted steps. The helpers here move the small host-side
results that the processes must agree on (clump tables, voxel counts, flags).

Each helper is a collective: every process must call it, in the same order.
With a single process they return their input unchanged and communicate
nothing.
"""

from __future__ import annotations

import jax
import numpy as np
from jax.experimental import multihost_utils


def _bucket(n):
    """Round ``n`` bytes up to a power of two, so the gather compiles for a few sizes only."""
    return 1 << max(10, int(n).bit_length())


def allgather(array):
    """
    ``array`` from every process, concatenated along axis 0 in process order.

    The arrays may have different lengths, but must have the same dtype and
    trailing shape on every process. They travel as raw bytes, so int64 and
    structured dtypes arrive intact even though JAX runs without x64. Meant
    for small arrays: each one is copied to every device for the gather.
    """
    array = np.ascontiguousarray(np.atleast_1d(array))
    if jax.process_count() == 1:
        return array
    raw = array.reshape(-1).view(np.uint8)
    sizes = multihost_utils.process_allgather(np.array([raw.size], dtype=np.int32))[:, 0]
    padded = np.zeros(_bucket(sizes.max()), dtype=np.uint8)
    padded[: raw.size] = raw
    gathered = multihost_utils.process_allgather(padded)
    data = np.concatenate([gathered[p, :size] for p, size in enumerate(sizes)])
    return data.view(array.dtype).reshape(-1, *array.shape[1:])


def process_sum(values):
    """Element-wise sum of the NumPy ``values`` (a scalar or 1D array) over all processes."""
    values = np.atleast_1d(np.asarray(values))
    return allgather(values[None]).sum(axis=0)


def any_process(flag):
    """True on every process if ``flag`` is true on any process."""
    return bool(allgather(np.array([bool(flag)])).any())


def to_numpy(x):
    """
    The whole of the sharded array ``x`` as a NumPy array on every process.

    For small grids only (checks, tests): the array is gathered on every
    device first.
    """
    if isinstance(x, jax.Array) and not x.is_fully_addressable:
        return np.asarray(multihost_utils.process_allgather(x, tiled=True))
    return np.asarray(x)
