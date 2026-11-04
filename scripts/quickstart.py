"""
Tag a density cube as cluster / filament / wall / void with NEXUS+.

    python scripts/quickstart.py density.npy

The input is δ+1 = ρ/⟨ρ⟩ on a periodic grid, float32 .npy. Writes
``<stem>_structure.npy`` (uint8: 1 cluster, 2 filament, 3 wall, 4 void) next to it.
Run it with ``scripts/quickstart.sh``, which sets the GPU count.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np
from jax.experimental import multihost_utils

import giggammf

giggammf.initialize()  # joins every process of the job; no-op for one process
mesh = giggammf.make_mesh()  # all GPUs of all nodes; the cube is split into one slab per GPU

path = Path(sys.argv[1])
rho = giggammf.load_density(path, mesh)

Rs = [2 ** (n / 2) for n in range(9)]  # smoothing scales, 1 to 16 voxels
sigmax = giggammf.maximum_signature(rho, Rs, mesh=mesh)
structure, summary = giggammf.tag_structures(
    rho, sigmax, overdensity_threshold=370, mesh=mesh, verbose=jax.process_index() == 0
)

# Each process writes its own slabs into one shared .npy.
out = path.with_name(f"{path.stem}_structure.npy")
if jax.process_index() == 0:
    np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=structure.shape).flush()
multihost_utils.sync_global_devices("file created")
array = np.load(out, mmap_mode="r+")
for shard in structure.addressable_shards:
    array[shard.index] = np.asarray(shard.data)
array.flush()
multihost_utils.sync_global_devices("file written")
if jax.process_index() == 0:
    print(f"wrote {out}")
