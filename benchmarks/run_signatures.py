"""Time ``maximum_signature`` on a density cube and report peak device memory.

python benchmarks/run_signatures.py density.npy [--algorithm NEXUS] [--tile 2]

Under ``srun`` with several tasks (several nodes) every task joins one job and
the cube is split over all their GPUs. ``--tile k`` repeats the cube k times
along each axis, for scaling tests.
"""

from __future__ import annotations

import argparse
import time

import jax
import numpy as np

import giggammf
from giggammf.distributed import allgather

GiB = 2**30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--algorithm", default="NEXUSPLUS", choices=["NEXUS", "NEXUSPLUS"])
    parser.add_argument("--n-scales", type=int, default=9)
    parser.add_argument("--offload", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--tile", type=int, default=1, help="repeat the cube k times per axis")
    args = parser.parse_args()

    giggammf.initialize()
    mesh = giggammf.make_mesh()
    Rs = [2 ** (n / 2) for n in range(args.n_scales)]
    offload = {"auto": None, "on": True, "off": False}[args.offload]

    t0 = time.perf_counter()
    rho = giggammf.load_density(args.path, mesh, tile=args.tile)
    rho.block_until_ready()
    t1 = time.perf_counter()
    sigmax = giggammf.maximum_signature(
        rho, Rs, algorithm=args.algorithm, mesh=mesh, offload=offload
    )
    sigmax.block_until_ready()
    t2 = time.perf_counter()

    local = [
        [device.id, (device.memory_stats() or {}).get("peak_bytes_in_use", -1)]
        for device in jax.local_devices()
    ]
    peaks = allgather(np.array(local, dtype=np.int64))
    if jax.process_index() == 0:
        print(
            f"{args.algorithm} {rho.shape} on {jax.device_count()} devices "
            f"({jax.process_count()} processes), {len(Rs)} scales"
        )
        print(f"  load {t1 - t0:.1f} s, signatures {t2 - t1:.1f} s")
        names = {device.id: str(device) for device in jax.devices()}
        for i, peak in peaks:
            if peak >= 0:
                print(f"  {names[i]}: peak {peak / GiB:.1f} GiB")


if __name__ == "__main__":
    main()
