"""
Tag a density cube as cluster / filament / wall / void and write the result to disk.

    python scripts/tag_cube.py density.npy [--out-dir DIR] [--algorithm NEXUSPLUS]
        [--overdensity-threshold 370] [--n-scales 9]
        [--clump-statistic max [mean mass_weighted median]] [--tile 2] [--no-save]

Writes ``<stem>_<algorithm>_structure_filter.npy`` (uint8, 1 cluster,
2 filament, 3 wall, 4 void, same shape as the cube) and
``<stem>_<algorithm>_summary.json`` (thresholds, curves, volume and mass
fractions as in PyCosmoMMF's ``summary_data``, plus the run parameters,
timings and peak device memory).

Each ``--clump-statistic`` (see ``jaxweb.tagging.virialized_clumps``) tags the
cube once from the same signatures, and writes its own pair of files, named
``<stem>_<algorithm>_<statistic>_...``. The default, ``max``, keeps the names
above; PyCosmoMMF's rule is ``mean``.

Runs on one GPU, one node or several nodes alike: under ``srun`` every task
joins one job (``jaxweb.initialize``) and the cube is split into one x-slab per
GPU over all nodes (see ``scripts/tag_cube_multinode.slurm``). ``--tile k``
tags the cube repeated k times along each axis (a periodic box k times larger
per side, for scaling tests), named ``<stem>_tile<k>_...``.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import jax
import numpy as np
from jax.experimental import multihost_utils

import jaxweb
from jaxweb.distributed import allgather

GiB = 2**30


def save_sharded(path, array):
    """Write a sharded array to one ``.npy``; each process writes only its own shards."""
    if jax.process_index() == 0:
        np.lib.format.open_memmap(path, mode="w+", dtype=array.dtype, shape=array.shape).flush()
    multihost_utils.sync_global_devices("save_sharded: file created")
    out = np.load(path, mmap_mode="r+")
    for shard in array.addressable_shards:
        if shard.replica_id == 0:
            out[shard.index] = np.asarray(shard.data)
    out.flush()
    del out
    multihost_utils.sync_global_devices("save_sharded: shards written")


def to_json(value):
    """NumPy scalars and arrays (and lists of them) as plain JSON values."""
    if isinstance(value, dict):
        return {k: to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if isinstance(value, np.ndarray | jax.Array):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def peak_memory():
    """
    Peak GiB in use on every device of every process (None where not
    reported). Every process must call it.
    """
    local = [
        [device.id, (device.memory_stats() or {}).get("peak_bytes_in_use", -1)]
        for device in jax.local_devices()
    ]
    rows = allgather(np.array(local, dtype=np.int64))
    names = {device.id: str(device) for device in jax.devices()}
    return {names[i]: None if peak < 0 else round(peak / GiB, 2) for i, peak in rows}


def describe_layout():
    """One line per process: its host and devices, printed by process 0."""
    line = f"process {jax.process_index()} on {socket.gethostname()}: {jax.local_devices()}"
    lines = allgather(np.frombuffer(line.encode().ljust(512)[:512], np.uint8)[None])
    if jax.process_index() == 0:
        print(f"{jax.process_count()} processes, {jax.device_count()} devices")
        for row in lines:
            print("  " + row.tobytes().decode().rstrip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="δ+1 = ρ/⟨ρ⟩ cube, .npy")
    parser.add_argument("--out-dir", default=None, help="default: the cube's directory")
    parser.add_argument("--algorithm", default="NEXUSPLUS", choices=["NEXUS", "NEXUSPLUS"])
    parser.add_argument("--n-scales", type=int, default=9, help="R = 2^(n/2), n < N voxels")
    parser.add_argument("--overdensity-threshold", type=float, default=330)
    parser.add_argument("--tile", type=int, default=1, help="repeat the cube k times per axis")
    parser.add_argument("--no-save", action="store_true", help="do not write the filter")
    parser.add_argument(
        "--clump-statistic",
        nargs="+",
        default=["max"],
        choices=list(jaxweb.tagging.CLUMP_STATISTICS),
        help="per-clump δ statistic(s) that decide virialization; one output per statistic",
    )
    args = parser.parse_args()

    jaxweb.initialize()
    mesh = jaxweb.make_mesh()
    describe_layout()
    Rs = [2 ** (n / 2) for n in range(args.n_scales)]
    path = Path(args.path)
    out_dir = Path(args.out_dir) if args.out_dir else path.parent
    main_process = jax.process_index() == 0
    if main_process:
        out_dir.mkdir(parents=True, exist_ok=True)

    times = {}
    t0 = time.perf_counter()
    rho = jaxweb.load_density(path, mesh, tile=args.tile)
    rho.block_until_ready()
    times["load"] = time.perf_counter() - t0

    t = time.perf_counter()
    floor = jaxweb.density_floor(rho, mesh) if args.algorithm == "NEXUSPLUS" else None
    sigmax = jaxweb.maximum_signature(rho, Rs, algorithm=args.algorithm, mesh=mesh, floor=floor)
    sigmax.block_until_ready()
    times["signatures"] = time.perf_counter() - t

    for statistic in args.clump_statistic:
        stem = path.stem + (f"_tile{args.tile}" if args.tile > 1 else "") + f"_{args.algorithm}"
        if statistic != "max":
            stem += f"_{statistic}"
        filter_path = out_dir / f"{stem}_structure_filter.npy"
        summary_path = out_dir / f"{stem}_summary.json"
        run_times = dict(times)

        t = time.perf_counter()
        structure, summary = jaxweb.tag_structures(
            rho,
            sigmax,
            overdensity_threshold=args.overdensity_threshold,
            clump_statistic=statistic,
            mesh=mesh,
            verbose=main_process,
        )
        structure.block_until_ready()
        run_times["tagging"] = time.perf_counter() - t
        run_times.update({f"tagging: {k}": v for k, v in summary.pop("seconds").items()})

        if not args.no_save:
            t = time.perf_counter()
            save_sharded(filter_path, structure)
            run_times["write"] = time.perf_counter() - t
        run_times["total"] = time.perf_counter() - t0  # since the start of the run

        peaks = peak_memory()
        if main_process:
            summary["run"] = {
                "density": str(path.resolve()),
                "tile": args.tile,
                "shape": list(structure.shape),
                "algorithm": args.algorithm,
                "Rs": Rs,
                "overdensity_threshold": args.overdensity_threshold,
                "density_floor": floor,
                "devices": len(jax.devices()),
                "processes": jax.process_count(),
                "structure_filter": None if args.no_save else str(filter_path.resolve()),
                "seconds": {k: round(v, 1) for k, v in run_times.items()},
                "peak_memory_gib": peaks,
            }
            summary_path.write_text(json.dumps(to_json(summary), indent=2) + "\n")
            print(
                f"{args.algorithm} {structure.shape} on {len(jax.devices())} devices, {statistic}"
            )
            print("  " + ", ".join(f"{k} {v:.1f} s" for k, v in run_times.items()))
            for device, peak in peaks.items():
                print(f"  {device}: peak {peak} GiB")
            if not args.no_save:
                print(f"  wrote {filter_path}")
            print(f"  wrote {summary_path}")
        del structure


if __name__ == "__main__":
    main()
