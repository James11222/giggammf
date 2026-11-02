"""
One process of the multi-process test (``test_multiprocess.py``), run as

    python multiprocess_worker.py PROCESS_ID N_PROCESSES PORT OUT_DIR

with ``XLA_FLAGS`` giving each process a few CPU devices, as a node has a few
GPUs. Every process computes everything on the mesh over all processes;
process 0 writes the gathered results to ``OUT_DIR/results.npz`` and the
tagging summaries to ``OUT_DIR/summary_<statistic>.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import numpy as np
from cubes import lognormal_cube
from jax.experimental import multihost_utils

import jaxweb
from jaxweb.distributed import allgather, to_numpy
from jaxweb.labels import SUMS, clump_sums

RS = [1.0, 2.0**0.5, 2.0, 2.0**1.5]
SUMMARY_KEYS = ("S_clus_thresh", "S_fil_thresh", "S_wall_thresh", "f_vir_clus")
SUMMARY_KEYS += ("volume_fractions", "mass_fractions")


def clump_problem():
    """A random mask and integer ρ (so every float32 sum is exact), as in test_tagging."""
    rng = np.random.default_rng(11)
    mask = rng.random((32, 24, 20)) < 0.1
    return mask, rng.integers(0, 40, mask.shape).astype(np.float32)


def main():
    process_id, n_processes, port, out = sys.argv[1:]
    out = Path(out)
    jaxweb.initialize(
        coordinator_address=f"localhost:{port}",
        num_processes=int(n_processes),
        process_id=int(process_id),
    )
    mesh = jaxweb.make_mesh()
    rho = lognormal_cube(64, seed=7)
    results = {"floor": np.float32(jaxweb.density_floor(rho, mesh))}

    sigmax = jaxweb.maximum_signature(rho, RS, mesh=mesh)
    results["sig_nexusplus"] = to_numpy(sigmax)
    sig_offload = jaxweb.maximum_signature(rho, RS[:2], mesh=mesh, offload=True)
    results["sig_offload"] = to_numpy(sig_offload)
    sig_nexus = jaxweb.maximum_signature(rho, RS, algorithm="NEXUS", mesh=mesh)
    results["sig_nexus"] = to_numpy(sig_nexus)

    summaries = {}
    for statistic in ("mean", "max"):
        structure, summary = jaxweb.tag_structures(
            rho, sigmax, overdensity_threshold=20, clump_statistic=statistic, mesh=mesh
        )
        results[f"structure_{statistic}"] = to_numpy(structure)
        summaries[statistic] = {
            key: None if summary[key] is None else np.asarray(summary[key], np.float64).tolist()
            for key in SUMMARY_KEYS
        }

    # 2-plane pieces: 4 pieces per slab, so clumps join across pieces, slabs
    # and the boundary between the two processes.
    mask, rho_int = clump_problem()
    jaxweb.labels.LABEL_ELEMENTS = 2 * 24 * 20
    sums = clump_sums(mask, rho_int, mesh, SUMS, 19.5)
    table = np.stack([sums[name].astype(np.float64) for name in SUMS], axis=1)
    results["clump_table"] = allgather(table)  # every process's share of the clumps

    results["tiled"] = to_numpy(jaxweb.load_density(out / "cube.npy", mesh, tile=2))
    results["n_processes"] = np.int32(jax.process_count())
    results["n_devices"] = np.int32(jax.device_count())

    if jax.process_index() == 0:
        np.savez(out / "results.npz", **results)
        for statistic, summary in summaries.items():
            (out / f"summary_{statistic}.json").write_text(json.dumps(summary))
    multihost_utils.sync_global_devices("multiprocess_worker: done")


if __name__ == "__main__":
    main()
