"""giggammf split over several processes, as on several nodes.

Two processes with two simulated CPU devices each run ``multiprocess_worker.py``
on one mesh of 4 devices, talking through JAX's distributed runtime (gloo on
CPU, NCCL on GPUs). Their results must match the single-process ones on the 4
devices of this process (``conftest.py``).
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from cubes import lognormal_cube
from multiprocess_worker import RS, SUMMARY_KEYS, clump_problem

import giggammf
from giggammf.labels import SUMS, clump_sums
from giggammf.mesh import slurm_local_device_ids

skimage_measure = pytest.importorskip("skimage.measure")

N_PROCESSES = 2
WORKER = Path(__file__).with_name("multiprocess_worker.py")


def _free_port():
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def small_cube():
    return np.random.default_rng(3).random((16, 8, 12)).astype(np.float32)


@pytest.fixture(scope="module")
def run(tmp_path_factory, small_cube):
    out = tmp_path_factory.mktemp("multiprocess")
    np.save(out / "cube.npy", small_cube)
    env = {
        **os.environ,
        "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
        "JAX_PLATFORMS": "cpu",
    }
    port = str(_free_port())
    logs = [out / f"process_{i}.log" for i in range(N_PROCESSES)]
    with contextlib.ExitStack() as stack:
        procs = [
            subprocess.Popen(
                [sys.executable, str(WORKER), str(i), str(N_PROCESSES), port, str(out)],
                env=env,
                stdout=stack.enter_context(log.open("w")),
                stderr=subprocess.STDOUT,
            )
            for i, log in enumerate(logs)
        ]
        try:
            for proc in procs:
                proc.wait(timeout=900)
        finally:
            for proc in procs:
                proc.kill()
                proc.wait()
    for proc, log in zip(procs, logs, strict=True):
        assert proc.returncode == 0, log.read_text()
    results = dict(np.load(out / "results.npz"))
    summaries = {s: json.loads((out / f"summary_{s}.json").read_text()) for s in ("mean", "max")}
    return results, summaries


@pytest.fixture(scope="module")
def rho():
    return lognormal_cube(64, seed=7)


def test_layout(run):
    results, _ = run
    assert results["n_processes"] == N_PROCESSES
    assert results["n_devices"] == 4


def test_density_floor_matches(mesh, run, rho):
    results, _ = run
    assert results["floor"] == giggammf.density_floor(rho, mesh)


@pytest.mark.parametrize(
    ("key", "algorithm", "Rs"),
    [
        ("sig_nexusplus", "NEXUSPLUS", RS),
        ("sig_offload", "NEXUSPLUS", RS[:2]),  # offloaded there, not here
        ("sig_nexus", "NEXUS", RS),
    ],
)
def test_signatures_match(mesh, run, rho, key, algorithm, Rs):
    results, _ = run
    want = np.asarray(giggammf.maximum_signature(rho, Rs, algorithm=algorithm, mesh=mesh))
    if algorithm == "NEXUS":
        np.testing.assert_array_equal(results[key], want)
        return
    # NEXUS+ renormalises the mass with two psums, which the processes add up
    # in another order: the total moves by an ulp, and so every signature.
    for c in range(3):
        np.testing.assert_allclose(results[key][c], want[c], rtol=0, atol=1e-6 * want[c].max())


@pytest.mark.parametrize("statistic", ["mean", "max"])
def test_tagging_matches(mesh, run, rho, statistic):
    results, summaries = run
    sigmax = giggammf.maximum_signature(rho, RS, mesh=mesh)
    structure, summary = giggammf.tag_structures(
        rho, sigmax, overdensity_threshold=20, clump_statistic=statistic, mesh=mesh
    )
    np.testing.assert_array_equal(results[f"structure_{statistic}"], np.asarray(structure))
    assert np.any(results[f"structure_{statistic}"] == giggammf.CLUSTER)
    for key in SUMMARY_KEYS:
        # Masses are float32 psums, added up in another order across processes.
        np.testing.assert_allclose(
            np.asarray(summaries[statistic][key], np.float64),
            np.asarray(summary[key], np.float64),
            rtol=1e-6 if key == "mass_fractions" else 0,
            err_msg=key,
        )


def test_clumps_join_across_processes(mesh, run, monkeypatch):
    results, _ = run
    mask, rho_int = clump_problem()
    labels, n = skimage_measure.label(mask, connectivity=3, return_num=True)
    delta = rho_int - np.float32(1.0)
    weights = {
        "count": np.ones_like(rho_int),
        "delta": delta,
        "mass": rho_int,
        "mass_delta": rho_int * delta,
        "above": delta > 19.5,
    }
    want = np.stack(
        [np.bincount(labels.ravel(), weights=weights[name].ravel())[1:] for name in SUMS], axis=1
    )

    def rows(table):
        return table[np.lexsort(table.T[::-1])]

    got = results["clump_table"]
    assert len(got) == n
    np.testing.assert_array_equal(rows(got), rows(want))
    # The same pieces in one process give the same clumps.
    monkeypatch.setattr(giggammf.labels, "LABEL_ELEMENTS", 2 * 24 * 20)
    single = clump_sums(mask, rho_int, mesh, SUMS, 19.5)
    np.testing.assert_array_equal(rows(np.stack([single[s] for s in SUMS], 1)), rows(got))


def test_tiled_load(run, small_cube):
    results, _ = run
    np.testing.assert_array_equal(results["tiled"], np.tile(small_cube, (2, 2, 2)))


@pytest.mark.parametrize(
    ("env", "want"),
    [
        # One task per node, 4 GPUs: all four (JAX's default would take GPU 0 only).
        ({"SLURM_STEP_TASKS_PER_NODE": "1(x2)", "SLURM_LOCALID": "0"}, [0, 1, 2, 3]),
        # One task per GPU, each task sees the 4 GPUs of its node.
        ({"SLURM_STEP_TASKS_PER_NODE": "4(x2)", "SLURM_LOCALID": "2"}, [2]),
        # Two tasks per node share the 4 GPUs.
        ({"SLURM_STEP_TASKS_PER_NODE": "2,2", "SLURM_LOCALID": "1", "SLURM_NODEID": "1"}, [2, 3]),
        # --gpus-per-task: SLURM already bound the GPUs, so the task sees only its own.
        (
            {"SLURM_STEP_TASKS_PER_NODE": "4", "SLURM_LOCALID": "3", "CUDA_VISIBLE_DEVICES": "3"},
            [0],
        ),
        (
            {
                "SLURM_STEP_TASKS_PER_NODE": "2",
                "SLURM_LOCALID": "1",
                "SLURM_GPUS_PER_TASK": "2",
                "CUDA_VISIBLE_DEVICES": "2,3",
            },
            [0, 1],
        ),
        # Not a SLURM task, or no GPUs: JAX's default.
        ({"SLURM_LOCALID": None}, None),
        ({"SLURM_LOCALID": "0", "CUDA_VISIBLE_DEVICES": None}, None),
    ],
)
def test_slurm_local_device_ids(env, want):
    environ = {"CUDA_VISIBLE_DEVICES": "0,1,2,3", **env}
    environ = {k: v for k, v in environ.items() if v is not None}
    assert slurm_local_device_ids(environ) == want


def test_slurm_local_device_ids_uneven_split_raises():
    environ = {
        "SLURM_STEP_TASKS_PER_NODE": "3",
        "SLURM_LOCALID": "0",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    }
    with pytest.raises(ValueError, match="evenly"):
        slurm_local_device_ids(environ)
