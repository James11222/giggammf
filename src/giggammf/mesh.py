"""Device mesh, multi-process start-up and the shardings every module uses.

giggammf uses a 1D mesh with one axis, ``"x"``, over every GPU of every process,
so a job on several nodes splits the cube into one slab per GPU across all of
them.

* Real-space fields of shape ``(nx, ny, nz)`` are split along x: each device
  holds ``nx / P`` consecutive x-planes (``REAL_SPEC``).
* Signature arrays of shape ``(3, nx, ny, nz)`` are split the same way
  (``SIG_SPEC``).
* k-space fields (the rfft half-spectrum, ``(nx, ny, nz // 2 + 1)``) are split
  along k_y (``KSPACE_SPEC``), so the x-transform is local.
"""

from __future__ import annotations

import os

import jax
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

AXIS = "x"

REAL_SPEC = P(AXIS, None, None)
KSPACE_SPEC = P(None, AXIS, None)
SIG_SPEC = P(None, AXIS, None, None)

# Elements per chunk in the chunked loops (2^25 float32 = 128 MiB). A loop
# body holds a few tens of chunk-sized temporaries, so ~2-4 GiB per device.
CHUNK_ELEMENTS = 2**25

_MULTI_PROCESS_ENV = ("SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE")


def initialize(**kwargs):
    """
    Start JAX's multi-process runtime when the job runs more than one process.

    Under SLURM or MPI with more than one task this calls
    ``jax.distributed.initialize(**kwargs)``, which finds the coordinator from
    the environment. With a single process (one node, any number of GPUs) it
    does nothing, so the same script runs everywhere. Call it before any other
    JAX call.

    Under SLURM each task drives its share of the GPUs it can see (see
    ``slurm_local_device_ids``), so one task per GPU and one task per node
    both work: ``srun`` with ``--ntasks-per-node`` equal to
    ``--gpus-per-node``, or with ``--ntasks-per-node=1``. Pass
    ``local_device_ids`` to choose yourself.

    Args:
        **kwargs:
            Passed on to ``jax.distributed.initialize``. Any keyword also
            starts the runtime with a single process.
    """
    n_tasks = max(int(os.environ.get(name, "1")) for name in _MULTI_PROCESS_ENV)
    if n_tasks <= 1 and not kwargs:
        return
    if "local_device_ids" not in kwargs:
        local_device_ids = slurm_local_device_ids(os.environ)
        if local_device_ids is not None:
            kwargs["local_device_ids"] = local_device_ids
    jax.distributed.initialize(**kwargs)


def _tasks_on_node(environ):
    """Number of tasks of this SLURM step on this node, from e.g. ``"4(x2),3"``."""
    spec = environ.get("SLURM_STEP_TASKS_PER_NODE") or environ.get("SLURM_TASKS_PER_NODE")
    if not spec:
        return 1
    counts = []
    for item in spec.split(","):
        count, _, repeat = item.partition("(x")
        counts += [int(count)] * (int(repeat.rstrip(")")) if repeat else 1)
    return counts[int(environ.get("SLURM_NODEID", "0"))]


def slurm_local_device_ids(environ):
    """
    The GPUs a SLURM task should drive.

    JAX's default gives each task the one GPU numbered by its ``SLURM_LOCALID``.
    That is right for one task per GPU when every task sees all the GPUs of its
    node, but with one task per node it drives one GPU out of four, and when
    SLURM binds GPUs to tasks (``--gpus-per-task``) that GPU may not exist.
    Here the GPUs a task sees are split evenly between the tasks on its node,
    unless SLURM bound them to tasks: then each task drives all it sees.

    Args:
        environ (:obj:`dict`):
            The environment variables, e.g. ``os.environ``.

    Returns:
        (:obj:`list` of :obj:`int`):
            The ids, among the GPUs the task can see, of the GPUs it should
            drive; ``None`` outside a SLURM task with GPUs, to keep JAX's
            default.
    """
    visible = environ.get("CUDA_VISIBLE_DEVICES", "")
    if "SLURM_LOCALID" not in environ or not visible:
        return None
    n_visible = len(visible.split(","))
    n_tasks = _tasks_on_node(environ)
    if "SLURM_GPUS_PER_TASK" in environ or n_visible < n_tasks:
        return list(range(n_visible))
    if n_visible % n_tasks:
        msg = (
            f"{n_tasks} SLURM tasks on this node cannot share its {n_visible} GPUs evenly: "
            "use one task per GPU or one task per node"
        )
        raise ValueError(msg)
    per_task = n_visible // n_tasks
    first = int(environ["SLURM_LOCALID"]) * per_task
    return list(range(first, first + per_task))


def make_mesh(devices=None):
    """
    The 1D device mesh, with one axis ``"x"``.

    Args:
        devices (:obj:`list` of :obj:`jax.Device`, optional):
            The devices of the mesh. Defaults to every device of every process.

    Returns:
        (:obj:`jax.sharding.Mesh`):
            A 1D ``Mesh`` over ``devices``, sorted by process then device id.
    """
    devices = jax.devices() if devices is None else list(devices)
    # Not jax.make_mesh: on GPUs every node is a separate "slice", which it
    # refuses. Each process's slabs are kept consecutive along x, so most
    # clump boundaries are joined within a process.
    devices = sorted(devices, key=lambda d: (d.process_index, d.id))
    # Auto axes: shard_map reshards inputs that arrive with another layout
    # (a NumPy array, say) instead of raising.
    return jax.sharding.Mesh(np.array(devices), (AXIS,), axis_types=(jax.sharding.AxisType.Auto,))


def real_sharding(mesh):
    """
    The sharding of real-space fields.

    Args:
        mesh (:obj:`jax.sharding.Mesh`):
            The device mesh.

    Returns:
        (:obj:`jax.sharding.NamedSharding`):
            ``(nx, ny, nz)`` fields split along x (``REAL_SPEC``).
    """
    return NamedSharding(mesh, REAL_SPEC)


def kspace_sharding(mesh):
    """
    The sharding of k-space fields.

    Args:
        mesh (:obj:`jax.sharding.Mesh`):
            The device mesh.

    Returns:
        (:obj:`jax.sharding.NamedSharding`):
            ``(nx, ny, nz // 2 + 1)`` half-spectra split along k_y
            (``KSPACE_SPEC``).
    """
    return NamedSharding(mesh, KSPACE_SPEC)


def sig_sharding(mesh):
    """
    The sharding of signature arrays.

    Args:
        mesh (:obj:`jax.sharding.Mesh`):
            The device mesh.

    Returns:
        (:obj:`jax.sharding.NamedSharding`):
            ``(3, nx, ny, nz)`` arrays split along x (``SIG_SPEC``).
    """
    return NamedSharding(mesh, SIG_SPEC)


def check_shape(shape, mesh):
    """
    Raise ``ValueError`` unless a cube of ``shape`` can be split over ``mesh``.

    ``nx`` and ``ny`` must be divisible by the number of devices, and ``nz``
    must be even.

    Args:
        shape (:obj:`tuple` of :obj:`int`):
            The shape ``(nx, ny, nz)`` of the cube.
        mesh (:obj:`jax.sharding.Mesh`):
            The device mesh.
    """
    if len(shape) != 3:
        msg = f"giggammf works on 3D fields, got shape {shape}"
        raise ValueError(msg)
    nx, ny, nz = shape
    n_dev = mesh.shape[AXIS]
    if nx % n_dev or ny % n_dev:
        msg = (
            f"nx = {nx} and ny = {ny} must both be divisible by the number of "
            f"devices ({n_dev}): real space is split along x and k-space along k_y"
        )
        raise ValueError(msg)
    if nz % 2:
        msg = f"nz = {nz} must be even for the real-to-complex FFT"
        raise ValueError(msg)


def as_sharded(x, sharding, dtype):
    """
    Convert ``x`` to a sharded JAX array.

    NumPy input is split on the host and sent slab by slab, never staged
    whole on one device.

    Args:
        x (:obj:`np.ndarray` or :obj:`jax.Array`):
            The array to convert.
        sharding (:obj:`jax.sharding.NamedSharding`):
            The sharding of the result.
        dtype (:obj:`numpy.dtype`):
            The dtype of the result.

    Returns:
        (:obj:`jax.Array`):
            ``x`` as a JAX array of ``dtype`` with ``sharding``.
    """
    if isinstance(x, jax.Array):
        return jax.device_put(x.astype(dtype), sharding)
    return jax.device_put(np.asarray(x, dtype=dtype), sharding)
