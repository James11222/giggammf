"""Connected clumps of a sharded boolean mask, with per-clump sums.

Each device labels its own x-slab (CuPy on GPUs, SciPy on CPU devices, both
26-connected) in pieces of at most ``LABEL_ELEMENTS`` voxels; the devices of a
process work at once, one host thread each. Clumps that touch across a piece
or slab boundary are then joined: the last x-plane of every piece is compared
with the first plane of the next one, on the device, and every pair of
touching labels becomes an edge. Each slab receives the first plane of the
next slab with a ``ppermute``, so slabs on other processes (other nodes) are
joined in the same way.

A label that is in no edge is a whole clump and stays with the process that
labelled it. Only the edges and the per-label sums of the labels in them are
gathered on every process, where a connected-components pass joins them;
process 0 keeps the joined clumps. So each clump belongs to exactly one
process.

Clumps are not joined across the faces of the box, to match PyCosmoMMF
(scikit-image's labelling is not periodic).
"""

from __future__ import annotations

import functools
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.csgraph
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from .distributed import allgather
from .fft import largest_divisor_at_most
from .mesh import AXIS, CHUNK_ELEMENTS, as_sharded, real_sharding

_STRUCTURE = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity

# Largest piece labelled at once (see ``clump_sums``).
LABEL_ELEMENTS = 2**30

# The per-clump sums ``clump_sums`` can make, as functions of ρ and δ = ρ - 1:
# the voxel count, Σδ, Σρ (the mass), Σρδ, and the number of voxels with δ
# above a threshold.
SUMS = ("count", "delta", "mass", "mass_delta", "above")
_INTEGER_SUMS = ("count", "above")


def _label_shard(mask):
    """Label one device's uint8 mask in place on that device: ``(labels, n)``."""
    device = next(iter(mask.devices()))
    if device.platform == "gpu":
        import cupy as cp
        import cupyx.scipy.ndimage

        with cp.cuda.Device(device.local_hardware_id):
            labels, n = cupyx.scipy.ndimage.label(
                cp.from_dlpack(mask).astype(bool), structure=_STRUCTURE
            )
            return jax.dlpack.from_dlpack(labels.astype(cp.int32, copy=False)), int(n)
    labels, n = scipy.ndimage.label(np.asarray(mask), structure=_STRUCTURE)
    return jax.device_put(labels.astype(np.int32), device), int(n)


@functools.partial(jax.jit, static_argnums=(2, 3))
def _segment_sums(labels, rho, num_segments, names, threshold):
    """
    The ``names`` sums per label (index 0 is background): float32 sums,
    int32 counts, a few x-planes at a time so no slab-sized temporary is made.
    """
    nx, ny, nz = labels.shape
    planes = largest_divisor_at_most(nx, CHUNK_ELEMENTS // (ny * nz))

    def body(c, sums):
        lab = lax.dynamic_slice_in_dim(labels, c * planes, planes, axis=0).ravel()
        r = lax.dynamic_slice_in_dim(rho, c * planes, planes, axis=0).ravel()
        delta = r - 1.0
        weights = {
            "count": lambda: jnp.ones_like(lab),
            "delta": lambda: delta,
            "mass": lambda: r,
            "mass_delta": lambda: r * delta,
            "above": lambda: (delta > threshold).astype(jnp.int32),
        }
        return tuple(
            s + jax.ops.segment_sum(weights[name](), lab, num_segments=num_segments)
            for s, name in zip(sums, names, strict=True)
        )

    init = tuple(
        jnp.zeros(num_segments, jnp.int32 if name in _INTEGER_SUMS else jnp.float32)
        for name in names
    )
    return lax.fori_loop(0, nx // planes, body, init)


def _bucket(n):
    """Round ``n`` up to a power of two, so ``_segment_sums`` compiles a few sizes only."""
    return 1 << max(4, int(n).bit_length())


def _pair_keys(xp, last_plane, first_plane):
    """
    ``(a << 32) | b`` for every distinct label pair ``(a, b)`` that touches
    across a boundary under 26-connectivity; ``xp`` is NumPy or CuPy.
    """
    ny, nz = last_plane.shape
    keys = []
    for dy in (-1, 0, 1):
        for dz in (-1, 0, 1):
            a = last_plane[max(0, -dy) : ny - max(0, dy), max(0, -dz) : nz - max(0, dz)]
            b = first_plane[max(0, dy) : ny + min(0, dy), max(0, dz) : nz + min(0, dz)]
            both = (a > 0) & (b > 0)
            keys.append((a[both].astype(xp.int64) << 32) | b[both].astype(xp.int64))
    return xp.unique(xp.concatenate(keys))


def _touching_pairs(last_plane, first_plane):
    """
    Label pairs ``(a, b)`` that touch across a boundary under 26-connectivity,
    an int64 ``(k, 2)`` NumPy array. The planes are JAX arrays on one device;
    on a GPU the pairs are found there, with CuPy, and only they are copied.
    """
    device = next(iter(last_plane.devices()))
    if device.platform == "gpu":
        import cupy as cp

        with cp.cuda.Device(device.local_hardware_id):
            keys = _pair_keys(cp, cp.from_dlpack(last_plane), cp.from_dlpack(first_plane)).get()
    else:
        keys = _pair_keys(np, np.asarray(last_plane), np.asarray(first_plane))
    return np.stack([keys >> 32, keys & 0xFFFFFFFF], axis=1)


class _Slab(NamedTuple):
    """One device's slab, labelled in pieces along x."""

    counts: list  # number of labels in each piece
    sums: list  # per piece, one host array per name; label l at index l - 1
    pairs: list  # touching label pairs between pieces p and p + 1
    first: jax.Array  # labels of the slab's first x-plane, (1, ny, nz), on its device
    last: jax.Array  # labels of its last x-plane, (ny, nz), on its device


def _label_slab(mask, rho, names, threshold):
    """
    Label one device's slab in pieces of at most LABEL_ELEMENTS voxels (a
    2048³ slab on one of 4 GPUs has exactly 2^31, where int32 indexing can
    break), with the per-label sums of each piece and the label pairs that
    touch between consecutive pieces.
    """
    nx, ny, nz = mask.shape
    planes = largest_divisor_at_most(nx, LABEL_ELEMENTS // (ny * nz))
    counts, sums, pairs = [], [], []
    first = last = None
    for start in range(0, nx, planes):
        piece = mask if planes == nx else mask[start : start + planes]
        labels, n = _label_shard(piece)
        rho_piece = rho if planes == nx else rho[start : start + planes]
        piece_sums = _segment_sums(labels, rho_piece, _bucket(n + 1), names, threshold)
        counts.append(n)
        sums.append([np.asarray(s)[1 : n + 1] for s in piece_sums])
        if first is None:
            first = labels[:1]
        else:
            pairs.append(_touching_pairs(last, labels[0]))
        last = labels[-1]
        del labels, piece, rho_piece
    return _Slab(counts, sums, pairs, first, last)


@functools.cache
def _from_next_slab(mesh):
    """Jitted: each slab's block of an x-sharded array goes to the previous slab (cyclic)."""
    n = mesh.shape[AXIS]
    perm = [(j, (j - 1) % n) for j in range(n)]
    fn = jax.shard_map(
        lambda block: lax.ppermute(block, AXIS, perm),
        mesh=mesh,
        in_specs=P(AXIS),
        out_specs=P(AXIS),
    )
    return jax.jit(fn)


def _next_first_planes(slabs, devices, mesh):
    """The first label plane of the next slab along x for each local slab, on its device."""
    ny, nz = slabs[0].first.shape[1:]
    planes = jax.make_array_from_single_device_arrays(
        (mesh.shape[AXIS], ny, nz),
        NamedSharding(mesh, P(AXIS, None, None)),
        [slab.first for slab in slabs],
    )
    received = {s.device: s.data[0] for s in _from_next_slab(mesh)(planes).addressable_shards}
    return [received[d] for d in devices]


def _sum_dtype(name):
    return np.int64 if name in _INTEGER_SUMS else np.float32


def clump_sums(mask, rho, mesh, names=("count", "delta"), threshold=0.0):
    """
    Per-clump sums over every 26-connected clump of ``mask``, across all slabs.

    Args:
        mask (:obj:`3D bool jax.Array`):
            x-sharded boolean or uint8 array ``(nx, ny, nz)``.
        rho (:obj:`3D float32 jax.Array`):
            δ+1 = ρ/⟨ρ⟩, same shape (δ is formed per slab, on the fly).
        mesh (:obj:`jax.sharding.Mesh`):
            The device mesh; both arrays are laid out as x-slabs on it.
        names (:obj:`tuple` of :obj:`str`, optional):
            Which sums to make, from ``SUMS``: ``"count"`` (voxels),
            ``"delta"`` (Σδ), ``"mass"`` (Σρ), ``"mass_delta"`` (Σρδ) and
            ``"above"`` (voxels with δ > ``threshold``). Defaults to
            ``("count", "delta")``.
        threshold (:obj:`float`, optional):
            The δ threshold of ``"above"``. Defaults to ``0.0``.

    Returns:
        (:obj:`dict`):
            A dict of NumPy arrays keyed by ``names``, one value per clump,
            the clumps in the same (arbitrary) order in every array: float32
            sums, int64 counts. With several processes, each process returns
            its own share of the clumps and each clump is in exactly one
            share, so totals over clumps need a sum over processes
            (``distributed.process_sum``).
    """
    mask = as_sharded(mask, real_sharding(mesh), jnp.uint8)
    rho = as_sharded(rho, real_sharding(mesh), jnp.float32)
    names = tuple(names)
    threshold = jnp.float32(threshold)
    n_slabs = mesh.shape[AXIS]
    nx_slab = mask.shape[0] // n_slabs

    def start(shard):
        return shard.index[0].start or 0

    rho_shards = {start(s): s.data for s in rho.addressable_shards}
    shards = sorted(mask.addressable_shards, key=start)
    slab_ids = [start(s) // nx_slab for s in shards]
    across = []
    # One thread per device, so the devices label at the same time: each
    # thread mostly waits on its device, with the GIL released.
    with ThreadPoolExecutor(max_workers=len(shards)) as pool:
        slabs = list(
            pool.map(lambda s: _label_slab(s.data, rho_shards[start(s)], names, threshold), shards)
        )
        if n_slabs > 1:
            next_first = _next_first_planes(slabs, [s.device for s in shards], mesh)
            across = list(pool.map(_touching_pairs, [slab.last for slab in slabs], next_first))

    # Label l of piece p of slab i is node offsets[i, p] + l - 1 of the graph.
    table = allgather(
        np.array([[i, *slab.counts] for i, slab in zip(slab_ids, slabs, strict=True)], np.int64)
    )
    counts = table[np.argsort(table[:, 0]), 1:]
    offsets = (np.cumsum(counts) - counts.ravel()).reshape(counts.shape)

    edges = [np.zeros((0, 2), dtype=np.int64)]
    for k, (i, slab) in enumerate(zip(slab_ids, slabs, strict=True)):
        for p, pairs in enumerate(slab.pairs):
            edges.append(pairs + offsets[i, [p, p + 1]] - 1)
        if i + 1 < n_slabs:  # not across the face of the box
            edges.append(across[k] + np.array([offsets[i, -1], offsets[i + 1, 0]]) - 1)
    edges = allgather(np.concatenate(edges))
    joined = np.unique(edges)

    ids = np.concatenate(
        [
            offsets[i, p] + np.arange(n)
            for i, slab in zip(slab_ids, slabs, strict=True)
            for p, n in enumerate(slab.counts)
        ]
    )
    columns = {
        name: np.concatenate([piece[k] for slab in slabs for piece in slab.sums]).astype(
            _sum_dtype(name)
        )
        for k, name in enumerate(names)
    }
    crosses = np.isin(ids, joined)
    out = {name: column[~crosses] for name, column in columns.items()}

    # The labels that cross a boundary, from every process, joined into clumps.
    rows = np.empty(
        np.count_nonzero(crosses),
        dtype=[("id", np.int64)] + [(name, _sum_dtype(name)) for name in names],
    )
    rows["id"] = ids[crosses]
    for name in names:
        rows[name] = columns[name][crosses]
    rows = allgather(rows)
    graph = scipy.sparse.coo_matrix(
        (np.ones(len(edges), dtype=np.int8), tuple(np.searchsorted(joined, edges).T)),
        shape=(len(joined), len(joined)),
    )
    n_clumps, clump_of = scipy.sparse.csgraph.connected_components(graph, directed=False)
    if jax.process_index() == 0:
        clump = clump_of[np.searchsorted(joined, rows["id"])]
        for name in names:
            joined_sums = np.zeros(n_clumps, dtype=_sum_dtype(name))
            np.add.at(joined_sums, clump, rows[name])
            out[name] = np.concatenate([out[name], joined_sums])
    return out
