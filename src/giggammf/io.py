"""Sharded loading of density cubes: each process reads only its own slabs."""

from __future__ import annotations

import jax
import numpy as np

from .mesh import check_shape, make_mesh, real_sharding


def _read_tiled(source, index, tile):
    """
    The block ``index`` (a tuple of slices) of ``source`` repeated ``tile``
    times along each axis, as float32.
    """
    ranges = [np.arange(*s.indices(tile * n)) % n for s, n in zip(index, source.shape, strict=True)]
    # Read whole x-planes in increasing runs (HDF5 takes no repeated or
    # decreasing indices), then pick the y and z indices in one copy.
    x = ranges[0]
    runs = np.split(x, np.flatnonzero(np.diff(x) != 1) + 1)
    block = np.concatenate([np.asarray(source[r[0] : r[-1] + 1], dtype=np.float32) for r in runs])
    return block[:, ranges[1][:, None], ranges[2][None, :]]


def load_density(path, mesh=None, dataset=None, tile=1):
    """
    Load a 3D density cube as an x-sharded float32 JAX array.

    ``.npy`` files are memory-mapped and ``.h5``/``.hdf5`` files (give the
    ``dataset`` name) are read by hyperslab, so each process reads only the
    x-planes of its own devices and no process ever holds the whole cube.

    ``tile > 1`` repeats the cube ``tile`` times along each axis, which is
    still a periodic box: a cube ``tile`` times larger per side, for scaling
    tests, read straight from the small file.

    Args:
        path (:obj:`str` or :obj:`pathlib.Path`):
            The ``.npy``, ``.h5`` or ``.hdf5`` file.
        mesh (:obj:`jax.sharding.Mesh`, optional):
            The device mesh. Defaults to ``giggammf.make_mesh()``.
        dataset (:obj:`str`, optional):
            The dataset name; required for HDF5 files.
        tile (:obj:`int`, optional):
            The number of times the cube is repeated along each axis.
            Defaults to ``1``.

    Returns:
        (:obj:`3D float32 jax.Array`):
            The density cube, x-sharded over ``mesh``.
    """
    mesh = make_mesh() if mesh is None else mesh
    path = str(path)
    if path.endswith((".h5", ".hdf5")):
        import h5py

        if dataset is None:
            msg = "pass dataset= to read an HDF5 file"
            raise ValueError(msg)
        handle = h5py.File(path, "r")
        source = handle[dataset]
    else:
        handle = None
        source = np.load(path, mmap_mode="r")

    try:
        shape = tuple(tile * n for n in source.shape)
        check_shape(shape, mesh)
        if tile == 1:
            return jax.make_array_from_callback(
                shape,
                real_sharding(mesh),
                lambda index: np.asarray(source[index], dtype=np.float32),
            )
        return jax.make_array_from_callback(
            shape, real_sharding(mesh), lambda index: _read_tiled(source, index, tile)
        )
    finally:
        if handle is not None:
            handle.close()
