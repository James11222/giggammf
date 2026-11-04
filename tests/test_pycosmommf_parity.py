"""giggammf against PyCosmoMMF, the reference implementation, on 4 simulated devices."""

from __future__ import annotations

import numpy as np
import pytest
from cubes import lognormal_cube

import giggammf
from giggammf.signatures import _chunking

pcm = pytest.importorskip("pycosmommf")

N = 32
RS = [1.0, 2.0**0.5, 2.0, 2.0**1.5, 4.0]


@pytest.fixture(scope="module")
def cube():
    return lognormal_cube(N)


@pytest.mark.parametrize("algorithm", ["NEXUS", "NEXUSPLUS"])
def test_smoothing_matches(mesh, cube, algorithm):
    kv = pcm.wavevectors3D(cube.shape)
    smooth_ref = pcm.smooth_gauss if algorithm == "NEXUS" else pcm.smooth_loggauss
    want = smooth_ref(cube + np.float32(1e-16), 2.0, kv)
    got = np.asarray(giggammf.smooth(cube, 2.0, mesh, algorithm=algorithm, floor=cube.min()))
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("algorithm", ["NEXUS", "NEXUSPLUS"])
def test_hessian_matches(mesh, cube, algorithm):
    kv = pcm.wavevectors3D(cube.shape)
    smooth_ref = pcm.smooth_gauss if algorithm == "NEXUS" else pcm.smooth_loggauss
    smoothed = smooth_ref(cube + np.float32(1e-16), 2.0, kv)
    want = np.moveaxis(pcm.fast_hessian_from_smoothed(smoothed, 2.0, kv), -1, 0)
    got = np.asarray(giggammf.hessian(cube, 2.0, mesh, algorithm=algorithm, floor=cube.min()))
    np.testing.assert_allclose(got, want, atol=1e-5 * np.abs(want).max())


@pytest.mark.parametrize("algorithm", ["NEXUS", "NEXUSPLUS"])
def test_maximum_signature_matches(mesh, cube, algorithm):
    want = pcm.maximum_signature(RS, cube, algorithm=algorithm)
    got = giggammf.to_pycosmommf_layout(
        giggammf.maximum_signature(cube, RS, algorithm=algorithm, mesh=mesh, floor=cube.min())
    )
    assert got.dtype == np.float32
    for c in range(3):
        scale = want[..., c].max()
        np.testing.assert_allclose(got[..., c], want[..., c], atol=5e-4 * scale)


def test_chunking_does_not_change_the_result(mesh):
    cube = lognormal_cube(64, seed=3)
    one_chunk = giggammf.maximum_signature(cube, RS[:3], mesh=mesh, chunk_elements=2**30)
    # 64·33 elements per k_y column and 64·64 per x-plane: 2**13 gives 8 chunks
    # of each kind per device.
    assert _chunking(cube.shape, 4, 2**13) == (8, 8)
    many_chunks = giggammf.maximum_signature(cube, RS[:3], mesh=mesh, chunk_elements=2**13)
    one_chunk, many_chunks = np.asarray(one_chunk), np.asarray(many_chunks)
    # Not bit-identical: XLA's FFTs and reductions round differently for
    # different batch sizes, at the float32 ulp level. The NEXUS+ smoothing is
    # chunked too, so the renormalising sum changes order; on this cube the two
    # differ by 7.5e-6 of the maximum, while each is 2.3-3.0e-5 from PyCosmoMMF.
    np.testing.assert_allclose(many_chunks, one_chunk, rtol=1e-5, atol=1e-5 * one_chunk.max())


@pytest.mark.parametrize(
    "make", [lambda: pcm.sphere(N, 6), lambda: pcm.cylinder(N, 4), lambda: pcm.wall(N)]
)
def test_synthetic_shapes_match(mesh, make):
    rho = make().astype(np.float32) + np.float32(1.0)
    rho /= rho.mean()
    want = pcm.maximum_signature(RS, rho, algorithm="NEXUS")
    got = giggammf.to_pycosmommf_layout(
        giggammf.maximum_signature(rho, RS, algorithm="NEXUS", mesh=mesh)
    )
    # Highly symmetric shapes have near-degenerate Hessians, where float32
    # rounding in the trig eigen-solver moves the signature by ~1e-3 relative.
    np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-4 * want.max())
    # The dominant structure type agrees voxel by voxel wherever it is clear.
    strong = want.max(axis=-1) > 0.05 * want.max()
    assert np.mean(got.argmax(-1)[strong] == want.argmax(-1)[strong]) > 0.999


def test_negative_density_raises_for_nexusplus(mesh, cube):
    with pytest.raises(ValueError, match="negative"):
        giggammf.maximum_signature(cube - 1.0, RS, algorithm="NEXUSPLUS", mesh=mesh)
    giggammf.maximum_signature(cube - 1.0, RS[:1], algorithm="NEXUS", mesh=mesh)


def test_load_density_npy(mesh, cube, tmp_path):
    path = tmp_path / "cube.npy"
    np.save(path, cube.astype(np.float64))
    loaded = giggammf.load_density(path, mesh)
    assert loaded.dtype == np.float32
    assert loaded.sharding.spec == giggammf.mesh.REAL_SPEC
    np.testing.assert_array_equal(np.asarray(loaded), cube)


def test_offload_gives_the_same_result(mesh, cube):
    on_device = giggammf.maximum_signature(cube, RS[:2], mesh=mesh, offload=False)
    offloaded = giggammf.maximum_signature(cube, RS[:2], mesh=mesh, offload=True)
    np.testing.assert_array_equal(np.asarray(on_device), np.asarray(offloaded))


def test_density_floor_is_an_exact_quantile(mesh, cube):
    rho = cube.copy()
    rho[::3, ::5, ::2] = 0.0
    positive = rho[rho > 0]
    for quantile in (0.01, 0.3):
        want = np.float32(0.1) * np.quantile(positive, quantile, method="inverted_cdf")
        assert giggammf.density_floor(rho, mesh, quantile=quantile) == want


def test_empty_voxels_are_clipped_to_the_floor(mesh, cube):
    rho = cube.copy()
    rho[::3, ::5, ::2] = 0.0
    floor = giggammf.density_floor(rho, mesh)
    want = pcm.maximum_signature(RS, np.maximum(rho, floor), algorithm="NEXUSPLUS")
    got = giggammf.to_pycosmommf_layout(giggammf.maximum_signature(rho, RS, mesh=mesh))
    for c in range(3):
        scale = want[..., c].max()
        np.testing.assert_allclose(got[..., c], want[..., c], atol=5e-4 * scale)
