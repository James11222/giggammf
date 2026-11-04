# giggammf

NEXUS and NEXUS+ cosmic web classification ([Cautun et al. 2013](https://arxiv.org/abs/1209.2043)) on many GPUs with JAX.

giggammf is a from-scratch, multi-GPU rewrite of [PyCosmoMMF](https://github.com/James11222/PyCosmoMMF). It is built so that a 2048³ single-precision density cube runs on one node of 4× A100 80 GB, and it scales to more GPUs and nodes with the same code.

Status: multi-scale NEXUS and NEXUS+ signatures and cluster / filament / wall / void tagging on one node or several (any number of GPUs that divides `nx` and `ny`).

## Install

```bash
pip install -e ".[gpu]"     # JAX with CUDA 12, and CuPy
pip install -e ".[test]"    # pytest, scipy, and PyCosmoMMF for the parity tests
```

## Use

```python
import giggammf

giggammf.initialize()          # starts jax.distributed under SLURM/MPI; no-op for one process
mesh = giggammf.make_mesh()    # 1D mesh over every GPU of every node

rho = giggammf.load_density("density.npy", mesh)   # δ+1 = ρ/⟨ρ⟩; each process reads only its slabs
Rs = [2 ** (n / 2) for n in range(9)]            # smoothing scales in voxels
sigmax = giggammf.maximum_signature(rho, Rs, algorithm="NEXUSPLUS", mesh=mesh)
# sigmax: float32 (3, nx, ny, nz), x-sharded: cluster, filament, wall signatures

structure, summary = giggammf.tag_structures(rho, sigmax, overdensity_threshold=370, mesh=mesh)
# structure: uint8 (nx, ny, nz), x-sharded: 1 cluster, 2 filament, 3 wall, 4 void
# summary: thresholds, dM²/dlog S curves, volume and mass fractions (PyCosmoMMF's keys)
```

A clump counts as virialized when its largest δ exceeds `overdensity_threshold` (`clump_statistic="max"`, the default). PyCosmoMMF uses the clump's mean δ instead, which leaves the nodes too small; `clump_statistic="mean"` reproduces it, and `"mass_weighted"` and `"median"` are also available (`giggammf.tagging.virialized_clumps`). `scripts/tag_cube.py --clump-statistic max mean mass_weighted median` tags a cube once per statistic from the same signatures.

NEXUS+ takes log10 ρ, so empty voxels need a floor: by default ρ is clipped to at least `giggammf.density_floor(rho, mesh)`, a tenth of the 1% quantile of the positive voxels (pass `floor=` to override). PyCosmoMMF adds `1e-16` instead, which puts empty voxels 16 decades down and gives spurious walls around every empty region.

The input must come from a periodic box, and `nx` and `ny` must be divisible by the number of GPUs.

On GPUs, set these before starting Python (see `benchmarks/run_a100_node.slurm`):

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
```

## Several nodes

Run one task per GPU under `srun` and change only the node count: `giggammf.initialize()` joins every task into one JAX job, `make_mesh()` spans all their GPUs, and the cube is split into one x-slab per GPU across the nodes. `scripts/tag_cube_multinode.slurm` does this (2 nodes of 4× A100 80 GB as written):

```bash
sbatch scripts/tag_cube_multinode.slurm ../data/dm_density_L1_m9_DMO_0077.npy --out-dir ../data/multinode
sbatch --nodes=8 scripts/tag_cube_multinode.slurm ../data/dm_density_L1_m9_DMO_0077.npy --tile 2 --no-save
```

One task per node driving all its GPUs works too (`--ntasks-per-node=1`): `initialize()` gives each task its share of the GPUs it sees, where JAX's default would give every task a single GPU. The fields only cross nodes in the all-to-alls of the FFTs (NCCL); the clump labelling sends only label planes and small tables between processes.

`--tile k` (`load_density(..., tile=k)`) repeats a cube k times per axis, a periodic box k times larger per side, read straight from the small file. The tiled cube's structure filter is the small cube's filter tiled (`benchmarks/compare_filters.py --tile k` checks it voxel by voxel); only the virialized fractions move a little, because clumps that cross the faces of the small box are joined inside the tiled one.

Per-GPU peak of the signatures from XLA's buffer assignment (`benchmarks/memory_estimate.py`, NEXUS+ with the log-spectrum offloaded; the tagging needs less):

| Cube | GPUs | Slab per GPU | Peak per GPU |
| --- | --- | --- | --- |
| 2048³ | 4 | 8 GiB | 65 GiB (measured 65.3) |
| 2048³ | 8 | 4 GiB | 33 GiB (fits A100 40 GB) |
| 4096³ | 32 | 8 GiB | 65 GiB |
| 4096³ | 16 | 16 GiB | 129 GiB (does not fit) |

The per-GPU work and memory depend only on the slab, so 4096³ on 32 GPUs runs the same per-GPU programs as 2048³ on 4. On Della, the gpu40 nodes have 2 GPUs each (8 GPUs is 4 nodes), and the `gpu-short` QoS allows 24 GPUs per user, so 4096³ (32 GPUs of 80 GB) needs a larger allocation.

## How it works

- Real-space fields are split along x into one slab per GPU; k-space fields are split along k_y. The 3D FFT is a local 2D FFT, one `all_to_all`, and a local 1D FFT, each done a few planes or columns at a time so no full-slab FFT work area or transpose is ever made. The NEXUS+ smoothing (inverse FFT, 10^x, forward FFT) is fused into one such pass.
- The Hessian H_ij = −k_i k_j R² f̂ is built from three partial inverse transforms over x only (k_x to the power 0, 1, 2). Each takes one chunked all-to-all. The six components are then made a few x-planes at a time, straight into the eigenvalues, signatures and running max, so they never exist at full size.
- Eigenvalues use the closed-form Smith (1961) solver, the same as PyCosmoMMF.
- Tagging follows PyCosmoMMF's `calc_structure_bools` step for step. Clumps are labelled per GPU (CuPy; SciPy on CPU devices) in pieces of at most 2³⁰ voxels. The label pairs that touch across piece and slab boundaries are found on the GPUs (each slab receives the next slab's first label plane), and only those pairs and the sums of the labels in them are gathered on every process for a connected-components pass. Each process keeps the clumps that lie wholly in its slabs. Like PyCosmoMMF, clumps are not joined across the box faces.

Measured on one node of 4× A100 80 GB (Della), 2048³, 9 scales, with the caller keeping the density on the GPUs (`benchmarks/run_a100_node.slurm`):

| Algorithm | Peak per GPU | Signatures |
| --- | --- | --- |
| NEXUS | 65.3 GiB | 13 s |
| NEXUS+ with `offload=True` (log-spectrum kept in host memory) | 65.3 GiB | 46 s |
| NEXUS+ with `offload=False` | 73.3 GiB | 25 s |

`offload=None` (the default) turns host offload on when NEXUS+ would need more than 90% of the memory XLA may use (`XLA_PYTHON_CLIENT_MEM_FRACTION` of the device), which it does at 2048³ on 4× 80 GB. `benchmarks/memory_estimate.py` gives the same numbers from XLA's buffer assignment without a GPU.

Tagging needs about 40 GiB per GPU at 2048³ on 4 GPUs (density, signatures and a few masks), plus CuPy's labelling workspace for one 2³⁰-voxel piece.

## Tests

```bash
pytest
```

`tests/conftest.py` simulates 4 CPU devices, so the sharding and all-to-alls are tested without a GPU. `tests/test_multiprocess.py` runs 2 processes of 2 CPU devices each, as on 2 nodes, and checks them against one process. The parity tests compare smoothing, Hessians, signatures and the tagging (thresholds and `structure_filter`) with PyCosmoMMF, and clump labelling with scikit-image.
