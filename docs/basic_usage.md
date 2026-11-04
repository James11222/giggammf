# Basic Use

The `giggammf` package contains the algorithms necessary for a Multiscale Morphological Analysis (MMF) of cosmological simulations distributed across many GPUs. For a simpler implementation, we encourage the user to fallback to the CPU version of the code [`pycosmommf`](https://pycosmommf.readthedocs.io/en/latest/index.html) which has all the same ideas, just not massively parallelized. This code is designed for users who want to construct the cosmic web across massive volumes or very high resolution simulations that need grids larger than 1024^3 in size. At its core `giggammf` is a slightly modified version of the NEXUS+ algorithm [Cautun et al. 2012](https://arxiv.org/abs/1209.2043). The details of the code are described in our paper ([Sunseri et al. 2022](https://ui.adsabs.harvard.edu/abs/2023PhRvD.107b3514S/abstract)). 

In general, we start with a density field (note: we more specifically mean a 1 + δ field), smooth it with a logarithmic Gaussian smoothing filter, then compute the hessian of the smoothed density field, use the eigenvalues of the hessian matrix to calculate the structure type signatures, find the maximum signatures over a range of smoothing scales, and apply physically based threshold criterion to categorize structures within the cosmic web. The novelty comes in distributing across multiple GPUs and more specifically managing the memory footprint to stay within the GPU's VRAM. To compute the hessian of the log-smoothed density field the code uses the idea that a 3D FFT is separable in dimension to create `N_gpu` slabs of equal width. For a 2048^3 grid the slabs would be `(2048/N_gpu, 2048, 2048)` in size each. Given a slab, the GPU will compute the 2D FFT in the y and z directions, then performs an all-to-all sharing of the results across the other GPUs to then handle the x-direction of the 3D FFT. Once the slabs are created, each GPU operates exclusively on its own slab, from hessian calculation to tagging. The code mostly uses JAX except in tagging which relies on CuPy. 

The result is an array `structure` which is an integer array of the same size as the original array with 1 = nodes, 2 = filaments, 3 = walls, 4 voids. The array can then be manipulated with boolean masking to extract the voxels belonging to each structure type. The input must come from a periodic box, and `nx` and `ny` must be divisible by the number of GPUs.

**Quickstart:**

We provide quickstart scripts in `scripts/`, but the gist of using the code looks like

```python
import giggammf

giggammf.initialize()          # starts jax.distributed under SLURM/MPI; no-op for one process
mesh = giggammf.make_mesh()    # 1D mesh over every GPU

rho = giggammf.load_density("density.npy", mesh)   # δ+1 = ρ/⟨ρ⟩; each process reads only its slabs
Rs = [2 ** (n / 2) for n in range(9)]            # smoothing scales in voxels
sigmax = giggammf.maximum_signature(rho, Rs, algorithm="NEXUSPLUS", mesh=mesh)
# sigmax: float32 (3, nx, ny, nz), x-sharded: cluster, filament, wall signatures

structure, summary = giggammf.tag_structures(rho, sigmax, overdensity_threshold=370, mesh=mesh)
# structure: uint8 (nx, ny, nz), x-sharded: 1 cluster, 2 filament, 3 wall, 4 void
# summary: thresholds, dM²/dlog S curves, volume and mass fractions (PyCosmoMMF's keys)
```

**Note on Tagging:** A clump counts as virialized when its largest δ exceeds `overdensity_threshold` (`clump_statistic="max"`, the default). To match the tagging algorithm of NEXUS+, switch to `clump_statistic="mean"`. We also provide median and mass-weighted mean as options. By default we set the `overdensity_threshold=370`, this means that $\rho_{\rm max, clump} \geq 370 \overline{\rho}_{\rm m}$. 
