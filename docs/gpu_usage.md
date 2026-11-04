## Notes on GPU Usage

GIGGA-MMF is a memory limited code, so the only real consideration one needs when running the code is allocating enough GPU VRAM to hold the calculation in memory. GIGGA-MMF is fully parallelized across GPUs with JAX's NCCL paradigm. You just need to know how much VRAM you need, and then throw enough GPU VRAM at it (in whatever configuration available) to store the computation. Below we provide a grid of total required VRAM per grid size.

### Needed VRAM

Summed over all GPUs; the per-GPU overhead of ~1 GiB comes on top. 4096³ and 8192³ are extrapolated.

| Grid  | Cube (float32) | Total (~8.2× cube) |
| ----- | -------------- | ------------------ |
| 512³  | 0.5 GiB        | ~4 GiB             |
| 1024³ | 4 GiB          | ~33 GiB            |
| 2048³ | 32 GiB         | ~262 GiB           |
| 4096³ | 256 GiB        | ~2.1 TiB           |
| 8192³ | 2 TiB          | ~16.8 TiB          |
### Recommended number of GPUs

The number of GPUs must be a power of two that divides N. Keep the peak at or below ~70 GiB on 80 GB cards and ~35 GiB on 40 GB cards.

| Grid  | Minimum A100 80 GB (peak VRAM used per GPU) |
| ----- | ------------------------------------------- |
| 512³  | 1 (~6 GiB)                                  |
| 1024³ | 1 (~38 GiB)                                 |
| 2048³ | 4, 1 node (65 GiB)                          |
| 4096³ | 32, 8 nodes (~65 GiB)                       |
| 8192³ | 256, 64 nodes (~65 GiB)                     |

**Note on Running the Code:** Set these before starting Python (see `benchmarks/run_a100_node.slurm`), this ensures that the code has access to the maximum possible amount of memory to work with. 

```shell
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
```
