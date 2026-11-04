#!/bin/bash
# NEXUS+ tagging of a density cube on GPUs. Usage: sbatch scripts/quickstart.sh density.npy
#
# How many GPUs: the run needs about 8.2x the cube's size in GPU memory, in total.
#     2048^3 float32 cube = 32 GiB  ->  ~262 GiB  ->  4x A100 80 GB (as below)
#     4096^3 float32 cube = 256 GiB ->  ~2.1 TiB  ->  32x A100 80 GB (8 nodes of 4)
# So GPUs >= 8.2 x cube size / memory per GPU, and the grid size must be divisible by
# the number of GPUs. Set --nodes and --gpus-per-node to reach it, with
# --ntasks-per-node equal to --gpus-per-node (one process per GPU).
#SBATCH --job-name=giggammf-quickstart
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:30:00

# Activate the environment where you ran `pip install -e ".[gpu]"`, e.g.:
# module load anaconda3 && conda activate giggammf

# Let JAX use almost all of each GPU, allocating on demand.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async

srun python scripts/quickstart.py "$1"
