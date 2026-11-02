#!/bin/bash
# Signatures + tagging of a density cube on several nodes, one process per GPU.
# The cube is split into one x-slab per GPU over all the nodes: set --nodes (here or on the
# sbatch command line) and --gpus-per-node, with --ntasks-per-node equal to --gpus-per-node.
# nx and ny must be divisible by the total number of GPUs.
# Usage (from the repository root):
#     sbatch scripts/tag_cube_multinode.slurm [path/to/density.npy] [extra tag_cube.py args]
# 2048^3 on 2 nodes of 4x A100 80 GB (8 GPUs, ~37 GiB per GPU for the signatures):
#     sbatch scripts/tag_cube_multinode.slurm ../data/dm_density_L1_m9_DMO_0077.npy \
#         --out-dir ../data/multinode
# 2048^3 on A100 40 GB: Della's gpu40 nodes have 2 GPUs, so 8 GPUs take 4 nodes (~33 GiB per GPU):
#     sbatch --nodes=4 --gpus-per-node=2 --ntasks-per-node=2 --constraint=gpu40 \
#         scripts/tag_cube_multinode.slurm ../data/dm_density_L1_m9_DMO_0077.npy
# 4096^3 test, the 2048^3 cube tiled twice per axis, on 8 nodes (32 GPUs, ~65 GiB per GPU; more
# GPUs than Della's gpu-short QoS allows one user, 24):
#     sbatch --nodes=8 scripts/tag_cube_multinode.slurm ../data/dm_density_L1_m9_DMO_0077.npy \
#         --tile 2 --no-save
#SBATCH --job-name=jaxweb-multinode
#SBATCH --account=jdunkley
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=00:30:00

set -eo pipefail  # no -u: conda activate reads unset variables
density=${1:-../data/dm_density_L1_m9_DMO_0077.npy}
shift || true
if [[ ! -f $density ]]; then
    echo "no density cube at $density" >&2
    exit 1
fi

module purge
module load anaconda3/2026.7
conda activate /scratch/gpfs/TEYSSIER/js7501/envs/jaxweb

# Let XLA use (almost) all of each GPU, allocate on demand from CUDA's
# stream-ordered pool, and avoid the fragmentation of the default allocator.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
# NCCL carries the all-to-alls between nodes; report only problems.
export NCCL_DEBUG=WARN

nvidia-smi --query-gpu=name,memory.total --format=csv
# Every task joins one JAX job (jaxweb.initialize) and drives its own GPU.
srun python scripts/tag_cube.py "$density" "$@"
