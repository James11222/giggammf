"""
Compare two structure filters voxel by voxel, a few x-planes at a time.

    python benchmarks/compare_filters.py reference.npy other.npy [--tile k]

Prints the fraction of voxels whose type differs and the confusion matrix
(rows: reference, columns: other; 1 cluster, 2 filament, 3 wall, 4 void).
With ``--tile k``, ``other`` is the filter of the reference cube tiled k times
per axis (``scripts/tag_cube.py --tile k``), and each of its k³ copies is
compared with the reference.
"""

from __future__ import annotations

import argparse

import numpy as np

KINDS = ("cluster", "filament", "wall", "void")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("other")
    parser.add_argument("--tile", type=int, default=1)
    parser.add_argument("--planes", type=int, default=8, help="x-planes per block")
    args = parser.parse_args()

    a = np.load(args.reference, mmap_mode="r")
    b = np.load(args.other, mmap_mode="r")
    k = args.tile
    if b.shape != tuple(k * n for n in a.shape):
        msg = f"shapes {a.shape} and {b.shape} do not match with --tile {k}"
        raise ValueError(msg)
    nx = a.shape[0]
    planes = max(d for d in range(1, args.planes + 1) if nx % d == 0)

    confusion = np.zeros(25, dtype=np.int64)
    for start in range(0, b.shape[0], planes):
        ref = np.asarray(a[start % nx : start % nx + planes])
        if k > 1:
            ref = np.tile(ref, (1, k, k))
        other = np.asarray(b[start : start + planes])
        confusion += np.bincount((5 * ref + other).ravel(), minlength=25)
    confusion = confusion.reshape(5, 5)[1:, 1:]

    total = confusion.sum()
    print(f"{args.other} against {args.reference}" + (f" tiled {k}x" if k > 1 else ""))
    print(f"  voxels differing: {total - np.trace(confusion)} of {total}", end="")
    print(f" ({(total - np.trace(confusion)) / total:.3e})")
    print("  " + " " * 10 + "".join(f"{name:>14}" for name in KINDS))
    for name, row in zip(KINDS, confusion, strict=True):
        print(f"  {name:>10}" + "".join(f"{v:>14}" for v in row))


if __name__ == "__main__":
    main()
