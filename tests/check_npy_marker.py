"""Verify Fortran stdlib save_npy marker file (serial round-trip rung 1)."""

import sys
from pathlib import Path

import numpy as np


def marker_value(i: int, j: int, k: int) -> float:
    """1-based i,j,k — must match test_npy_marker.f90."""
    return 100.0 * i + 10.0 * j + float(k)

def main(argv: list[str]) -> int:
    path = Path(argv[1] if len(argv) > 1 else "marker_3d.npy")
    if not path.is_file():
        print(f"FAIL missing file: {path}", file=sys.stderr)
        return 1

    raw = np.load(path)
    # stdlib writes fortran_order=True; np.load already presents indexable [i,j,k]
    # matching Fortran a(i,j,k). Do not silently transpose.
    print(f"path={path}")
    print(f"shape={raw.shape} dtype={raw.dtype} flags.f_contiguous={raw.flags.f_contiguous} "
          f"flags.c_contiguous={raw.flags.c_contiguous}")

    nx, ny, nz = 3, 4, 5
    if raw.shape != (nx, ny, nz):
        print(f"FAIL shape: want {(nx, ny, nz)} got {raw.shape}", file=sys.stderr)
        return 1

    expect = np.empty((nx, ny, nz), dtype=np.float64, order="F")
    for k in range(1, nz + 1):
        for j in range(1, ny + 1):
            for i in range(1, nx + 1):
                expect[i - 1, j - 1, k - 1] = marker_value(i, j, k)

    if raw.dtype != expect.dtype and not np.issubdtype(raw.dtype, np.floating):
        print(f"FAIL dtype: got {raw.dtype}", file=sys.stderr)
        return 1

    got = np.asarray(raw, dtype=np.float64)
    if not np.allclose(got, expect, rtol=0.0, atol=0.0):
        # exact match expected for this marker in float64
        bad = np.argwhere(got != expect)
        print(f"FAIL values: {len(bad)} mismatches", file=sys.stderr)
        for idx in bad[:5]:
            i, j, k = (int(idx[0]) + 1, int(idx[1]) + 1, int(idx[2]) + 1)
            print(
                f"  idx 1-based ({i},{j},{k}) got {got[tuple(idx)]} want {expect[tuple(idx)]}",
                file=sys.stderr,
            )
        return 1

    # Spot-check corners in 0-based numpy indexing
    assert got[0, 0, 0] == marker_value(1, 1, 1)
    assert got[nx - 1, ny - 1, nz - 1] == marker_value(nx, ny, nz)

    print("OK python load marker round-trip")
    print(f"sample [0,0,0]={got[0, 0, 0]}  [-1,-1,-1]={got[-1, -1, -1]}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
