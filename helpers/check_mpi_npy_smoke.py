"""Verify MPI smoke shards merge to global marker (rung 3 formula)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from merge_npy_shards import (  # noqa: E402
    discover_shards,
    expected_global_marker,
    merge_equal_slabs,
)


def main() -> None:
    # usage: check_mpi_npy_smoke.py <dir> <npx> <npy> <npz> <nx> <ny> <nz> [field]
    if len(sys.argv) < 8:
        print(
            "usage: python helpers/check_mpi_npy_smoke.py "
            "<dir> <npx> <npy> <npz> <nx> <ny> <nz> [field]",
            file=sys.stderr,
        )
        sys.exit(2)
    d = Path(sys.argv[1])
    npx, npy_, npz = map(int, sys.argv[2:5])
    nx, ny, nz = map(int, sys.argv[5:8])
    field = sys.argv[8] if len(sys.argv) > 8 else "temperature"

    paths = discover_shards(d, field)
    g = merge_equal_slabs(paths, field=field)
    want = expected_global_marker((npx, npy_, npz), (nx, ny, nz), dtype=np.float32)
    if g.shape != want.shape:
        print(f"FAIL shape {g.shape} want {want.shape}", file=sys.stderr)
        sys.exit(1)
    if g.dtype != np.float32:
        print(f"FAIL dtype {g.dtype}", file=sys.stderr)
        sys.exit(1)
    if not np.array_equal(g, want):
        print("FAIL values != global marker", file=sys.stderr)
        sys.exit(1)
    print(
        f"OK mpi smoke merge field={field!r} shards={len(paths)} "
        f"shape={g.shape} dtype={g.dtype}"
    )
    print(f"sample [0,0,0]={g[0,0,0]} [-1,-1,-1]={g[-1,-1,-1]}")


if __name__ == "__main__":
    main()
