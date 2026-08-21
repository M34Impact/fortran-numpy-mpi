"""Tests for equal-slab shard merge (rung 3)."""

import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from numpy_mpi.merge_npy_shards import (
    discover_shards,
    expected_global_marker,
    merge_equal_slabs,
    write_fake_shards,
)
from numpy_mpi.npy_names import format_npy_shard_name


def expect_ok(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def expect_raises(fn, *args, exc=ValueError, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def main() -> None:
    field = "temperature"
    nproc = (2, 2, 1)
    local = (3, 4, 5)
    dtype = np.float32

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        paths = write_fake_shards(
            td_path, field=field, nproc=nproc, local_shape=local, dtype=dtype
        )
        expect_ok(len(paths) == 2 * 2 * 1, f"shard count {len(paths)}")

        found = discover_shards(td_path, field)
        expect_ok(len(found) == len(paths), "discover count")

        g = merge_equal_slabs(found, field=field)
        expect_ok(g.dtype == dtype, f"dtype {g.dtype}")
        expect_ok(g.shape == (6, 8, 5), f"shape {g.shape}")
        expect_ok(bool(g.flags.f_contiguous), "f_contiguous")

        want = expected_global_marker(nproc, local, dtype=dtype)
        expect_ok(np.array_equal(g, want), "marker mismatch")
        expect_ok(g[0, 0, 0] == 0.0, "corner")
        # last global index in i: 5, j: 7, k: 4 -> 100*5 + 10*7 + 4 = 574
        expect_ok(g[-1, -1, -1] == np.dtype(dtype).type(574), f"end {g[-1, -1, -1]}")

        # --- fail loud: missing shard ---
        missing_paths = [
            p for p in found if p.name != format_npy_shard_name(field, 1, 1, 0)
        ]
        expect_raises(merge_equal_slabs, missing_paths, field=field)

        # --- fail loud: duplicate coord ---
        dup = found + [found[0]]
        expect_raises(merge_equal_slabs, dup, field=field)

        # --- fail loud: shape mismatch ---
        bad = td_path / format_npy_shard_name(field, 0, 0, 0)
        np.save(bad, np.zeros((2, 4, 5), dtype=dtype, order="F"))
        expect_raises(merge_equal_slabs, discover_shards(td_path, field), field=field)

    # fresh dir for dtype mix / 2x2x2 smoke
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        write_fake_shards(
            td_path,
            field="c",
            nproc=(2, 2, 2),
            local_shape=(2, 2, 2),
            dtype=np.float32,
        )
        g2 = merge_equal_slabs(discover_shards(td_path, "c"), field="c")
        expect_ok(g2.shape == (4, 4, 4), f"2x2x2 shape {g2.shape}")
        expect_ok(
            np.array_equal(g2, expected_global_marker((2, 2, 2), (2, 2, 2))), "2x2x2"
        )

    print("OK merge equal slabs (rung 3)")


if __name__ == "__main__":
    main()
