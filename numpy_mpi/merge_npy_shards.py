"""Merge equal-slab per-rank .npy shards into one global array.

Assumes:
- filenames: {field}__i{i}_j{j}_k{k}.npy (see npy_names)
- equal local shapes on every shard
- complete rectangular process grid (no holes, no duplicates)
- axis order matches Fortran/NumPy a[i,j,k], no transpose
- dtype preserved from shards (typically float32)
"""
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from numpy_mpi.npy_names import (
    NpyShardName,
    format_npy_shard_name,
    parse_npy_shard_name,
)


def _as_path_list(paths: Iterable[str | Path]) -> list[Path]:
    out = [Path(p) for p in paths]
    if not out:
        raise ValueError("no shard paths given")
    return out


def discover_shards(directory: str | Path, field: str) -> list[Path]:
    """List shards for `field` in directory (non-recursive)."""
    d = Path(directory)
    if not d.is_dir():
        raise FileNotFoundError(f"not a directory: {d}")
    if not field:
        raise ValueError("field must be non-empty")
    prefix = f"{field}__i"
    found: list[Path] = []
    for p in sorted(d.iterdir()):
        if not p.is_file() or p.suffix != ".npy":
            continue
        if not p.name.startswith(prefix):
            continue
        # Validate full contract (rejects wrong patterns).
        parsed = parse_npy_shard_name(p.name)
        if parsed.field != field:
            continue
        found.append(p)
    if not found:
        raise FileNotFoundError(f"no shards for field={field!r} in {d}")
    return found


def merge_equal_slabs(
    paths: Iterable[str | Path],
    *,
    field: str | None = None,
) -> np.ndarray:
    """Load shards and assemble global array (F-order placement).

    If `field` is set, every shard must parse to that field.
    Process grid must be the full box [0..ni) x [0..nj) x [0..nk).
    """
    path_list = _as_path_list(paths)

    records: list[tuple[NpyShardName, Path, np.ndarray]] = []
    for p in path_list:
        meta = parse_npy_shard_name(p.name)
        if field is not None and meta.field != field:
            raise ValueError(
                f"field mismatch: want {field!r}, got {meta.field!r} from {p.name}"
            )
        arr = np.load(p)
        if arr.ndim != 3:
            raise ValueError(f"expected 3D array in {p.name}, got ndim={arr.ndim}")
        records.append((meta, p, arr))

    fields = {m.field for m, _, _ in records}
    if len(fields) != 1:
        raise ValueError(f"mixed fields in shard set: {sorted(fields)}")
    field_name = fields.pop()

    shapes = {a.shape for _, _, a in records}
    if len(shapes) != 1:
        raise ValueError(f"mixed local shapes: {sorted(shapes)}")
    dtypes = {a.dtype for _, _, a in records}
    if len(dtypes) != 1:
        raise ValueError(f"mixed dtypes: {sorted(dtypes, key=str)}")

    nx, ny, nz = shapes.pop()
    dtype = dtypes.pop()

    coords = [(m.i, m.j, m.k) for m, _, _ in records]
    if len(coords) != len(set(coords)):
        dup = [c for c in coords if coords.count(c) > 1]
        raise ValueError(f"duplicate topology coords: {sorted(set(dup))}")

    ni = max(c[0] for c in coords) + 1
    nj = max(c[1] for c in coords) + 1
    nk = max(c[2] for c in coords) + 1
    expected = {(i, j, k) for i in range(ni) for j in range(nj) for k in range(nk)}
    got = set(coords)
    missing = sorted(expected - got)
    extra = sorted(got - expected)  # should be empty if max-box logic holds
    if missing:
        raise ValueError(
            f"incomplete process grid for field={field_name!r}: "
            f"grid {ni}x{nj}x{nk}, missing {missing}"
        )
    if extra:
        raise ValueError(f"coords outside inferred grid: {extra}")

    global_shape = (ni * nx, nj * ny, nk * nz)
    g = np.empty(global_shape, dtype=dtype, order="F")

    for meta, p, arr in records:
        i0, j0, k0 = meta.i * nx, meta.j * ny, meta.k * nz
        g[i0 : i0 + nx, j0 : j0 + ny, k0 : k0 + nz] = arr

    return g


def write_fake_shards(
    directory: str | Path,
    *,
    field: str,
    nproc: tuple[int, int, int],
    local_shape: tuple[int, int, int],
    dtype=np.float32,
) -> list[Path]:
    """Write analytic marker shards for a full equal-slab grid. Returns paths.

    Marker at global 0-based (I,J,K):
        v = 1000*I + 10*J + K
    (fits exactly in float32 for small test grids).
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    npx, npy_, npz = nproc
    nx, ny, nz = local_shape
    if min(npx, npy_, npz) < 1:
        raise ValueError(f"nproc must be >= 1 each, got {nproc}")
    if min(nx, ny, nz) < 1:
        raise ValueError(f"local_shape must be >= 1 each, got {local_shape}")

    paths: list[Path] = []
    for ip in range(npx):
        for jp in range(npy_):
            for kp in range(npz):
                a = np.empty((nx, ny, nz), dtype=dtype, order="F")
                for i in range(nx):
                    for j in range(ny):
                        for k in range(nz):
                            I = ip * nx + i
                            J = jp * ny + j
                            K = kp * nz + k
                            a[i, j, k] = np.dtype(dtype).type(100 * I + 10 * J + K)
                name = format_npy_shard_name(field, ip, jp, kp)
                path = d / name
                np.save(path, a)
                # np.save may add .npy if missing; we already have .npy
                paths.append(path)
    return paths


def expected_global_marker(
    nproc: tuple[int, int, int],
    local_shape: tuple[int, int, int],
    dtype=np.float32,
) -> np.ndarray:
    """Analytic global array matching write_fake_shards."""
    npx, npy_, npz = nproc
    nx, ny, nz = local_shape
    g = np.empty((npx * nx, npy_ * ny, npz * nz), dtype=dtype, order="F")
    for I in range(npx * nx):
        for J in range(npy_ * ny):
            for K in range(npz * nz):
                g[I, J, K] = np.dtype(dtype).type(100 * I + 10 * J + K)
    return g


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print(
            "usage: python -m numpy_mpi.merge_npy_shards <field> <dir> [out.npy]",
            file=sys.stderr,
        )
        sys.exit(2)
    field_arg, dir_arg = args[0], args[1]
    out_arg = args[2] if len(args) > 2 else None
    try:
        paths = discover_shards(dir_arg, field_arg)
        g = merge_equal_slabs(paths, field=field_arg)
    except (OSError, ValueError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
    print(
        f"OK merge field={field_arg!r} shards={len(paths)} "
        f"shape={g.shape} dtype={g.dtype} f_contiguous={g.flags.f_contiguous}"
    )
    print(f"sample [0,0,0]={g[0, 0, 0]} [-1,-1,-1]={g[-1, -1, -1]}")
    if out_arg:
        # Save without forcing C-order; use Fortran order via ndarray
        np.save(out_arg, g)
        print(f"wrote {out_arg}")
