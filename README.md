# fortran-numpy-mpi

Dump Euclidean 3D field data from MPI Fortran to NumPy `.npy` shards, then merge them in Python for analysis.

## Problem

Distributed CFD/structural solvers hold fields as per-rank arrays with halos. Writing a single global file from rank 0 requires gathers, extra buffers, and often a custom binary format. Post-processing already lives in Python/NumPy. The missing piece is a small, explicit path:

1. Each rank writes **owned** cells only to a shard file whose name encodes its place in the process grid.
2. A Python helper loads those shards and assembles one global `ndarray`.

## Scope

- 3D Cartesian MPI topology with 0-based coords in filenames
- SP / DP / default `integer` rank-3 dumps via *single* generic interface
- Owned-only slices (caller strips halos)
- Equal-slab merge and name parse/format in Python
- Activation by editing the solver call site
- Axis order matches Fortran and NumPy without transpose: `a(i,j,k)` ↔ `a[i,j,k]`.

### Out of scope

- Unequal local sizes (v1 assumes equal slabs)
- Automatic timestep directory policy
- Silent rename of illegal field names
- True 4D multi-field dumps
- Namelist / JSON switches

## Filename contract

```text
{field}__i{i}_j{j}_k{k}.npy
```

- `field` — non-empty basename stem. Allowed characters: `[A-Za-z0-9_+=.-]`. No whitespace, path separators, `.npy` suffix, or substring `__i`. Invalid names fail with non-zero `stat` / `ValueError` (no underscore substitution).
- `i`, `j`, `k` — non-negative **0-based** topology indices supplied by the caller.

Examples: `temperature__i0_j1_k0.npy`, `cell_state__i1_j0_k0.npy`.

#### Legacy 1-based coords

simply convert at the call site:

```fortran
rank_coords = [ip - 1, jp - 1, kp - 1]
```

#### Modern Cartesian

MPI already supplies a method to construct the coordinates:

```fortran
call MPI_Cart_coords(comm, rank, 3, rank_coords, ierr)
```

## Fortran API

Package layout:

- `npy_dump_names` — pure name builder, no MPI, no I/O.
- `npy_dump_field` — generic `mpi_dump_field` → `save_npy` (stdlib).

```fortran
call mpi_dump_field(field_name, rank_coords, array, stat, msg, directory=outdir)
```

| Argument | Meaning |
|----------|---------|
| `field_name` | Stem used in the basename |
| `rank_coords(3)` | 0-based `(i,j,k)` in the process grid |
| `array` | Rank-3 owned block (`real32`, `real64`, or `integer`) |
| `stat` / `msg` | `stat /= 0` on failure; message when available |
| `directory` | Optional output directory (default: current working directory) |

Caller responsibilities:

- Pass owned cells only - no halos, e.g. `field(1:nx,1:ny,1:nz)`.
- Create the output directory if needed (and barrier before ranks write).
- Check `stat` after every dump.
- Choose filesystem-safe names (`cell_state`, not `"cell state"`).

Depends on [fortran-lang/stdlib](https://github.com/fortran-lang/stdlib) `stdlib_io_npy` and MPI (`mpi_f08` in tests).

## Python package `numpy_mpi`

Install the project env as you prefer (`uv sync`, `uv pip install -e .`, etc.)

| Module / script | Role |
|-----------------|------|
| `npy_names.py` | Format / parse / round-trip shard basenames |
| `merge_npy_shards.py` | Discover shards for a field, merge equal slabs, optional `np.save` |
| `check_mpi_npy_smoke.py` | Compare merged MPI smoke output to the analytic marker |
| `viz_npy_field.py` | Quick 3-plane slices of a merged `.npy` |
| `run_npy_tests.sh` | Serial ladder: unit checks, `fpm test`, MPI smoke, merge, check |

Merge assumptions (v1):

- One field per merge invocation.
- All shards same local shape and dtype.
- Full rectangular process grid (no holes, no duplicate coords).
- Placement: global index `I = i_rank * nx + i_local` (and likewise for `j`, `k`), Fortran order.

```sh
python -m numpy_mpi.merge_npy_shards temperature _npy_mpi_smoke temperature_global.npy
python -m numpy_mpi.check_mpi_npy_smoke _npy_mpi_smoke 2 2 1 3 4 5 temperature
python -m numpy_mpi.viz_npy_field temperature_global.npy
```

## Marker convention (tests)

Global 0-based indices `(I,J,K)`:

```text
v = 100*I + 10*J + K
```

(Use a larger leading coefficient if your global extent needs unique values beyond hundreds.) Serial and MPI tests must use the **same** formula as the Python checker.

## Solver integration

Not wired to any input deck. In the time loop (or a debug branch):

```fortran
call mpi_dump_field("temperature", rank_coords, T(1:nx,1:ny,1:nz), stat, msg, directory=dir)
if (stat /= 0) then
   ! log msg, abort or skip
end if
```

Change variables by changing the array and the stem string. Multiple variables = multiple calls, each with its own stem. Optional `directory` can point at a timestep folder when you introduce one; the library does not invent that tree.

## Build and test

```sh
fpm build --profile release
fpm test --profile release
# MPI smoke (example; match your launcher and process count to the test grid)
fpm test --target test_npy_mpi_dump --profile release \
  --runner mpirun --runner-args "-np 4"
./scripts/run_npy_tests.sh
```

Default MPI smoke grid in-tree: `2×2×1` processes, local `3×4×5` → global `6×8×5`.

## Design notes

- **Rank-only filenames are insufficient** without a separate rank→coords map. Coords in the name keep merge self-describing.
- **Offsets** assume a regular Euclidean grid with unit spacing for indexing; geometry is not stored in the `.npy`.
- **4D multi-variable dumps** need request queuing and a packing convention; deferred. Prefer one 3D array per file.

## License

[MIT](./LICENSE)

## Funding

Developed as part of the [M³4Impact](https://www.gre.ac.uk/research/m34impact) project at the University of Greenwich.

## References

- [`fpm`](https://github.com/fortran-lang/fpm)
- [Modern Fortran extension](https://github.com/fortran-lang/vscode-fortran-support)
- [`fortls`](https://github.com/fortran-lang/fortls)
- [`pre-commit`](https://pre-commit.com/)
- [`uv`](https://github.com/astral-sh/uv)
