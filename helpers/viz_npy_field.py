"""Visualize a merged (or shard-dir) field for smoke verification.

Examples:
  python helpers/viz_npy_field.py merged.npy
  python helpers/viz_npy_field.py _npy_mpi_smoke --field temperature --merge
  python helpers/viz_npy_field.py merged.npy --out smoke_viz.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_array(path: Path, *, field: str | None, do_merge: bool) -> tuple[np.ndarray, str]:
    if do_merge or path.is_dir():
        from merge_npy_shards import discover_shards, merge_equal_slabs

        if not field:
            raise SystemExit("--field is required when merging a directory")
        paths = discover_shards(path, field)
        g = merge_equal_slabs(paths, field=field)
        return g, f"merge:{path.name}:{field}"
    if not path.is_file():
        raise SystemExit(f"not a file: {path}")
    g = np.load(path)
    if g.ndim != 3:
        raise SystemExit(f"expected 3D array, got shape {g.shape}")
    return g, path.name


def main() -> None:
    p = argparse.ArgumentParser(description="Slice view of a 3D .npy field")
    p.add_argument("path", type=Path, help="merged .npy file, or shard directory with --merge")
    p.add_argument("--field", default=None, help="field name when merging a directory")
    p.add_argument("--merge", action="store_true", help="treat path as shard directory")
    p.add_argument("--out", type=Path, default=None, help="save figure (default: show)")
    p.add_argument("--k", type=int, default=None, help="k index for mid-k override")
    p.add_argument("--save-merged", type=Path, default=None,
                   help="if merging, also write one global .npy here")
    args = p.parse_args()

    g, label = _load_array(args.path, field=args.field, do_merge=args.merge or args.path.is_dir())
    if args.save_merged is not None:
        # preserve dtype; np.save adds .npy if needed
        outp = args.save_merged
        np.save(outp, np.asfortranarray(g))
        print(f"wrote merged {outp} shape={g.shape} dtype={g.dtype}")

    import matplotlib.pyplot as plt

    nx, ny, nz = g.shape
    k = args.k if args.k is not None else nz // 2
    j = ny // 2
    i = nx // 2
    if not (0 <= k < nz and 0 <= j < ny and 0 <= i < nx):
        raise SystemExit(f"slice out of range for shape {g.shape}")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), constrained_layout=True)
    fig.suptitle(
        f"{label}  shape={g.shape}  dtype={g.dtype}\n"
        f"[{0,0,0}]={g[0,0,0]}  [-1,-1,-1]={g[-1,-1,-1]}",
        fontsize=10,
    )

    panels = [
        (axes[0], g[:, :, k].T, f"i–j at k={k}", "i", "j"),
        (axes[1], g[:, j, :].T, f"i–k at j={j}", "i", "k"),
        (axes[2], g[i, :, :].T, f"j–k at i={i}", "j", "k"),
    ]
    vmin, vmax = float(np.min(g)), float(np.max(g))
    im = None
    for ax, img, title, xl, yl in panels:
        im = ax.imshow(
            img,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="value")
    if args.out is not None:
        fig.savefig(args.out, dpi=140)
        print(f"wrote {args.out}")
        plt.close(fig)
    else:
        # non-interactive fallback: write default next to cwd
        fallback = Path("npy_field_viz.png")
        fig.savefig(fallback, dpi=140)
        plt.close(fig)
        print(f"wrote {fallback} (no display); pass --out to choose path")


if __name__ == "__main__":
    main()
