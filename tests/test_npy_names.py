"""Tests for helpers.npy_names — encode the filename contract."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python helpers/test_npy_names.py` from repo root without install.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from numpy_mpi.npy_names import (
    NpyShardName,
    format_npy_shard_name,
    parse_npy_shard_name,
    roundtrip_name,
)


def expect_ok(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def expect_raises(fn, *args, exc=ValueError) -> None:
    try:
        fn(*args)
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} from {fn.__name__}{args!r}")


def main() -> None:
    # --- happy path / round-trip ---
    cases = [
        ("temperature", 0, 0, 0),
        ("temperature", 1, 2, 0),
        ("c", 0, 0, 9),
        ("phi_gas", 10, 0, 3),
        ("T", 99, 99, 99),
    ]
    for field, i, j, k in cases:
        name = format_npy_shard_name(field, i, j, k)
        expect_ok(
            name == f"{field}__i{i}_j{j}_k{k}.npy",
            f"format mismatch: {name!r}",
        )
        parsed = parse_npy_shard_name(name)
        expect_ok(parsed == NpyShardName(field, i, j, k), f"parse mismatch: {parsed}")
        expect_ok(roundtrip_name(field, i, j, k) == parsed, "roundtrip")
        # path prefix stripped
        expect_ok(
            parse_npy_shard_name(f"out/step_3/{name}") == parsed,
            "path prefix",
        )

    # --- reject bad names (fail loud) ---
    bad_names = [
        "",
        "temperature.npy",
        "temperature__i1_j2.npy",
        "temperature__i1_j2_k0.np",
        "temperature_i1_j2_k0.npy",
        "__i0_j0_k0.npy",
        "temperature__i-1_j0_k0.npy",
        "temperature__i0_j0_k0_extra.npy",
        "temperature__i0_j0_k0.NPY",
    ]
    for bad in bad_names:
        expect_raises(parse_npy_shard_name, bad)

    bad_format = [
        (lambda: format_npy_shard_name("", 0, 0, 0), ValueError),
        (lambda: format_npy_shard_name("a/b", 0, 0, 0), ValueError),
        (lambda: format_npy_shard_name("x.npy", 0, 0, 0), ValueError),
        (lambda: format_npy_shard_name("bad__i", 0, 0, 0), ValueError),
        (lambda: format_npy_shard_name("t", -1, 0, 0), ValueError),
        (lambda: format_npy_shard_name("t", 0, -2, 0), ValueError),
        (lambda: format_npy_shard_name("t", 0, 0, -3), ValueError),
        (lambda: format_npy_shard_name("t", 1.5, 0, 0), TypeError),  # type: ignore[arg-type]
    ]
    for fn, exc in bad_format:
        expect_raises(fn, exc=exc)

    print("OK npy_names format/parse round-trip and rejects")


if __name__ == "__main__":
    main()
