"""Filename contract for per-rank field dumps.

Format (normative):
    {field}__i{i}_j{j}_k{k}.npy

- field: non-empty stem; [A-Za-z0-9_+=.-] only (no spaces/shell-unsafe).
  Must not contain the substring "__i" (keeps parse unambiguous).
- i, j, k: non-negative 0-based topology indices (already normalized by the caller).
- No path components in field; basename only.

Topology is assumed given. This module does not talk to MPI.
"""
import re
import sys
from dataclasses import dataclass

# Full match on basename (no directories).
# Field: no whitespace/path; exclude "__i" via negative lookahead inside class is hard —
# validate field after match. Char class keeps shells/glob happy.
_FIELD_RE = re.compile(r"^[A-Za-z0-9_+=.-]+$")
_NAME_RE = re.compile(
    r"^(?P<field>[A-Za-z0-9_+=.-]+)__i(?P<i>\d+)_j(?P<j>\d+)_k(?P<k>\d+)\.npy$"
)


@dataclass(frozen=True, slots=True)
class NpyShardName:
    field: str
    i: int
    j: int
    k: int

    def filename(self) -> str:
        return format_npy_shard_name(self.field, self.i, self.j, self.k)


def format_npy_shard_name(field: str, i: int, j: int, k: int) -> str:
    """Build basename from field + 0-based topology coords. Fail loud on bad input."""
    if not isinstance(field, str) or field == "":
        raise ValueError("field must be a non-empty str")
    if "/" in field or "\\" in field:
        raise ValueError(f"field must be a basename, not a path: {field!r}")
    if field.endswith(".npy"):
        raise ValueError(f"field must not include .npy suffix: {field!r}")
    if any(ch.isspace() for ch in field):
        raise ValueError(
            f"field must not contain whitespace (use cell_state not 'cell state'): {field!r}"
        )
    if not _FIELD_RE.fullmatch(field):
        raise ValueError(
            f"field must match [A-Za-z0-9_+=.-]+ (no spaces/shell-unsafe): {field!r}"
        )
    if "__i" in field:
        raise ValueError(
            f"field must not contain '__i' (ambiguous with coord suffix): {field!r}"
        )
    for label, val in (("i", i), ("j", j), ("k", k)):
        if not isinstance(val, int) or isinstance(val, bool):
            raise TypeError(f"{label} must be int, got {type(val).__name__}")
        if val < 0:
            raise ValueError(f"{label} must be >= 0 (0-based topology), got {val}")
    return f"{field}__i{i}_j{j}_k{k}.npy"


def parse_npy_shard_name(name: str) -> NpyShardName:
    """Parse basename (or path ending in basename). Fail loud if it does not match."""
    if not isinstance(name, str) or name == "":
        raise ValueError("name must be a non-empty str")
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    m = _NAME_RE.match(base)
    if m is None:
        raise ValueError(
            f"not a shard npy name (want field__i{{i}}_j{{j}}_k{{k}}.npy): {base!r}"
        )
    field = m.group("field")
    if any(ch.isspace() for ch in field) or not _FIELD_RE.fullmatch(field):
        raise ValueError(f"invalid field segment in shard name: {base!r}")
    if "__i" in field:
        raise ValueError(f"ambiguous field segment containing '__i': {base!r}")
    return NpyShardName(
        field=field,
        i=int(m.group("i")),
        j=int(m.group("j")),
        k=int(m.group("k")),
    )


def roundtrip_name(field: str, i: int, j: int, k: int) -> NpyShardName:
    """format → parse; used by tests and as a sanity helper."""
    return parse_npy_shard_name(format_npy_shard_name(field, i, j, k))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(
            "usage: python helpers/npy_names.py <name.npy> [more...]",
            file=sys.stderr,
        )
        print(
            "       python helpers/npy_names.py --format <field> <i> <j> <k>",
            file=sys.stderr,
        )
        sys.exit(2)

    # Optional: build a name from pieces (handy when checking Fortran output).
    if args[0] == "--format":
        if len(args) != 5:
            print(
                "usage: python helpers/npy_names.py --format <field> <i> <j> <k>",
                file=sys.stderr,
            )
            sys.exit(2)
        field, si, sj, sk = args[1], args[2], args[3], args[4]
        try:
            name = format_npy_shard_name(field, int(si), int(sj), int(sk))
        except (TypeError, ValueError) as e:
            print(f"FAIL format: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"name={name}")
        parsed = parse_npy_shard_name(name)
        print(
            f"parse field={parsed.field!r} i={parsed.i} j={parsed.j} k={parsed.k}"
        )
        sys.exit(0)

    nerr = 0
    for raw in args:
        try:
            parsed = parse_npy_shard_name(raw)
        except ValueError as e:
            print(f"FAIL {raw!r}: {e}", file=sys.stderr)
            nerr += 1
            continue
        base = raw.replace("\\", "/").rsplit("/", 1)[-1]
        print(f"input={raw}")
        print(f"basename={base}")
        print(
            f"field={parsed.field!r} i={parsed.i} j={parsed.j} k={parsed.k}"
        )
        print(f"rebuild={parsed.filename()}")
    sys.exit(1 if nerr else 0)
