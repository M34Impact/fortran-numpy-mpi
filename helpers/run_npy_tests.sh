#!/usr/bin/env bash
# Serial exercise of the npy dump ladder (rungs 1–4 + MPI smoke).
# Run from repo root. Fail-loud: first failure exits non-zero.
#
# Usage:
#   ./helpers/run_npy_tests.sh
#   ./helpers/run_npy_tests.sh --skip-mpi
#   ./helpers/run_npy_tests.sh --skip-fpm
#   NP=4 FPM_FLAG="-L./lib" ./helpers/run_npy_tests.sh
#
# Env:
#   NP          MPI ranks for cart smoke (default 4 = 2x2x1)
#   FPM         fpm binary (default: fpm)
#   FPM_FLAG    extra --flag to fpm (default: -L./lib if ./lib exists)
#   MPIRUN      launcher (default: mpirun)
#   MPIRUN_ARGS extra runner-args (default: bind/map if using mpirun)
#   PYTHON      python (default: python3 then python)
#   KEEP_SMOKE  if 1, leave _npy_mpi_smoke/ and merged artifacts

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SKIP_MPI=0
SKIP_FPM=0
for arg in "$@"; do
  case "$arg" in
    --skip-mpi) SKIP_MPI=1 ;;
    --skip-fpm) SKIP_FPM=1 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

NP="${NP:-4}"
FPM="${FPM:-fpm}"
MPIRUN="${MPIRUN:-mpirun}"
KEEP_SMOKE="${KEEP_SMOKE:-0}"
SMOKE_DIR="${SMOKE_DIR:-_npy_mpi_smoke}"
FIELD="${FIELD:-temperature}"

# process grid / local sizes must match test/test_npy_mpi_dump.f90
NPX=2; NPY=2; NPZ=1
NX=3; NY=4; NZ=5

if [[ -z "${FPM_FLAG+x}" ]]; then
  if [[ -d "$ROOT/lib" ]]; then
    FPM_FLAG="-L./lib"
  else
    FPM_FLAG=""
  fi
fi

if [[ -z "${MPIRUN_ARGS+x}" ]]; then
  MPIRUN_ARGS="--bind-to core --map-by ppr:1:core -np ${NP}"
fi

if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

step=0
section() {
  step=$((step + 1))
  echo
  echo "=== [$step] $* ==="
}

run() {
  echo "+ $*"
  "$@"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "FAIL: missing command: $1" >&2
    exit 1
  }
}

cleanup() {
  if [[ "$KEEP_SMOKE" == "1" ]]; then
    return 0
  fi
  rm -rf "$SMOKE_DIR"
  rm -f \
    "${SMOKE_DIR%/}_temperature_global.npy" \
    temperature_global_smoke.npy \
    npy_mpi_smoke_viz.png \
    marker_3d.npy marker_3d_sp.npy \
    temperature__i1_j2_k0.npy
}

need_cmd "$PY"

echo "ROOT=$ROOT"
echo "PY=$PY  FPM=$FPM  SKIP_FPM=$SKIP_FPM  SKIP_MPI=$SKIP_MPI  NP=$NP"

# ---------------------------------------------------------------------------
# Python unit tests (no compiler)
# ---------------------------------------------------------------------------
section "Python npy_names"
run "$PY" helpers/test_npy_names.py

section "Python equal-slab merge"
run "$PY" helpers/test_merge_npy_shards.py

section "Python merge CLI smoke (fake shards)"
FAKE="$(mktemp -d "${TMPDIR:-/tmp}/npy_fake.XXXXXX")"
run "$PY" - <<PY
from pathlib import Path
import sys
sys.path.insert(0, "helpers")
from merge_npy_shards import write_fake_shards
write_fake_shards(Path("$FAKE"), field="$FIELD", nproc=($NPX, $NPY, $NPZ),
                  local_shape=($NX, $NY, $NZ))
print("wrote fake shards -> $FAKE")
PY
run "$PY" helpers/merge_npy_shards.py "$FIELD" "$FAKE"
run "$PY" helpers/check_mpi_npy_smoke.py "$FAKE" "$NPX" "$NPY" "$NPZ" "$NX" "$NY" "$NZ" "$FIELD"
MERGED_FAKE="$FAKE/${FIELD}_global.npy"
run "$PY" helpers/merge_npy_shards.py "$FIELD" "$FAKE" "$MERGED_FAKE"
run "$PY" helpers/viz_npy_field.py "$MERGED_FAKE" --out "$FAKE/viz.png"
echo "OK fake merge+viz ($FAKE/viz.png)"
rm -rf "$FAKE"

section "Python npy_names CLI"
run "$PY" helpers/npy_names.py --format "$FIELD" 1 2 0
run "$PY" helpers/npy_names.py "${FIELD}__i1_j2_k0.npy"

# ---------------------------------------------------------------------------
# Fortran via fpm (optional)
# ---------------------------------------------------------------------------
if [[ "$SKIP_FPM" == "1" ]]; then
  section "skip fpm tests (--skip-fpm)"
else
  need_cmd "$FPM"

  FPM_COMMON=(test --profile debug)
  if [[ -n "$FPM_FLAG" ]]; then
    FPM_COMMON+=(--flag "$FPM_FLAG")
  fi

  # Serial fpm tests: prefer named targets when present; else full `fpm test`.
  SERIAL_TARGETS=()
  for t in test_npy_names test_npy_dump_field test_npy_marker; do
    if [[ -f "test/${t}.f90" ]]; then
      SERIAL_TARGETS+=("$t")
    fi
  done

  if ((${#SERIAL_TARGETS[@]} > 0)); then
    for t in "${SERIAL_TARGETS[@]}"; do
      section "fpm test --target $t (serial)"
      # np 1 runner keeps MPI-linked binaries honest if the package always links MPI
      if command -v "$MPIRUN" >/dev/null 2>&1; then
        run "$FPM" "${FPM_COMMON[@]}" --target "$t" \
          --runner "$MPIRUN" --runner-args " -np 1"
      else
        run "$FPM" "${FPM_COMMON[@]}" --target "$t"
      fi
    done
  else
    section "fpm test (all serial targets — no named npy tests found)"
    if command -v "$MPIRUN" >/dev/null 2>&1; then
      run "$FPM" "${FPM_COMMON[@]}" --runner "$MPIRUN" --runner-args " -np 1"
    else
      run "$FPM" "${FPM_COMMON[@]}"
    fi
  fi

  # Optional marker python check if files appeared
  if [[ -f marker_3d.npy ]] || [[ -f marker_3d_sp.npy ]]; then
    section "Python marker check (if helper present)"
    if [[ -f helpers/check_npy_marker.py ]]; then
      [[ -f marker_3d.npy ]] && run "$PY" helpers/check_npy_marker.py marker_3d.npy
      [[ -f marker_3d_sp.npy ]] && run "$PY" helpers/check_npy_marker.py marker_3d_sp.npy
    else
      run "$PY" - <<'PY'
import numpy as np
from pathlib import Path
for p in ["marker_3d.npy", "marker_3d_sp.npy"]:
    path = Path(p)
    if not path.exists():
        continue
    a = np.load(path)
    assert a.shape == (3, 4, 5), a.shape
    assert a[0, 0, 0] == 111
    assert a[-1, -1, -1] == 345
    print(f"OK marker {p} shape={a.shape} dtype={a.dtype}")
PY
    fi
  fi

  if [[ -f temperature__i1_j2_k0.npy ]]; then
    section "Python parse dump-field artifact"
    run "$PY" helpers/npy_names.py temperature__i1_j2_k0.npy
  fi
fi

# ---------------------------------------------------------------------------
# MPI dump smoke + merge + viz
# ---------------------------------------------------------------------------
if [[ "$SKIP_MPI" == "1" ]]; then
  section "skip MPI smoke (--skip-mpi)"
elif [[ "$SKIP_FPM" == "1" ]]; then
  section "skip MPI smoke (needs fpm)"
else
  need_cmd "$MPIRUN"
  if [[ ! -f test/test_npy_mpi_dump.f90 ]]; then
    echo "FAIL: test/test_npy_mpi_dump.f90 missing" >&2
    exit 1
  fi
  if [[ "$NP" -ne $((NPX * NPY * NPZ)) ]]; then
    echo "FAIL: NP=$NP but test cart is ${NPX}x${NPY}x${NPZ} (need $((NPX*NPY*NPZ)))" >&2
    exit 1
  fi

  section "cleanup old smoke dir"
  rm -rf "$SMOKE_DIR"

  section "fpm test --target test_npy_mpi_dump (np=$NP)"
  run "$FPM" test --profile release --target test_npy_mpi_dump \
    ${FPM_FLAG:+--flag "$FPM_FLAG"} \
    --runner "$MPIRUN" --runner-args " ${MPIRUN_ARGS}"

  section "merge MPI shards + check + viz"
  MERGED="${SMOKE_DIR}/${FIELD}_global.npy"
  run "$PY" helpers/merge_npy_shards.py "$FIELD" "$SMOKE_DIR" "$MERGED"
  run "$PY" helpers/check_mpi_npy_smoke.py \
    "$SMOKE_DIR" "$NPX" "$NPY" "$NPZ" "$NX" "$NY" "$NZ" "$FIELD"
  run "$PY" helpers/viz_npy_field.py "$MERGED" --out npy_mpi_smoke_viz.png
  echo "OK MPI smoke artifacts: $SMOKE_DIR/  $MERGED  npy_mpi_smoke_viz.png"
fi

if [[ "$KEEP_SMOKE" != "1" ]]; then
  section "cleanup (KEEP_SMOKE=0)"
  cleanup
else
  section "keep smoke artifacts (KEEP_SMOKE=1)"
fi

echo
echo "=== ALL NPY CHECKS PASSED ==="
