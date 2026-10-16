#!/usr/bin/env bash
set -euo pipefail
TWO_PLAYER_PYTHON="${TWO_PLAYER_PYTHON:-/home/jdingav/.conda/envs/dreamer/bin/python}"
TWO_PLAYER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TWO_PLAYER_CONTROL="$TWO_PLAYER_ROOT/logdir/two_player_sweep_20261015"
mkdir -p "$TWO_PLAYER_CONTROL/tmp" "$TWO_PLAYER_CONTROL/cache"
export TMPDIR="$TWO_PLAYER_CONTROL/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR" XDG_CACHE_HOME="$TWO_PLAYER_CONTROL/cache"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd -- "$TWO_PLAYER_CONTROL"
"$TWO_PLAYER_PYTHON" -B -u "$TWO_PLAYER_ROOT/tools/verify_two_player.py" "$@" 2>&1 | tee -a "$TWO_PLAYER_CONTROL/verification.log"
