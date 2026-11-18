#!/usr/bin/env bash
# Train one model on one protocol and seed, with the settings of configs/base.yaml.
#
#   bash scripts/train.sh <protocol> <seed> [model]
#
#   protocol  session (SeaShips), smdplus, aboships, mcships, wsodd; random (SeaShips, for the leak study)
#   model     default: the base detector (method.base_detector), followed by Module A (WiSE-FT, weight
#             interpolation, no training of its own) and the ship-type classifier of Module B. Any other model
#             is trained alone, as a baseline: deim-r18, rtdetrv2-s, rtdetr-r18, yolo26n.pt, faster-rcnn-r50, ...
#
# Steps whose output exists are skipped, so the script can be re-run after an interruption. Train one model at a
# time (README.md, Notes). Environment: PYTHON (default python), CONFIG (default configs/base.yaml).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
CONFIG=${CONFIG:-configs/base.yaml}
export PYTHONIOENCODING=utf-8 YOLO_AUTOINSTALL=False
RUNS=runs/seaships TYPES=runs/types   # fixed in the code: the runs of every dataset go to runs/seaships/

cfg() {  # cfg <dotted key>: a value of $CONFIG; lists are printed space-separated
  "$PY" -c 'import functools, sys, yaml
v = functools.reduce(lambda d, k: d[k], sys.argv[2].split("."), yaml.safe_load(open(sys.argv[1], encoding="utf-8")))
print(" ".join(map(str, v)) if isinstance(v, list) else v)' "$CONFIG" "$1"
}

if [ $# -lt 2 ]; then awk 'NR > 1 && /^#/ { print; next } NR > 1 { exit }' "$0"; exit 2; fi
PROTOCOL=$1 SEED=$2
BASE=$(cfg method.base_detector)
MODEL=${3:-$BASE}
WORKERS=$(cfg workers)
run() { echo "${1%.pt}_${PROTOCOL}_s${SEED}"; }   # run name of a model: <model stem>_<protocol>_s<seed>

R=$(run "$MODEL")
if [ -f "$RUNS/$R/weights/best.pt" ]; then
  echo "[skip] $R: trained"
elif [ -e "$RUNS/$R" ]; then
  echo "$RUNS/$R exists without weights/best.pt (an interrupted run?): move it aside and re-run" >&2
  exit 1
else
  "$PY" src/models/train_detector.py --model "$MODEL" --protocol "$PROTOCOL" --seed "$SEED" --workers "$WORKERS"
fi
[ "$MODEL" = "$BASE" ] || exit 0

# Module A: WiSE-FT. Also scores alpha and alpha 1 on the protocol's val split (wise.json).
ALPHA=$(cfg method.wise_ft.alpha)
WISE=$("$PY" -c 'import sys; print("%s-wise%03d" % (sys.argv[1], round(100 * float(sys.argv[2]))))' "$BASE" "$ALPHA")
if [ -f "$RUNS/$(run "$WISE")/weights/best.pt" ]; then
  echo "[skip] $(run "$WISE"): Module A done"
else
  "$PY" src/models/wise_ft.py --run "$R" --alphas "$ALPHA" --val
fi

# Module B: the ship-type classifier, trained on the protocol's ground-truth crops (data/crops/, src/data/build_crops*.py)
ARM=$(cfg method.type_stage.arm)
A=${ARM}_${PROTOCOL}_s${SEED}
if [ -f "$TYPES/$A/metrics.json" ]; then
  echo "[skip] $A: classifier trained"
elif [ -e "$TYPES/$A" ]; then
  echo "$TYPES/$A exists without metrics.json (an interrupted run?): move it aside and re-run" >&2
  exit 1
else
  "$PY" src/models/type_classifier.py --arm "$ARM" --protocol "$PROTOCOL" --seed "$SEED" --workers "$WORKERS"
fi
