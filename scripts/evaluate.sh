#!/usr/bin/env bash
# Test the runs of one protocol and seed, or compare models over their seeds, with the settings of configs/base.yaml.
#
#   bash scripts/evaluate.sh <protocol> <seed> [model]
#       model default: the base detector, tested step by step as the method: the detector, + Module A (WiSE-FT),
#       + Module B (ship-type stage at weight w; also without Module A, for the ablation), and with sliced inference
#       where method.slicing.protocols lists the protocol. Any other model is tested alone. SeaShips runs are also
#       tested zero-shot on SMD and ABOships. Then runs/seaships/results.csv and results_mean.csv are rebuilt.
#   bash scripts/evaluate.sh compare <protocol> [model ...]
#       two-level bootstrap (test recordings and training seeds) of our method minus each model, on the protocol's
#       test splits; the models default to the baselines of configs/base.yaml, MODEL_A=<name> replaces our method.
#       Seeds are paired when both sides derive from the same training runs. Writes
#       runs/seaships/compare_<a>_vs_<b>_<split>_valnms_s<seeds>_cluster_seeds.txt.
#
# Steps whose output exists are skipped; runs without weights are skipped with a note.
# Environment: PYTHON (default python), CONFIG (default configs/base.yaml).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
CONFIG=${CONFIG:-configs/base.yaml}
export PYTHONIOENCODING=utf-8 YOLO_AUTOINSTALL=False
RUNS=runs/seaships   # fixed in the code: the runs of every dataset go here

cfg() {  # cfg <dotted key>: a value of $CONFIG; lists are printed space-separated
  "$PY" -c 'import functools, sys, yaml
v = functools.reduce(lambda d, k: d[k], sys.argv[2].split("."), yaml.safe_load(open(sys.argv[1], encoding="utf-8")))
print(" ".join(map(str, v)) if isinstance(v, list) else v)' "$CONFIG" "$1"
}
fmt() {  # fmt <format> <scale> <value>, e.g. fmt %03d 100 0.9 -> 090
  "$PY" -c 'import sys; print(sys.argv[1] % round(float(sys.argv[2]) * float(sys.argv[3])))' "$1" "$2" "$3"
}
seaships() { case $1 in session | random | uploader) return 0 ;; *) return 1 ;; esac }
sliced() { [[ " $(cfg method.slicing.protocols) " == *" $1 "* ]]; }
root_of() {  # a run name without its post-hoc steps (WiSE-FT, type stage, slicing, type-stage variants)
  local n=$1
  while [[ $n =~ ^(.+)-(wise[0-9]+|fw[0-9]+|sl[0-9]+|fused|T1|prod|g05)$ ]]; do n=${BASH_REMATCH[1]}; done
  echo "$n"
}

if [ $# -lt 2 ]; then awk 'NR > 1 && /^#/ { print; next } NR > 1 { exit }' "$0"; exit 2; fi

# our run names, as src/models/wise_ft.py, slice_eval.py and type_rescore.py --save-as write them
BASE=$(cfg method.base_detector)
WISE=$BASE-wise$(fmt %03d 100 "$(cfg method.wise_ft.alpha)")
FW=fw$(fmt %02d 10 "$(cfg method.type_stage.w)")
SL=sl$(cfg method.slicing.grid)
final() { if sliced "$1"; then echo "$WISE-$SL-$FW"; else echo "$WISE-$FW"; fi; }

if [ "$1" = compare ]; then
  PROTOCOL=$2
  shift 2
  A=${MODEL_A:-$(final "$PROTOCOL")}
  if [ $# -eq 0 ]; then
    set -- $(cfg baselines.all)
    if seaships "$PROTOCOL"; then set -- "$@" $(cfg baselines.seaships); fi
  fi
  SEEDS=$(cfg seeds)
  N=$(cfg evaluation.bootstrap.n)
  BS=$(cfg evaluation.bootstrap.seed)
  for model in "$@"; do
    B=${model%.pt}
    ra=() rb=() missing=
    for s in $SEEDS; do ra+=("${A}_${PROTOCOL}_s$s"); rb+=("${B}_${PROTOCOL}_s$s"); done
    for r in "${ra[@]}" "${rb[@]}"; do [ -n "$missing" ] || [ -f "$RUNS/$r/eval/summary.json" ] || missing=$r; done
    if [ -n "$missing" ]; then echo "[skip] $A vs $B: $missing is not tested"; continue; fi
    paired=()
    if [ "$(root_of "$A")" = "$(root_of "$B")" ]; then paired=(--paired-seeds); fi
    for split in $(cfg "protocols.$PROTOCOL.test"); do
      "$PY" src/evaluation/boot_cache.py cache --split "$split" --cluster --n "$N" --seed "$BS" --runs "${ra[@]}" "${rb[@]}"
      out=$RUNS/compare_${A}_vs_${B}_${split}_valnms_s${SEEDS// /}_cluster_seeds.txt
      "$PY" src/evaluation/boot_cache.py compare --split "$split" --cluster --n "$N" --seed "$BS" --seed-boot \
        ${paired[@]+"${paired[@]}"} --a "${ra[@]}" --b "${rb[@]}" | tee "$out"
    done
  done
  exit 0
fi

PROTOCOL=$1 SEED=$2
MODEL=${3:-$BASE}
ARM=$(cfg method.type_stage.arm)_${1}_s${2}
run() { echo "${1%.pt}_${PROTOCOL}_s${SEED}"; }   # run name of a model: <model stem>_<protocol>_s<seed>
ZS=(--smd-imgsz)   # no zero-shot tests, except for SeaShips runs
if seaships "$PROTOCOL"; then
  ZS=(--smd-imgsz $(cfg evaluation.zero_shot.smd_imgsz) --aboships-imgsz $(cfg evaluation.zero_shot.aboships_imgsz))
fi

test_run() {  # test_run <run>: src/evaluation/eval_detector.py on a trained detector
  if [ -f "$RUNS/$1/eval/summary.json" ]; then echo "[skip] $1: tested"; return; fi
  if [ ! -f "$RUNS/$1/weights/best.pt" ]; then echo "[skip] $1: no weights/best.pt"; return; fi
  "$PY" src/evaluation/eval_detector.py --run "$1" "${ZS[@]}"
}
fuse() {  # fuse <detector run> <name>: Module B on a tested detector run, saved as run <name>_<protocol>_s<seed>
  if [ -f "$RUNS/$(run "$2")/eval/summary.json" ]; then echo "[skip] $(run "$2"): fused"; return; fi
  if [ ! -f "$RUNS/$1/eval/summary.json" ]; then echo "[skip] $(run "$2"): $1 is not tested"; return; fi
  "$PY" src/models/type_rescore.py --run "$1" --arm "$ARM" --top "$(cfg method.type_stage.top)" --fuse \
    --w "$(cfg method.type_stage.w)" --save-as "$2"
}

if [ "$MODEL" = "$BASE" ]; then
  test_run "$(run "$BASE")"            # the base detector
  test_run "$(run "$WISE")"            # + Module A
  fuse "$(run "$WISE")" "$WISE-$FW"    # + Module B: the method without slicing
  fuse "$(run "$BASE")" "$BASE-$FW"    # Module B without Module A (ablation)
  if sliced "$PROTOCOL"; then          # sliced inference, then Module B: the method on this protocol
    if [ -f "$RUNS/$(run "$WISE-$SL")/eval/summary.json" ]; then
      echo "[skip] $(run "$WISE-$SL"): sliced"
    elif [ -f "$RUNS/$(run "$WISE")/weights/best.pt" ]; then
      "$PY" src/models/slice_eval.py --run "$(run "$WISE")" --grid "$(cfg method.slicing.grid)" --frac "$(cfg method.slicing.frac)"
    fi
    fuse "$(run "$WISE-$SL")" "$WISE-$SL-$FW"
  fi
  echo "our method on $PROTOCOL: run $(run "$(final "$PROTOCOL")")"
else
  test_run "$(run "$MODEL")"
fi
"$PY" src/evaluation/tabulate_runs.py > "$RUNS/tabulate.log"
echo "tables: $RUNS/results.csv, $RUNS/results_mean.csv"
