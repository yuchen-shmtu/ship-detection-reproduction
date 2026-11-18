"""Tabulate every runs/seaships/<run>/eval*/summary.json (from eval_detector.py).

Writes runs/seaships/results.csv (one row per run and checkpoint) and results_mean.csv
(mean and std over seeds per model, protocol and checkpoint; std has ddof 1), and prints
the second. Columns, all x100:
  own_*       ultralytics val on the run's own protocol test split (6 classes, as papers
              report); empty for torchvision runs
  cross_*     the same on cross_test = random_test & session_test
  own_coco_*, cross_coco_*
              det_eval 6-class COCO AP of the same per-class-NMS dets, every model: the
              columns to compare ultralytics with torchvision models
  ag_*     det_eval, class-agnostic, own test split; ag_small = mAP50-95 on boxes < 32^2 px
  smd_*    det_eval, class-agnostic, SMD-VIS lenient ignore set, per imgsz and subset
  abo_*    det_eval, class-agnostic, ABOships lenient ignore set, per imgsz (runs that have it)

    python src/evaluation/tabulate_runs.py
"""
import json
from pathlib import Path

import pandas as pd

RUNS = Path(__file__).resolve().parents[2] / "runs" / "seaships"


def row(s):
    p = s["protocol"] + "_test"
    r = {}
    for tag, split in (("own", p), ("cross", "cross_test")):
        for col, key in ((tag, "seaships"), (tag + "_coco", "seaships_coco")):
            cell = s.get(key, {}).get(split)
            r[col + "_map50"], r[col + "_map"] = (cell["map50"], cell["map"]) if cell else (float("nan"),) * 2
    for split, cell in s.get("seaships_coco", {}).items():  # other in-domain test sets, e.g. mcships_testdedup
        if split not in (p, "cross_test"):
            tag = split.split("_", 1)[1]
            r[tag + "_coco_map50"], r[tag + "_coco_map"] = cell["map50"], cell["map"]
    r.update(ag_map50=s["seaships_agnostic"][p]["all"]["map50"], ag_map=s["seaships_agnostic"][p]["all"]["map"],
             ag_small=s["seaships_agnostic"][p]["small"]["map"])
    for key, cell in s["smd"].items():
        mode, imgsz = key.split("_")
        if mode != "lenient":
            continue
        for subset in ("all", "onshore", "onboard"):
            tag = "smd%s_%s" % (imgsz, subset)
            r[tag + "_map50"], r[tag + "_map"] = cell[subset]["all"]["map50"], cell[subset]["all"]["map"]
        r["smd%s_all_small" % imgsz] = cell["all"]["small"]["map"]
    for key, cell in s.get("aboships", {}).items():
        mode, imgsz = key.split("_")
        if mode == "lenient":
            r["abo%s_map50" % imgsz], r["abo%s_map" % imgsz] = cell["all"]["map50"], cell["all"]["map"]
            r["abo%s_small" % imgsz] = cell["small"]["map"]
    return r


def main():
    rows = []
    for f in sorted(RUNS.glob("*/eval*/summary.json")):
        run = f.parent.parent.name
        model, protocol, seed = run.rsplit("_", 2)
        ckpt = "best" if f.parent.name == "eval" else f.parent.name[len("eval_"):]
        rows.append(dict(run=run, model=model, protocol=protocol, seed=int(seed[1:]), ckpt=ckpt,
                         **{k: v * 100 for k, v in row(json.loads(f.read_text())).items()}))
    df = pd.DataFrame(rows)
    df.round(3).to_csv(RUNS / "results.csv", index=False)  # 3 decimals: seed means are taken from this file

    keys = ["model", "protocol", "ckpt"]
    num = df.columns.drop(keys + ["run", "seed"])
    g = df.groupby(keys)
    mean = g[num].agg(["mean", "std"]).round(1)
    mean.insert(0, ("n_seeds", ""), g.size())
    mean.to_csv(RUNS / "results_mean.csv")
    cell = pd.DataFrame({c: mean[(c, "mean")].map("{:.1f}".format) + " ± " + mean[(c, "std")].map("{:.1f}".format)
                         for c in num})
    cell.insert(0, "n_seeds", g.size())
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.max_colwidth", 20):
        print(cell.T)


if __name__ == "__main__":
    main()
