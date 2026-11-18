"""Variants of the type stage on one detector run, from a single pass of the classifier:

  base   the method's rule, f_c = max(s_c, eps*s_max)^(1-w) * (s_max * p_c)^w with the arm's temperature T (reproduces
         the run saved by type_rescore.py --fuse --w; checked against its summary)
  g05    the deployment gate: boxes with s_max < 0.05 keep the detector's scores (w = 0), the others are fused as base;
         also reports the boxes re-typed per image
  T1     base with T = 1 (no temperature scaling)
  prod   the product rule f_c = max(s_c, eps*s_max) * p_c, which at w = 0.5 ranks boxes as the base rule would without
         its s_max factor (sqrt is monotone)

For the run's test split (and SeaShips' cross_test, McShips' deduplicated test), each variant is written as a regular run
runs/seaships/<save-as>-<variant>_<protocol>_s<seed>/eval/ (pred6 CSV, the detector's class-agnostic CSVs, summary.json
with seaships_coco replaced and a "fusion" record), so tabulate_runs.py and boot_cache.py treat it like
any model; the summary of all variants goes to runs/types/rescore/<run>__<arm>__variants.json.

    python src/models/type_variants.py --run deim-r18-nomix-wise090_wsodd_s0 --arm convnext-t-ft_wsodd_s0 --save-as deim-r18-nomix-wise090-fw05
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

SRC_PARTS = [str(Path(__file__).resolve().parents[1] / p) for p in ("data", "models", "evaluation")]
sys.path[:0] = [p for p in SRC_PARTS if p not in sys.path]  # the src/ modules import one another by name
import ship_data  # noqa: E402
import type_classifier as tc  # noqa: E402
import type_rescore as tr  # noqa: E402
from eval_detector import RUNS, class_ap, image_sizes, seaships_files  # noqa: E402

EPS, GATE = 1e-3, 0.05


@torch.no_grad()
def logits_of(boxes, logits, canvas, workers, split):
    loader = DataLoader(tr.DetCrops(boxes, canvas, ship_data.root(split), getattr(logits, "pad", tr.PAD)), batch_size=4,
                        num_workers=workers, collate_fn=tr.concat, timeout=600 if workers else 0)
    out = np.zeros((len(boxes), logits.n_out), np.float32)
    pos = {k: i for i, k in enumerate(boxes.index)}
    for x, idx in loader:
        with torch.autocast("cuda"):
            lg = logits(x.cuda())
        out[[pos[int(k)] for k in idx]] = np.asarray(lg, dtype=np.float32)
    return out


def softmax(z, t):
    z = z / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def dets(boxes, f):
    k = f.shape[1]
    dt6 = pd.DataFrame({c: np.repeat(boxes[c].to_numpy(), k) for c in ("image", "x", "y", "w", "h")})
    dt6["score"] = f.ravel()
    dt6["pred_cls"] = np.tile(np.arange(k), len(boxes))
    return dt6


def variants(boxes, s, lg, t, w):
    k = s.shape[1]
    p_t, p_1 = softmax(lg, t)[:, :k], softmax(lg, 1.0)[:, :k]
    smax = s.max(1, keepdims=True)
    floor = np.maximum(s, EPS * smax)
    base = floor ** (1 - w) * (smax * p_t) ** w
    gated = np.where(smax >= GATE, base, floor)
    return {"base": base, "g05": gated, "T1": floor ** (1 - w) * (smax * p_1) ** w, "prod": floor * p_t}, \
        float((smax >= GATE).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--save-as", required=True, help="the base variant's run name, e.g. deim-r18-nomix-wise090-fw05")
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    protocol, seed = a.run.split("_")[-2], a.run.rsplit("_s", 1)[1]
    logits, canvas, t = tr.load_arm(a.arm)
    splits = [protocol + "_test"] + (["cross_test"] if ship_data.dataset(protocol) == "seaships" else []) \
        + (["mcships_testdedup"] if protocol == "mcships" else [])
    res = dict(run=a.run, arm=a.arm, w=a.w, top=a.top, temperature=t, gate=GATE)
    det_summ = json.loads((RUNS / a.run / "eval" / "summary.json").read_text())
    outs = {}
    for split in splits:
        images = sorted(Path(f).stem for f in seaships_files(split))
        sizes = image_sizes(images, split)
        boxes, s = tr.box_scores(tr.detector_pred6(a.run, split), a.top, ship_data.num_classes(split))
        lg = logits_of(boxes, logits, canvas, a.workers, split)
        v, n_gate = variants(boxes, s, lg, t, a.w)
        res[split] = {"boxes_per_image": len(boxes) / len(images), "gated_boxes_per_image": n_gate / len(images)}
        for name, f in v.items():
            dt6 = dets(boxes, f)
            dt6 = dt6[dt6.score >= 1e-4]
            r = class_ap(dt6, images, sizes, split)
            res[split][name] = dict(map50=100 * r["map50"], map=100 * r["map"])
            outs.setdefault(name, {})[split] = (dt6, r)
        print(split, json.dumps({k: v for k, v in res[split].items()}), flush=True)
    for name, per in outs.items():
        if name == "base":
            continue  # already saved by type_rescore.py --save-as; compared below
        run = "%s-%s_%s_s%s" % (a.save_as, name, protocol, seed)
        out = RUNS / run / "eval"
        out.mkdir(parents=True, exist_ok=True)
        summ = json.loads(json.dumps(det_summ))
        for split, (dt6, r) in per.items():
            dt6.to_csv(out / ("pred6_%s_640.csv" % split), index=False)
            summ["seaships_coco"][split] = r
        for f in (RUNS / a.run / "eval").glob("pred_*.csv"):
            shutil.copy2(f, out / f.name)
        summ.update(run=run, seaships={}, fusion=dict(detector=a.run, arm=a.arm, w=a.w, top=a.top, variant=name,
                                                      gate=GATE if name == "g05" else None,
                                                      temperature=1.0 if name == "T1" else t))
        (out / "summary.json").write_text(json.dumps(summ, indent=1))
        print("saved", run, flush=True)
    saved = RUNS / ("%s_%s_s%s" % (a.save_as, protocol, seed)) / "eval" / "summary.json"
    if saved.exists():
        ref = json.loads(saved.read_text())["seaships_coco"][protocol + "_test"]
        res["base_check"] = dict(saved_map=100 * ref["map"], recomputed_map=res[protocol + "_test"]["base"]["map"])
        print("base check: saved %.3f, recomputed %.3f" % (100 * ref["map"], res[protocol + "_test"]["base"]["map"]),
              flush=True)
    out = tc.OUT / "rescore"
    out.mkdir(parents=True, exist_ok=True)
    (out / ("%s__%s__variants.json" % (a.run, a.arm))).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
