"""Crops of a detector's own boxes on a protocol's train / val images, for training the type stage
on what it will see (2026-10-20; pilot of the "background" class and of detector-box positives).

The detector's per-class dets on each split (type_rescore.detector_pred6: computed once with the
official model and cached as eval/pred6_<split>_640.csv) are grouped into boxes as type_rescore does
(top 100 per image). Against the split's GT (ship_data.gt; ignore regions as det_eval treats them):
  - positive: per GT box, the highest-scoring detector box with IoU >= --pos-iou, labelled with that
    GT box's class (so the arm also sees the detector's imperfect boxes);
  - negative ("background"): a box with best score >= --neg-score whose IoU with every GT box is
    < --neg-iou and which covers no ignore region (IoA < 0.5); at most --neg-per-image per image by
    score, then at most --neg-ratio x the split's GT boxes overall, again by score.
Each is cut as type_rescore.DetCrops cuts (box + 12.5% per side, at least 4 px, clipped) from the
source image into data/crops/_det/<run>/<split>/; data/crops/_det/<run>/crops.csv lists file
(relative to data/crops/), dataset, image, split, kind (pos / neg), label (class name or
"background"), score, iou, x, y, w, h. The detector was trained on the train images, so its train
false positives are fewer and easier than on test: this is a known bias of the pilot.

    python src/data/mine_det_crops.py --run deim-r18-nomix-wise090_wsodd_s0
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SRC_PARTS = [str(Path(__file__).resolve().parents[1] / p) for p in ("data", "models", "evaluation")]
sys.path[:0] = [p for p in SRC_PARTS if p not in sys.path]  # the src/ modules import one another by name
import ship_data  # noqa: E402
import type_rescore as tr  # noqa: E402
from det_eval import box_iou  # noqa: E402

CROPS = ship_data.DATA / "crops"


def cut(job):
    root, image, rows, out = job
    im = Image.open(root / "images" / (image + ".jpg")).convert("RGB")
    for r in rows:
        px, py = max(tr.PAD * r["w"], tr.MIN_PAD), max(tr.PAD * r["h"], tr.MIN_PAD)
        box = (max(0, r["x"] - px), max(0, r["y"] - py), min(im.width, r["x"] + r["w"] + px), min(im.height, r["y"] + r["h"] + py))
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            box = (max(0, r["x"]), max(0, r["y"]), max(r["x"] + 2, r["x"] + r["w"]), max(r["y"] + 2, r["y"] + r["h"]))
        im.crop(tuple(round(v) for v in box)).save(out / r["file"], quality=95)


def mine(run, split, a):
    k = ship_data.num_classes(split)
    names = ship_data.names(split)
    images = sorted(ship_data.split_ids(split))
    sizes = ship_data.image_sizes(images, split)
    gt = ship_data.gt(images, sizes, split, classes=True)
    gt = pd.concat([gt[~gt.ignore], gt[gt.ignore].drop_duplicates(["image", "x", "y", "w", "h"])], ignore_index=True)
    boxes, _ = tr.box_scores(tr.detector_pred6(run, split), 100, k)
    by_gt = dict(tuple(gt.groupby("image")))
    pos, neg = [], []
    for image, b in boxes.groupby("image"):
        g = by_gt.get(image)
        bx = b[["x", "y", "w", "h"]].to_numpy(float)
        if g is None or not len(g):
            cand = b[b.score >= a.neg_score]
            neg += [dict(image=image, score=float(r.score), iou=0.0, x=r.x, y=r.y, w=r.w, h=r.h)
                    for r in cand.sort_values("score", ascending=False).head(a.neg_per_image).itertuples()]
            continue
        ig = g.ignore.to_numpy(bool)
        iou = box_iou(bx, g[["x", "y", "w", "h"]].to_numpy(float), ig)
        real = iou[:, ~ig] if (~ig).any() else np.zeros((len(b), 0))
        # positives: best-scoring detector box per GT box
        gr = g[~ig].reset_index(drop=True)
        for j in range(real.shape[1]):
            ok = np.flatnonzero(real[:, j] >= a.pos_iou)
            if ok.size:
                i = ok[np.argmax(b.score.to_numpy()[ok])]
                r = b.iloc[i]
                pos.append(dict(image=image, score=float(r.score), iou=float(real[i, j]), x=r.x, y=r.y, w=r.w, h=r.h,
                                label=names[int(gr.cls.iloc[j])]))
        # negatives: confident boxes on nothing
        best = real.max(1) if real.shape[1] else np.zeros(len(b))
        free = ~(iou[:, ig] >= 0.5).any(1) if ig.any() else np.ones(len(b), bool)
        for i in np.flatnonzero((best < a.neg_iou) & free & (b.score.to_numpy() >= a.neg_score)):
            r = b.iloc[i]
            neg.append(dict(image=image, score=float(r.score), iou=float(best[i]), x=r.x, y=r.y, w=r.w, h=r.h))
    neg = pd.DataFrame(neg)
    if len(neg):
        neg = neg.sort_values("score", ascending=False).groupby("image").head(a.neg_per_image)
        neg = neg.head(int(a.neg_ratio * int((~gt.ignore).sum()))).assign(label="background")
    pos = pd.DataFrame(pos)
    rows = pd.concat([pos.assign(kind="pos"), neg.assign(kind="neg")], ignore_index=True)
    rows["split"], rows["dataset"] = split, ship_data.dataset(split)
    rows["file"] = ["_det/%s/%s/%s_%d_%s.jpg" % (run, split, im, i, kd) for i, (im, kd) in enumerate(zip(rows.image, rows.kind))]
    return rows, int((~gt.ignore).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--pos-iou", type=float, default=0.5)
    ap.add_argument("--neg-iou", type=float, default=0.3)
    ap.add_argument("--neg-score", type=float, default=0.1)
    ap.add_argument("--neg-per-image", type=int, default=3)
    ap.add_argument("--neg-ratio", type=float, default=1.0)
    a = ap.parse_args()
    protocol = a.run.split("_")[-2]
    out = CROPS / "_det" / a.run
    allrows = []
    for sp in a.splits:
        split = "%s_%s" % (protocol, sp)
        (out / split).mkdir(parents=True, exist_ok=True)
        rows, n_gt = mine(a.run, split, a)
        root = ship_data.root(split)
        jobs = [(root, im, g.to_dict("records"), CROPS) for im, g in rows.groupby("image")]
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(cut, jobs))
        print("%s: %d GT boxes, %d positive and %d negative crops (scores of negatives: median %.2f)" % (
            split, n_gt, (rows.kind == "pos").sum(), (rows.kind == "neg").sum(),
            rows[rows.kind == "neg"].score.median() if (rows.kind == "neg").any() else float("nan")), flush=True)
        allrows.append(rows.assign(split=sp))
    pd.concat(allrows, ignore_index=True)[["file", "dataset", "image", "split", "kind", "label", "score", "iou", "x", "y", "w", "h"]] \
        .to_csv(out / "crops.csv", index=False)


if __name__ == "__main__":
    main()
