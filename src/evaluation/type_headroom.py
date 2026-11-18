"""How much of a detector's error is ship type, and how the type stage's two sources of type evidence differ
(the error analysis of the paper). For our detector with WiSE-FT
(deim-r18-nomix-wise090) on each dataset's test split and seed, from saved predictions only (CPU):

  detector  6-class mAP50 / mAP50-95 of the detector's own per-class output (as results.csv);
  replace   the classifier's types on the detector's boxes (w = 1 of the fusion sweep in
            runs/types/rescore/<run>__convnext-t-ft_<p>_s<k>__fuse_w05.json);
  fused     the type stage at w = 0.5 (same file);
  oracle    a typing oracle: the detector's 100 best boxes per image (as the type stage sees them), each box that
            matches a ground-truth ship (greedy in score order, IoU >= 0.5, each ship once; type_confusion.match)
            given that ship's type at the box's best detector score; unmatched boxes keep their per-class scores.
            oracle - detector is the mAP that perfect type recognition would add with the same boxes.

On the boxes with best detector score >= 0.25 that match a ship (the boxes of type_confusion.py): the detector's
top-1 type, the classifier's top-1 type and the fused top-1 type. The classifier's probabilities are recovered
exactly from the saved scores, p_c = f_c^2 / (max(s_c, 1e-3 s_max) s_max) at w = 0.5 (type_rescore.fuse); classes the
fused file dropped (f < 1e-4) count as p = 0. Counts: both right, only the detector right, only the classifier right,
both wrong, and which top-1 the fused type follows when the two disagree.

Writes runs/types/headroom/<protocol>.json (per seed and pooled).

    python src/evaluation/type_headroom.py [protocol ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_PARTS = [str(Path(__file__).resolve().parents[1] / p) for p in ("data", "models", "evaluation")]
sys.path[:0] = [p for p in SRC_PARTS if p not in sys.path]  # the src/ modules import one another by name
import ship_data  # noqa: E402
import type_confusion as tconf  # noqa: E402
from eval_detector import RUNS, class_ap, class_dets  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "types" / "headroom"
DET, FUSED = "deim-r18-nomix-wise090", "deim-r18-nomix-wise090-fw05"
KEY = ["image", "x", "y", "w", "h"]


def scores_by_box(dt6, k):
    """Box table (KEY, score = best class score, cls = its class) and the K-vector of class scores per box,
    boxes keyed on coordinates rounded to 3 decimals as type_confusion.boxes_of does."""
    dt6 = dt6.assign(**{c: dt6[c].round(3) for c in ("x", "y", "w", "h")})
    s = dt6.pivot_table(index=KEY, columns="pred_cls", values="score", aggfunc="max",
                        fill_value=0.0).reindex(columns=range(k), fill_value=0.0)
    b = s.index.to_frame(index=False)
    sv = s.to_numpy()
    b["score"], b["cls"] = sv.max(1), sv.argmax(1)
    return b, sv


def oracle_dets(boxes, sv, gt):
    """Per-class dets of the typing oracle on the 100 best boxes per image."""
    keep = boxes.sort_values("score", ascending=False).groupby("image").head(100).index
    b, s = boxes.loc[keep].reset_index(drop=True), sv[keep]
    m, _ = tconf.match(b.assign(row=np.arange(len(b))), gt, thr=0.0)
    hit = m[m.gt_cls >= 0]
    rows = [pd.DataFrame({"image": hit.image, "x": hit.x, "y": hit.y, "w": hit.w, "h": hit.h, "score": hit.score,
                          "pred_cls": hit.gt_cls.astype(int)})]
    miss = m[m.gt_cls < 0].row.to_numpy()
    k = s.shape[1]
    for c in range(k):
        v = s[miss, c]
        nz = v > 0
        rows.append(pd.DataFrame({col: b.loc[miss[nz], col].to_numpy() for col in KEY}).assign(score=v[nz], pred_cls=c))
    return pd.concat(rows, ignore_index=True)


def agreement(boxes, sv, fv, gt, thr=0.25):
    """Top-1 types of the detector, the classifier (p recovered from the fused scores) and the fusion on the matched
    boxes with best detector score >= thr."""
    m, _ = tconf.match(boxes.assign(row=np.arange(len(boxes))), gt, thr=thr)
    m = m[m.gt_cls >= 0]
    r = m.row.to_numpy()
    s, f = sv[r], fv[r]
    smax = s.max(1, keepdims=True)
    p = f ** 2 / (np.maximum(s, 1e-3 * smax) * smax)
    y = m.gt_cls.to_numpy().astype(int)
    d_ok, c_ok, f_ok = s.argmax(1) == y, p.argmax(1) == y, f.argmax(1) == y
    dis = s.argmax(1) != p.argmax(1)
    return dict(boxes=int(len(y)), both_right=int((d_ok & c_ok).sum()), only_detector=int((d_ok & ~c_ok).sum()),
                only_classifier=int((~d_ok & c_ok).sum()), both_wrong=int((~d_ok & ~c_ok).sum()),
                disagree=int(dis.sum()), fused_follows_detector=int((dis & (f.argmax(1) == s.argmax(1))).sum()),
                fused_follows_classifier=int((dis & (f.argmax(1) == p.argmax(1))).sum()),
                fused_right_when_disagree=int((dis & f_ok).sum()),
                detector_acc=float(d_ok.mean()), classifier_acc=float(c_ok.mean()), fused_acc=float(f_ok.mean()),
                p_sum_median=float(np.median(p.sum(1))), matched_without_fused=int((f.max(1) == 0).sum()))


def main(protocols):
    OUT.mkdir(parents=True, exist_ok=True)
    for proto in protocols:
        split = proto + "_test"
        k = ship_data.num_classes(split)
        images = sorted(ship_data.split_ids(split))
        sizes = ship_data.image_sizes(images, split)
        gt = ship_data.gt(images, sizes, split, classes=True)
        gt = pd.concat([gt[~gt.ignore], gt[gt.ignore].drop_duplicates(KEY)], ignore_index=True)
        res = dict(protocol=proto, split=split, seeds={})
        for seed in (0, 1, 2):
            run, frun = "%s_%s_s%d" % (DET, proto, seed), "%s_%s_s%d" % (FUSED, proto, seed)
            fj = ROOT / "runs" / "types" / "rescore" / ("%s__convnext-t-ft_%s_s%d__fuse_w05.json" % (run, proto, seed))
            if not fj.exists():
                print("missing", fj)
                continue
            sweep = json.loads(fj.read_text())["sweep"][split]
            det6 = class_dets(RUNS / run / "eval", split)
            boxes, sv = scores_by_box(det6, k)
            fb, fsv = scores_by_box(class_dets(RUNS / frun / "eval", split), k)
            fpos = {t: i for i, t in enumerate(map(tuple, fb[KEY].to_numpy()))}
            fv = np.zeros_like(sv)
            for i, t in enumerate(map(tuple, boxes[KEY].to_numpy())):
                j = fpos.get(t)
                if j is not None:
                    fv[i] = fsv[j]
            orc = class_ap(oracle_dets(boxes, sv, gt), images, sizes, split)
            res["seeds"][seed] = dict(
                detector={m: 100 * v for m, v in class_ap(det6, images, sizes, split).items() if m in ("map50", "map")},
                replace={m: sweep["1.0"][m] for m in ("map50", "map")},
                fused={m: sweep["0.5"][m] for m in ("map50", "map")},
                oracle={m: 100 * orc[m] for m in ("map50", "map")},
                agreement=agreement(boxes, sv, fv, gt), boxes_without_fused=int((fv.max(1) == 0).sum()))
            print(proto, seed, json.dumps({kk: v for kk, v in res["seeds"][seed].items() if kk != "agreement"}),
                  res["seeds"][seed]["agreement"], flush=True)
        (OUT / (proto + ".json")).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["session", "smdplus", "aboships", "mcships", "wsodd"]))
