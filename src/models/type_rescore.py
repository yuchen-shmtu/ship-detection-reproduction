"""Apply a second-stage type arm (src/models/type_classifier.py) to a detector run's boxes and score
the result with det_eval's 6-class COCO AP, as eval_detector.class_ap scores the detectors.

For the run's own test split and cross_test, the detector's class-agnostic dets
(runs/seaships/<run>/eval/pred_<split>_<size>.csv: NMS across classes at IoU 0.7, score >=
0.001), at most --top per image by score, are cropped from the pool image as build_crops.py
crops GT boxes (box plus 12.5% of its size per side, at least 4 px, clipped), classified by
the arm (calibrated probabilities p(c), temperature from its metrics.json), and expanded to
per-class dets: each box gives six, scored s_det * p(c) (--mode prob) or s_det * p(c) / max p
(--mode relative, so the top type keeps the detector's score), with the detector's score
s_det as the objectness. The detector's own class decision is dropped.

Reported per split, all x100: the detector's own 6-class mAP50 / mAP50-95 (its per-class
output, as in results.csv), the re-typed one, and the oracle bound (each box given the class
of the GT box it overlaps at IoU >= 0.5, as in the error analysis). Written to
runs/types/rescore/<run>__<arm-run>__<mode>.json.

    python src/models/type_rescore.py --run rtdetr-r18_session_s0 --arm dinov2-s-probe_session_s0
"""
import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SRC_PARTS = [str(Path(__file__).resolve().parents[1] / p) for p in ("data", "models", "evaluation")]
sys.path[:0] = [p for p in SRC_PARTS if p not in sys.path]  # the src/ modules import one another by name
import ship_data  # noqa: E402
import type_classifier as tc  # noqa: E402
from det_eval import box_iou  # noqa: E402
from eval_detector import RUNS, SEA, class_ap, class_dets, image_sizes, seaships_files, seaships_gt  # noqa: E402

PAD, MIN_PAD = 0.125, 4  # as build_crops.py


class DetCrops(Dataset):
    """One item per image: the letterboxed crops of its selected dets and their row indices."""

    def __init__(self, dt, canvas, root=SEA, pad=PAD):
        self.groups = list(dt.groupby("image").groups.items())
        self.dt, self.canvas, self.root, self.pad = dt, canvas, root, pad

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        image, idx = self.groups[i]
        im = Image.open(self.root / "images" / (image + ".jpg"))
        crops = []
        for _, r in self.dt.loc[idx].iterrows():
            px, py = max(self.pad * r.w, MIN_PAD), max(self.pad * r.h, MIN_PAD)
            box = (max(0, r.x - px), max(0, r.y - py), min(im.width, r.x + r.w + px), min(im.height, r.y + r.h + py))
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                box = (max(0, r.x), max(0, r.y), max(r.x + 2, r.x + r.w), max(r.y + 2, r.y + r.h))
            crops.append(tc.letterbox(im.crop(tuple(round(v) for v in box)), *self.canvas))
        return torch.stack(crops), torch.as_tensor(np.asarray(idx))


def concat(batch):
    return torch.cat([b[0] for b in batch]), torch.cat([b[1] for b in batch])


def load_arm(arm_run):
    """The arm's logits function, canvas and temperature. The function also carries the arm's crop
    padding (.pad) and output size (.n_out: the classes, plus a last "background" class for arms trained
    with detector negatives, 2026-10-20), both read from its metrics.json."""
    d = tc.OUT / arm_run
    m = json.loads((d / "metrics.json").read_text())
    arm = m["arm"]
    n_out = len(m.get("classes") or []) or ship_data.num_classes(m["protocol"])
    net = tc.Net(arm, n_out).cuda().eval()
    canvas = tc.CANVAS["dinov2" if arm.startswith("dinov2") else "convnext"]
    if arm.endswith("probe"):
        pk = pickle.loads((d / "probe.pkl").read_bytes())
        sc, lr = pk["scaler"], pk["probe"]

        def logits(x):
            return lr.decision_function(sc.transform(net.features(x).float().cpu().numpy()))
    else:
        net.load_state_dict(torch.load(d / "model.pt", map_location="cuda"))

        def logits(x):
            return net(x).float().cpu().numpy()
    logits.pad, logits.n_out = m.get("pad", PAD), n_out
    return logits, canvas, m["temperature"]


@torch.no_grad()
def type_probs(dt, logits, canvas, t, workers, split="session_test"):
    loader = DataLoader(DetCrops(dt, canvas, ship_data.root(split), getattr(logits, "pad", PAD)), batch_size=4,
                        num_workers=workers, collate_fn=concat, timeout=600 if workers else 0)
    p = np.zeros((len(dt), getattr(logits, "n_out", ship_data.num_classes(split))), np.float32)
    pos = {k: i for i, k in enumerate(dt.index)}
    for x, idx in loader:
        with torch.autocast("cuda"):
            lg = logits(x.cuda())
        p[[pos[int(k)] for k in idx]] = torch.softmax(torch.as_tensor(lg, dtype=torch.float32) / t, 1).numpy()
    return p


def oracle(dt, gt):
    o = dt.copy()
    gg = dict(tuple(gt.groupby("image")))
    for im, idx in o.groupby("image").groups.items():
        g = gg.get(im)
        if g is None:
            continue
        d = o.loc[idx]
        iou = box_iou(d[["x", "y", "w", "h"]].to_numpy(float), g[["x", "y", "w", "h"]].to_numpy(float),
                      np.zeros(len(g), bool))
        hit = iou.max(1) >= 0.5
        o.loc[idx[hit], "pred_cls"] = g.cls.to_numpy()[iou.argmax(1)[hit]]
    return o


FUSE_W = [round(0.1 * k, 1) for k in range(11)]


def detector_pred6(run, split):
    """A run's per-class dets on a split: eval/pred6_<split>_<size>.csv (DETR-family and
    torchvision runs) or ultralytics val's eval/val_<split>/predictions.json (YOLO runs); for a
    DETR-family run computed and cached for splits eval_detector.py does not test (e.g. session_val)."""
    ev = RUNS / run / "eval"
    found = sorted(ev.glob("pred6_%s_*.csv" % split))
    if found:
        return pd.read_csv(found[0], dtype={"image": str})
    if (ev / ("val_" + split) / "predictions.json").exists():
        return class_dets(ev, split)
    import rtdetr_official
    import tv_detector

    model = rtdetr_official.load(run.rsplit("_", 2)[0], RUNS / run / "weights" / "best.pt")
    dt6, _ = rtdetr_official.predict(model, tv_detector.frames_loader(seaships_files(split)), rtdetr_official.SIZE)
    del model
    torch.cuda.empty_cache()
    dt6.to_csv(ev / ("pred6_%s_%d.csv" % (split, rtdetr_official.SIZE)), index=False)
    return dt6


def box_scores(dt6, top, n_classes=len(tc.CLASSES)):
    """One row per detected box with its class scores (a query's (query, class) dets share
    its box exactly, as do an anchor's dets in ultralytics' multi-label NMS); the top boxes per
    image by their best class score."""
    s = dt6.pivot_table(index=["image", "x", "y", "w", "h"], columns="pred_cls", values="score", aggfunc="max",
                        fill_value=0.0).reindex(columns=range(n_classes), fill_value=0.0)
    boxes = s.index.to_frame(index=False)
    boxes["score"] = s.to_numpy().max(1)
    keep = boxes.sort_values("score", ascending=False).groupby("image").head(top).index
    return boxes.loc[keep].reset_index(drop=True), s.to_numpy()[keep]


def fuse(boxes, s, p, w):
    """Per-class dets scored s_c^(1-w) * (s_max * p_c)^w: w = 0 is the detector alone, w = 1
    its best score times the type probabilities. Classes the detector did not output get a
    floor of 1e-3 * s_max, so the type stage can still raise them. An arm with a "background" class
    gives p one more column; only the real classes are used, so their probabilities sum to
    1 - p(background) and a box the arm calls background loses score in every class."""
    p = p[:, : s.shape[1]]
    smax = s.max(1, keepdims=True)
    f = np.maximum(s, 1e-3 * smax) ** (1 - w) * (smax * p) ** w
    k = s.shape[1]
    dt6 = pd.DataFrame({c: np.repeat(boxes[c].to_numpy(), k) for c in ("image", "x", "y", "w", "h")})
    dt6["score"] = f.ravel()
    dt6["pred_cls"] = np.tile(np.arange(k), len(boxes))
    return dt6


def save_fused_run(a, protocol, kept, res):
    """--save-as NAME: write runs/seaships/NAME_<protocol>_s<seed>/eval/ for the fused method, so
    tabulate_runs.py and compare_runs.py treat it like any run: pred6_<split>_640.csv (the fused
    per-class dets at the chosen w, score >= 1e-4), the detector's class-agnostic pred_*.csv
    (the type stage moves no box), and summary.json = the detector's with seaships_coco
    replaced by the fused scores and a "fusion" record."""
    seed = a.run.rsplit("_s", 1)[1]
    out = RUNS / ("%s_%s_s%s" % (a.save_as, protocol, seed)) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    det = RUNS / a.run / "eval"
    summ = json.loads((det / "summary.json").read_text())
    w = float(res["w"])
    for split, (boxes, s, p, images, sizes) in kept.items():
        dt6 = fuse(boxes, s, p, w)
        dt6 = dt6[dt6.score >= 1e-4]
        dt6.to_csv(out / ("pred6_%s_640.csv" % split), index=False)
        summ["seaships_coco"][split] = class_ap(dt6, images, sizes, split)
    for f in det.glob("pred_*.csv"):
        shutil.copy2(f, out / f.name)
    # ultralytics val's own scores (YOLO runs) describe the detector's dets, not the fused ones
    summ.update(run=out.parent.name, seaships={}, fusion=dict(detector=a.run, arm=a.arm, w=w, top=a.top))
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print("saved", out.parent.name, flush=True)


def fuse_main(a, protocol, logits, canvas, t):
    """--fuse: sweep w on the protocol's val split, report the test splits at the best w, or at
    --w when given (the val sweep is still recorded). Since 2026-10-13 the method uses w = 0.5 for
    every dataset (the value SeaShips' val picked for our detector before the other datasets
    ran); the small SMD-Plus val split (5 videos) picked w unreliably. --no-val (with --w) skips
    the val split, for detectors whose val predictions are not stored (the ultralytics YOLOs);
    --grid scores only the given w (plus 0, 1 and --w) instead of the 0.1 grid."""
    res = dict(run=a.run, arm=a.arm, top=a.top, sweep={})
    kept = {}
    sea = ship_data.dataset(protocol) == "seaships"
    tests = ["cross_test"] if sea else ["mcships_testdedup"] if protocol == "mcships" else []
    grid = FUSE_W
    if a.grid:
        grid = sorted({round(float(x), 1) for x in a.grid.split(",")} | {0.0, 1.0}
                      | ({round(a.w, 1)} if a.w is not None else set()))
    for split in ([] if a.no_val else [protocol + "_val"]) + [protocol + "_test", *tests]:
        images = sorted(Path(f).stem for f in seaships_files(split))
        sizes = image_sizes(images, split)
        boxes, s = box_scores(detector_pred6(a.run, split), a.top, ship_data.num_classes(split))
        p = type_probs(boxes, logits, canvas, t, a.workers, split)
        if split != protocol + "_val":
            kept[split] = (boxes, s, p, images, sizes)
        res["sweep"][split] = {}
        for w in grid:
            r = class_ap(fuse(boxes, s, p, w), images, sizes, split)
            res["sweep"][split][str(w)] = dict(map50=100 * r["map50"], map=100 * r["map"])
        print(split, {w: round(v["map"], 1) for w, v in res["sweep"][split].items()}, flush=True)
    val = res["sweep"].get(protocol + "_val")
    res["w_val"] = max(val, key=lambda w: val[w]["map"]) if val else None
    res["w"] = str(round(a.w, 1)) if a.w is not None else res["w_val"]
    res["w_rule"] = "fixed" if a.w is not None else "val"
    for split in kept:
        res[split] = dict(detector=res["sweep"][split]["0.0"], fused=res["sweep"][split][res["w"]],
                          typed=res["sweep"][split]["1.0"])
    print("w %s (val's pick %s):" % (res["w"], res["w_val"]), json.dumps({s: res[s] for s in kept}), flush=True)
    if a.save_as:
        save_fused_run(a, protocol, kept, res)
    out = tc.OUT / "rescore"
    out.mkdir(parents=True, exist_ok=True)
    tag = "" if a.w is None else "_w%02d" % round(10 * a.w)
    (out / ("%s__%s__fuse%s.json" % (a.run, a.arm, tag))).write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="detector run, e.g. rtdetr-r18_session_s0")
    ap.add_argument("--arm", required=True, help="type arm run in runs/types/, e.g. dinov2-s-probe_session_s0")
    ap.add_argument("--top", type=int, default=100, help="dets per image to re-type, by score")
    ap.add_argument("--mode", default="prob", choices=["prob", "relative"])
    ap.add_argument("--fuse", action="store_true",
                    help="DETR-family and YOLO runs: fuse the detector's per-class scores with the type probabilities")
    ap.add_argument("--save-as", help="--fuse: also write the fused method as run NAME_<protocol>_s<seed>")
    ap.add_argument("--w", type=float, help="--fuse: use this w instead of the val split's pick (a 0.1 grid value)")
    ap.add_argument("--no-val", action="store_true", help="--fuse with --w: skip the val split (YOLO runs)")
    ap.add_argument("--grid", help="--fuse: comma-separated w values to score (0, 1 and --w always), default 0.0-1.0")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if a.no_val and a.w is None:
        ap.error("--no-val needs --w")
    ev = RUNS / a.run / "eval"
    protocol = a.run.split("_")[-2]
    logits, canvas, t = load_arm(a.arm)
    if a.fuse:
        return fuse_main(a, protocol, logits, canvas, t)
    res = dict(run=a.run, arm=a.arm, top=a.top, mode=a.mode)
    for split in (protocol + "_test", "cross_test"):
        images = sorted(Path(f).stem for f in seaships_files(split))
        sizes = image_sizes(images)
        own = class_ap(class_dets(ev, split), images, sizes)
        dt = pd.read_csv(next(ev.glob("pred_%s_*.csv" % split)), dtype={"image": str})
        dt = dt.sort_values("score", ascending=False).groupby("image").head(a.top).reset_index(drop=True)
        p = type_probs(dt, logits, canvas, t, a.workers)
        q = p / p.max(1, keepdims=True) if a.mode == "relative" else p
        dt6 = pd.DataFrame({k: np.repeat(dt[k].to_numpy(), 6) for k in ("image", "x", "y", "w", "h")})
        dt6["score"] = (dt.score.to_numpy()[:, None] * q).ravel()
        dt6["pred_cls"] = np.tile(np.arange(6), len(dt))
        typed = class_ap(dt6, images, sizes)
        orc = class_ap(oracle(dt, seaships_gt(images, sizes, classes=True)), images, sizes)
        res[split] = {k: dict(map50=100 * v["map50"], map=100 * v["map"]) for k, v in
                      (("detector", own), ("typed", typed), ("oracle", orc))}
        print(split, json.dumps({k: {m: round(x, 1) for m, x in v.items()} for k, v in res[split].items()}), flush=True)
    out = tc.OUT / "rescore"
    out.mkdir(parents=True, exist_ok=True)
    (out / ("%s__%s__%s.json" % (a.run, a.arm, a.mode))).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
