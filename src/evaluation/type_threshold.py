"""Accuracy and cost of the type stage when it re-types only confident boxes.

For one fused run (its runs/types/rescore/<det>__<arm>__fuse.json gives the detector run, the
crop arm and the val-chosen w), boxes whose best detector score is below --thr keep the
detector's own class scores (w = 0) and only the others are cropped and classified, as a
deployment would do. Per threshold, on the protocol's test split and cross_test:
  - boxes re-typed per image (mean), and the fused 6-class COCO mAP50 / mAP50-95 (det_eval);
  - type-stage latency per image on cross_test: the frame already decoded in memory (the
    detector has it), so only crop + letterbox + the arm's forward (fp16 autocast, one batch
    per image) + softmax, timed with CUDA syncs; the first pass is warm-up, the mean of the
    next two passes is kept. Add the detector's own latency (src/evaluation/model_cost.py) for an
    end-to-end figure.
Written to runs/types/rescore/<det>__<arm>__threshold.json. Run with the GPU idle.

    python src/evaluation/type_threshold.py --fuse-json runs/types/rescore/deim-r18-nomix_session_s0__convnext-t-ft_session_s0__fuse.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SRC_PARTS = [str(Path(__file__).resolve().parents[1] / p) for p in ("data", "models", "evaluation")]
sys.path[:0] = [p for p in SRC_PARTS if p not in sys.path]  # the src/ modules import one another by name
import type_classifier as tc  # noqa: E402
import type_rescore as tr  # noqa: E402
from eval_detector import SEA, class_ap, image_sizes, seaships_files  # noqa: E402

THRESHOLDS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]


def crop_box(im, r):
    px, py = max(tr.PAD * r.w, tr.MIN_PAD), max(tr.PAD * r.h, tr.MIN_PAD)
    box = (max(0, r.x - px), max(0, r.y - py), min(im.width, r.x + r.w + px), min(im.height, r.y + r.h + py))
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        box = (max(0, r.x), max(0, r.y), max(r.x + 2, r.x + r.w), max(r.y + 2, r.y + r.h))
    return im.crop(tuple(round(v) for v in box))


@torch.no_grad()
def time_type_stage(boxes, logits, canvas, t, passes=2):
    """ms per image for crop + letterbox + forward + softmax of the given boxes, frames in memory."""
    frames = {im: Image.open(SEA / "images" / (im + ".jpg")).convert("RGB") for im in boxes.image.unique()}
    for im in frames.values():
        im.load()
    groups = list(boxes.groupby("image"))
    out = []
    for k in range(passes + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for image, g in groups:
            x = torch.stack([tc.letterbox(crop_box(frames[image], r), *canvas) for r in g.itertuples()]).cuda()
            with torch.autocast("cuda"):
                lg = logits(x)
            torch.softmax(torch.as_tensor(lg, dtype=torch.float32) / t, 1)
            torch.cuda.synchronize()
        if k:
            out.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuse-json", required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    fj = json.loads(Path(a.fuse_json).read_text())
    run, arm, w, top = fj["run"], fj["arm"], float(fj["w"]), fj["top"]
    protocol = run.split("_")[-2]
    logits, canvas, t = tr.load_arm(arm)
    res = dict(run=run, arm=arm, w=w, top=top, thresholds={})
    for split in (protocol + "_test", "cross_test"):
        images = sorted(Path(f).stem for f in seaships_files(split))
        sizes = image_sizes(images)
        boxes, s = tr.box_scores(tr.detector_pred6(run, split), top)
        p_all = tr.type_probs(boxes, logits, canvas, t, a.workers)
        n_img = len(images)
        for thr in THRESHOLDS:
            sel = boxes.score.to_numpy() >= thr
            f = tr.fuse(boxes, s, p_all, w)
            f0 = tr.fuse(boxes, s, p_all, 0.0)
            keep = np.repeat(sel, len(tc.CLASSES))
            f.loc[~keep, "score"] = f0.loc[~keep, "score"]
            r = class_ap(f, images, sizes)
            cell = dict(retyped_per_image=float(sel.sum() / n_img), map50=100 * r["map50"], map=100 * r["map"])
            if split == "cross_test":
                cell["type_ms_per_image"] = time_type_stage(boxes[sel], logits, canvas, t) / n_img if sel.any() else 0.0
            res["thresholds"].setdefault(str(thr), {})[split] = cell
            print(split, thr, {k: round(v, 2) for k, v in cell.items()}, flush=True)
    out = tc.OUT / "rescore" / ("%s__%s__threshold.json" % (run, arm))
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
