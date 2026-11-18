"""COCO-style box AP in numpy, with ignore regions and area ranges.

ultralytics val has no ignore regions, so SMD is scored with this. It reproduces pycocotools
COCOeval (iouType "bbox", maxDets 100), with ignore regions passed as iscrowd GT:

  - per image and class, dets are sorted by score (stable) and cut to max_det;
  - in score order, each det takes the highest-IoU unmatched real GT with IoU >= t. Only if
    there is none may it take an ignore region, scored as intersection over the det's own
    area. An ignore region can absorb any number of dets;
  - dets on ignore regions count as neither TP nor FP. For an area range, GT outside the
    range is ignored as well, and so are unmatched dets outside it;
  - AP at each IoU threshold is the mean envelope precision at 101 recall points. A class
    with no non-ignored GT is left out of the mean (COCOeval's -1).

Boxes are x, y, w, h in pixels from the top left.

Checked on 2026-09-29 against pycocotools 2.0.11 COCOeval (GT area = w*h, ignore =
iscrowd), using a YOLO11n checkpoint's dets:
  - SMD-VIS lenient, strict, no-ignore and onboard-only; SeaShips session_test, both
    class-agnostic and per class;
  - AP at all 10 IoU thresholds and all four area ranges came out identical (|diff| = 0);
  - also identical at max_det 10, with 1,460 SMD frames cut.

Matching is per image, so a bootstrap over images can call match_images once and then
average_precision on each resample (compare_runs.py).

    from det_eval import evaluate, summary
"""
import numpy as np
import pandas as pd

IOU_THRS = np.linspace(0.5, 0.95, 10)
REC_THRS = np.linspace(0.0, 1.0, 101)
AREAS = {"all": (0, 1e10), "small": (0, 32 ** 2), "medium": (32 ** 2, 96 ** 2), "large": (96 ** 2, 1e10)}


def box_iou(d, g, crowd):
    """D x G IoU; for crowd (ignore) columns, intersection over the det's area."""
    ix = np.minimum(d[:, None, 0] + d[:, None, 2], g[None, :, 0] + g[None, :, 2]) - np.maximum(d[:, None, 0], g[None, :, 0])
    iy = np.minimum(d[:, None, 1] + d[:, None, 3], g[None, :, 1] + g[None, :, 3]) - np.maximum(d[:, None, 1], g[None, :, 1])
    inter = np.clip(ix, 0, None) * np.clip(iy, 0, None)
    da = (d[:, 2] * d[:, 3])[:, None]
    union = np.where(crowd[None, :], da, da + (g[:, 2] * g[:, 3])[None, :] - inter)
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match(iou, g_ig, crowd):
    """COCOeval.evaluateImg matching for dets in score order. Returns matched, ignored (T x D)."""
    n_d, n_g = iou.shape
    matched = np.zeros((len(IOU_THRS), n_d), bool)
    ignored = np.zeros((len(IOU_THRS), n_d), bool)
    if n_g == 0:
        return matched, ignored
    for ti, t in enumerate(IOU_THRS):
        taken = np.zeros(n_g, bool)
        for d in np.nonzero(iou.max(1) >= t)[0]:
            ok = (~taken | crowd) & (iou[d] >= t)
            pick = ok & ~g_ig if (ok & ~g_ig).any() else ok
            if not pick.any():
                continue
            idx = np.nonzero(pick)[0]
            m = idx[len(idx) - 1 - np.argmax(iou[d, idx][::-1])]  # ties go to the later GT, as in COCOeval
            matched[ti, d], ignored[ti, d], taken[m] = True, g_ig[m], True
    return matched, ignored


def match_images(gt, dt, images=None, max_det=100, areas=AREAS):
    """Per-image matching. Returns {(cls, area): {image: (scores, matched, ignored, n_gt)}}.

    matched and ignored are T x D over the image's dets in score order; n_gt counts
    non-ignored GT. An image with neither GT nor dets of a class has no entry. images
    defaults to those in gt; dets on other images are dropped.
    """
    gt_by, dt_by = dict(tuple(gt.groupby(["image", "cls"], sort=False))), dict(tuple(dt.groupby(["image", "cls"], sort=False)))
    images = sorted(set(gt["image"]) if images is None else set(images))
    out = {}
    for c in sorted(set(gt["cls"]) | set(dt["cls"])):
        per_area = {a: {} for a in areas}
        for im in images:
            g, d = gt_by.get((im, c)), dt_by.get((im, c))
            if g is None and d is None:
                continue
            gb = g[["x", "y", "w", "h"]].to_numpy(float) if g is not None else np.zeros((0, 4))
            crowd = g["ignore"].to_numpy(bool) if g is not None else np.zeros(0, bool)
            if d is not None:
                order = np.argsort(-d["score"].to_numpy(float), kind="mergesort")[:max_det]
                db, ds = d[["x", "y", "w", "h"]].to_numpy(float)[order], d["score"].to_numpy(float)[order]
            else:
                db, ds = np.zeros((0, 4)), np.zeros(0)
            iou = box_iou(db, gb, crowd)
            for a, (lo, hi) in areas.items():
                ga = gb[:, 2] * gb[:, 3]
                g_ig = crowd | (ga < lo) | (ga > hi)
                go = np.argsort(g_ig, kind="mergesort")
                m, ig = match(iou[:, go], g_ig[go], crowd[go])
                da = db[:, 2] * db[:, 3]
                ig |= ~m & ((da < lo) | (da > hi))[None, :]
                per_area[a][im] = (ds, m, ig, int((~g_ig).sum()))
        out.update({(c, a): p for a, p in per_area.items()})
    return out


def average_precision(parts):
    """AP at each IoU threshold from per-image parts, pooled in the order given.

    Returns (ap array of len(IOU_THRS), n_gt); ap is NaN when n_gt == 0.
    """
    n_pos = sum(p[3] for p in parts)
    if n_pos == 0:
        return np.full(len(IOU_THRS), np.nan), 0
    scores = np.concatenate([p[0] for p in parts])
    order = np.argsort(-scores, kind="mergesort")
    m = np.concatenate([p[1] for p in parts], 1)[:, order]
    ig = np.concatenate([p[2] for p in parts], 1)[:, order]
    tps, fps = np.cumsum(m & ~ig, 1).astype(float), np.cumsum(~m & ~ig, 1).astype(float)
    ap = np.zeros(len(IOU_THRS))
    for ti, (tp, fp) in enumerate(zip(tps, fps)):
        q = np.zeros(len(REC_THRS))
        if len(tp):
            rc, pr = tp / n_pos, tp / (tp + fp + np.spacing(1))
            pr = np.maximum.accumulate(pr[::-1])[::-1]
            idx = np.searchsorted(rc, REC_THRS, side="left")
            q[idx < len(pr)] = pr[idx[idx < len(pr)]]
        ap[ti] = q.mean()
    return ap, n_pos


def evaluate(gt, dt, images=None, max_det=100, areas=AREAS):
    """gt: image, cls, x, y, w, h, ignore. dt: image, cls, x, y, w, h, score.

    Returns one row per (cls, area): n_gt (non-ignored) and ap50 ... ap95 (NaN if n_gt == 0).
    images defaults to those in gt; dets on other images are dropped. An image listed twice
    counts twice (bootstrap resamples).
    """
    images = sorted(gt["image"].unique() if images is None else images)
    rows = []
    for (c, a), by_im in match_images(gt, dt, images, max_det, areas).items():
        ap, n = average_precision([by_im[im] for im in images if im in by_im])
        rows.append(dict(cls=c, area=a, n_gt=n, **{"ap%d" % round(t * 100): v for t, v in zip(IOU_THRS, ap)}))
    return pd.DataFrame(rows)


def summary(res):
    """Per area range: mAP50 and mAP50-95 over classes with GT, as COCOeval.summarize."""
    ap_cols = [c for c in res.columns if c.startswith("ap")]
    out = {}
    for a, r in res.groupby("area", sort=False):
        r = r[r.n_gt > 0]
        out[a] = dict(n_gt=int(r.n_gt.sum()), map50=float(r.ap50.mean()) if len(r) else float("nan"),
                      map=float(r[ap_cols].to_numpy().mean()) if len(r) else float("nan"))
    return out
