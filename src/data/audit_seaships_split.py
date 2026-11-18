"""Audit SeaShips splits for near-duplicate leakage between train and val/test.

For each evaluation image (val or test), against the train images of the same protocol:

  time    gap to the nearest train frame from the same camera (OSD timestamps)
  boxes   best train image with the same classes, mean greedy per-class box IoU
  pixels  best cosine similarity of 32x32 grayscale thumbnails

Reads data/seaships/{images,labels,groups.csv,splits/<protocol>_*.txt}.

    python src/data/audit_seaships_split.py uploader random session
"""
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

POOL = Path(__file__).resolve().parents[2] / "data" / "seaships"


def load_boxes(i):
    rows = [[float(v) for v in ln.split()] for ln in (POOL / "labels" / (i + ".txt")).read_text().splitlines()
            if ln.strip()]
    return np.array(rows).reshape(-1, 5)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-12)


def box_match(q, c):
    """Mean IoU of greedy same-class matching; callers only pass equal class multisets."""
    used, total = set(), 0.0
    for bq in q:
        best, bj = 0.0, -1
        for j, bc in enumerate(c):
            if j in used or bc[0] != bq[0]:
                continue
            v = iou(bq[1:], bc[1:])
            if v > best:
                best, bj = v, j
        if bj < 0:
            return 0.0
        used.add(bj)
        total += best
    return total / len(q)


def thumbs(ids):
    out = np.zeros((len(ids), 32 * 32), np.float32)
    for k, i in enumerate(ids):
        g = cv2.imread(str(POOL / "images" / (i + ".jpg")), cv2.IMREAD_GRAYSCALE)
        v = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        v -= v.mean()
        out[k] = v / (np.linalg.norm(v) + 1e-6)
    return out


def audit(train, evals, boxes, thumb, meta):
    by_key = defaultdict(list)
    for i in train:
        by_key[tuple(sorted(boxes[i][:, 0]))].append(i)
    t_train = defaultdict(list)
    for i in train:
        if pd.notna(meta.t[i]):
            t_train[meta.camera[i]].append(meta.t[i])
    t_train = {c: np.sort(np.array(v)) for c, v in t_train.items()}
    tr_thumbs = np.stack([thumb[i] for i in train])
    gaps, bm, nn = [], [], []
    for i in evals:
        cand = by_key.get(tuple(sorted(boxes[i][:, 0])), [])
        bm.append(max((box_match(boxes[i], boxes[c]) for c in cand), default=0.0))
        nn.append(float((tr_thumbs @ thumb[i]).max()))
        ts = t_train.get(meta.camera[i])
        if pd.notna(meta.t[i]) and ts is not None and len(ts):
            j = np.searchsorted(ts, meta.t[i])
            gaps.append(min(abs(meta.t[i] - ts[k]) for k in (j - 1, j) if 0 <= k < len(ts)))
    return np.array(gaps), np.array(bm), np.array(nn)


def main(protocols):
    meta = pd.read_csv(POOL / "groups.csv", dtype={"id": str}).set_index("id")
    meta["t"] = pd.to_datetime(meta.timestamp).astype("int64") // 10 ** 9
    meta.loc[meta.timestamp.isna(), "t"] = np.nan
    ids = meta.index.tolist()
    boxes = {i: load_boxes(i) for i in ids}
    thumb = dict(zip(ids, thumbs(ids)))
    print("%-9s %-5s %5s | %-26s | %-23s | %s" % ("protocol", "eval", "n", "time to nearest train (s)",
                                                  "box match: IoU >= .8/.9", "thumbnail: med, >=.95/.98"))
    for p in protocols:
        split = {s: (POOL / "splits" / ("%s_%s.txt" % (p, s))).read_text().split() for s in ("train", "val", "test")
                 if (POOL / "splits" / ("%s_%s.txt" % (p, s))).exists()}
        for s in ("val", "test"):
            if s not in split:
                continue
            g, bm, nn = audit(split["train"], split[s], boxes, thumb, meta)
            print("%-9s %-5s %5d | med %6d, <=5 %5.1f%%, <=60 %5.1f%% | %5.1f%% / %5.1f%%        | %.3f, %5.1f%% / %5.1f%%"
                  % (p, s, len(split[s]), np.median(g), 100 * (g <= 5).mean(), 100 * (g <= 60).mean(),
                     100 * (bm >= 0.8).mean(), 100 * (bm >= 0.9).mean(),
                     np.median(nn), 100 * (nn >= 0.95).mean(), 100 * (nn >= 0.98).mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["uploader", "random", "session"]))
