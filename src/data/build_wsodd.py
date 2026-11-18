"""Build WSODD for in-domain training, with a leak-free session split.

WSODD (Zhou et al., "An Image-Based Benchmark Dataset and a Novel Object Detector for Water
Surface Object Detection", Front. Neurorobot. 2021; github.com/sunjiaen/WSODD) has 7,467 USV
photos of lakes, rivers and coast with Pascal-VOC XML in 14 classes (21,911 boxes), distributed
through the authors' Baidu share (README.md); its archive WSODD USV_dataset.zip (1.80 GB) unpacks
to data/wsodd_raw/WSODD USV_dataset/{image,annotation}/.

The photos are video-rate frames: 7,300 come from one IP camera named
192.168.0.64_01_<yyyymmddHHMMSSmmm>, 129 from a Windows camera (WIN_<date>_<time>_Pro), 16 from a
device named <yyyymmddHHMMSS>, and 22 (3.jpg ... 24.jpg, phone photos) carry no time. Consecutive
frames of one camera are a median 1 s apart (88% within 10 s), so a random split puts near copies
in train and test, as on SeaShips. As there, a session is one camera's frames in time order, split
at gaps over SESSION_GAP; the untimed photos are one group. That gives 43 groups on 15 days.
Frames of other sessions of the same day are no more alike than frames of other days (32x32
thumbnail cosine >= 0.98 for 0.9% vs 1.1% of frames), so whole sessions, not days, go to one
split, about 20 / 10 / 70% of the frames (test / val / train), chosen by a local search for the
split whose per-class box shares are closest to those targets, with every class in train and test
(see session_split).

Writes data/wsodd/:

    images/<stem>.jpg     hard links to the raw JPEGs (all RGB, no EXIF rotation); the stem is the
                          file name with '.' replaced by '-' (the IP-camera names hold an IP)
    labels/<stem>.txt     YOLO boxes in NAMES order; boxes clipped to the image
    annotations.csv       every box with its class, truncated flag and clipping
    frames.csv            image, orig (raw name), video (= session; compare_runs --cluster reads
                          it), source, time, day
    splits/wsodd_{train,val,test}.txt, yolo/wsodd.{yaml,_*.txt}, coco/wsodd_*.json

No box is flagged difficult, so there is no ignore.csv. The 380 photos without boxes (empty water,
checked on a sheet) stay in as negatives.

    python src/data/build_wsodd.py
"""
import datetime as dt
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2] / "data"
RAW = ROOT / "wsodd_raw" / "WSODD USV_dataset"
OUT = ROOT / "wsodd"
NAMES = ["boat", "ship", "ball", "bridge", "rock", "harbor", "person", "rubbish", "platform", "mast", "tree",
         "buoy", "grass", "animal"]  # by box count
SESSION_GAP = 600  # seconds, as SeaShips
FRAC = [0.2, 0.1, 0.7]  # test, val, train
RESTARTS = 300
SEED = 0
DUP_COS = 0.98


def when(name):
    """(source, capture time or None) from a raw file name."""
    m = re.match(r"192\.168\.0\.64_01_(\d{14})", name)
    if m:
        return "ipcam", dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    m = re.match(r"WIN_(\d{8})_(\d{2})_(\d{2})_(\d{2})_Pro$", name)
    if m:
        return "wincam", dt.datetime.strptime("".join(m.groups()), "%Y%m%d%H%M%S")
    if re.fullmatch(r"\d{14}", name):
        return "device14", dt.datetime.strptime(name, "%Y%m%d%H%M%S")
    return "untimed", None


def sessions(frames):
    frames = frames.copy()
    frames["video"] = "untimed-000"
    t = frames[frames.time.notna()].sort_values(["source", "time"])
    k = ((t.source != t.source.shift()) | (t.time.diff().dt.total_seconds() > SESSION_GAP)).cumsum() - 1
    frames.loc[t.index, "video"] = ["%s-%03d" % (s, v) for s, v in zip(t.source, k)]
    return frames


def thumb(path):
    with Image.open(path) as im:
        im.draft("L", (160, 120))
        a = np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR), np.float32).ravel()
    a -= a.mean()
    return a / (np.linalg.norm(a) + 1e-6)


def score(assign, sizes, counts, frac):
    """Mean over classes of |share of the class's boxes in the split - the split's target share|,
    summed over splits (classes weigh equally, as in mAP), plus twice the image-share error.
    Every class must be in test (0) and train (2); each class missing from val (1) costs 0.05."""
    total = counts.sum(0)
    s = 0.0
    for k, f in enumerate(frac):
        m = assign == k
        c = counts[m].sum(0)
        missing = int((c == 0).sum())
        if missing and k != 1:
            return np.inf
        s += np.abs(c / total - f).mean() + 2 * abs(sizes[m].sum() / sizes.sum() - f) + 0.05 * missing
    return s


def session_split(groups, frac):
    """Assign whole groups to test (0), val (1), train (2): RESTARTS random starts, each improved by
    moving one group or swapping two until no move lowers the score; the best end state wins.
    Rare classes sit in few sessions (88 of the 94 animal boxes in the largest one), so their
    shares cannot all be near the targets; the score trades them off."""
    names = groups.index.values
    sizes, counts = groups["n"].values, groups[NAMES].values
    rng = np.random.default_rng(SEED)
    best, best_s = None, np.inf
    for _ in range(RESTARTS):
        a = rng.choice(3, size=len(names), p=frac)
        s = score(a, sizes, counts, frac)
        improved = True
        while improved:
            improved = False
            for g in range(len(names)):
                for k in range(3):
                    if k == a[g]:
                        continue
                    b = a.copy()
                    b[g] = k
                    t = score(b, sizes, counts, frac)
                    if t < s:
                        a, s, improved = b, t, True
            for g in range(len(names)):
                for h in range(g + 1, len(names)):
                    if a[g] == a[h]:
                        continue
                    b = a.copy()
                    b[g], b[h] = a[h], a[g]
                    t = score(b, sizes, counts, frac)
                    if t < s:
                        a, s, improved = b, t, True
        if s < best_s:
            best, best_s = a, s
    return dict(zip(names, best)), best_s


def main():
    for d in ("images", "labels", "yolo", "splits", "coco"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    rows, frames = [], []
    for xml in sorted((RAW / "annotation").glob("*.xml")):
        orig, stem = xml.stem, xml.stem.replace(".", "-")
        src_img = RAW / "image" / (orig + ".jpg")
        with Image.open(src_img) as im:
            W, H = im.size
        dst = OUT / "images" / (stem + ".jpg")
        if not dst.exists():
            os.link(src_img, dst)
        lines = []
        for o in ET.parse(xml).getroot().findall("object"):
            name = o.findtext("name").strip()
            b = o.find("bndbox")
            x, y, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            cx1, cy1, cx2, cy2 = max(x, 0), max(y, 0), min(x2, W), min(y2, H)
            if cx2 - cx1 < 2 or cy2 - cy1 < 2 or name not in NAMES:
                print("dropped", orig, name, (x, y, x2, y2), (W, H))
                continue
            rows.append((stem, name, int(o.findtext("truncated") == "1"), cx1, cy1, cx2 - cx1, cy2 - cy1,
                         int((cx1, cy1, cx2, cy2) != (x, y, x2, y2))))
            lines.append("%d %.6f %.6f %.6f %.6f" % (NAMES.index(name), (cx1 + cx2) / 2 / W, (cy1 + cy2) / 2 / H,
                                                     (cx2 - cx1) / W, (cy2 - cy1) / H))
        (OUT / "labels" / (stem + ".txt")).write_text("".join(ln + "\n" for ln in lines))
        src, t = when(orig)
        frames.append((stem, orig, src, t, W, H))
    ann = pd.DataFrame(rows, columns=["image", "name", "truncated", "x", "y", "w", "h", "clipped"])
    fr = sessions(pd.DataFrame(frames, columns=["image", "orig", "source", "time", "width", "height"]))
    fr["day"] = [str(t.date()) if pd.notna(t) else "untimed" for t in fr.time]

    counts = np.zeros((len(fr), len(NAMES)), int)
    idx = {s: k for k, s in enumerate(fr.image)}
    for r in ann.itertuples():
        counts[idx[r.image], NAMES.index(r.name)] += 1
    g = pd.DataFrame(counts, columns=NAMES)
    g["video"], g["n"] = fr.video.values, 1
    groups = g.groupby("video").sum()
    n = len(fr)
    assign, s = session_split(groups, FRAC)
    fr["split"] = fr.video.map(assign).map({0: "test", 1: "val", 2: "train"})
    ann["split"] = ann.image.map(dict(zip(fr.image, fr.split)))
    ann.to_csv(OUT / "annotations.csv", index=False)
    fr[["image", "orig", "video", "source", "time", "day", "split", "width", "height"]].to_csv(OUT / "frames.csv",
                                                                                              index=False)

    for split in ("train", "val", "test"):
        f = fr[fr.split == split].sort_values("image")
        stems = f.image.tolist()
        (OUT / "splits" / ("wsodd_%s.txt" % split)).write_text("".join(i + "\n" for i in stems))
        (OUT / "yolo" / ("wsodd_%s.txt" % split)).write_text("".join("./../images/%s.jpg\n" % i for i in stems))
        ids = {st: k + 1 for k, st in enumerate(stems)}
        a = ann[ann.image.isin(ids)]
        d = dict(images=[dict(id=ids[r.image], file_name=r.image + ".jpg", height=int(r.height), width=int(r.width))
                         for r in f.itertuples()],
                 annotations=[dict(id=k + 1, image_id=ids[r.image], category_id=NAMES.index(r.name),
                                   bbox=[r.x, r.y, r.w, r.h], area=r.w * r.h, iscrowd=0)
                              for k, r in enumerate(a.itertuples())],
                 categories=[dict(id=k, name=nm) for k, nm in enumerate(NAMES)])
        (OUT / "coco" / ("wsodd_%s.json" % split)).write_text(json.dumps(d))
    (OUT / "yolo" / "wsodd.yaml").write_text(
        "path: %s\ntrain: wsodd_train.txt\nval: wsodd_val.txt\ntest: wsodd_test.txt\nnames:\n%s"
        % ((OUT / "yolo").as_posix(), "".join("  %d: %s\n" % (k, nm) for k, nm in enumerate(NAMES))))

    print("%d images, %d boxes, %d groups (gap > %d s) on %d days; split score %.3f"
          % (n, len(ann), fr.video.nunique(), SESSION_GAP, fr.day.nunique(), s))
    print(fr.groupby("split").agg(images=("image", "size"), sessions=("video", "nunique"), days=("day", "nunique"))
          .to_string())
    print(ann.pivot_table(index="name", columns="split", values="x", aggfunc="size", fill_value=0)
          .reindex(NAMES).to_string())
    for a_, b_ in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = set(fr.video[fr.split == a_]) & set(fr.video[fr.split == b_])
        print("sessions shared by %s and %s: %d" % (a_, b_, len(shared)))

    # leakage audit, as audit_seaships_split.py: test / val frames against train
    T = {st: thumb(OUT / "images" / (st + ".jpg")) for st in fr.image}
    tr = fr[fr.split == "train"]
    tr_t = np.stack([T[st] for st in tr.image])
    for split in ("val", "test"):
        f = fr[fr.split == split]
        cos = (np.stack([T[st] for st in f.image]) @ tr_t.T).max(1)
        gaps = []
        for r in f[f.time.notna()].itertuples():
            same = tr[(tr.source == r.source) & tr.time.notna()]
            if len(same):
                gaps.append(np.abs((same.time - r.time).dt.total_seconds()).min())
        gaps = np.array(gaps)
        print("%s vs train: nearest same-camera train frame median %.0f s, %.1f%% within 60 s; "
              "thumbnail cosine >= %.2f: %.1f%%" % (split, np.median(gaps), 100 * np.mean(gaps <= 60), DUP_COS,
                                                    100 * np.mean(cos >= DUP_COS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
