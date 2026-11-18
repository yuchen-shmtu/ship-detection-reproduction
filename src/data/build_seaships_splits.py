"""Build the SeaShips splits used for training and evaluation.

Needs data/seaships/ from build_seaships_pool.py and osd.csv from seaships_osd.py.

Groups. Frames are grouped by camera session: frames of one camera, sorted by the OSD
timestamp, start a new session after a gap longer than SESSION_GAP. HQ101 has no readable
timestamp and is one group; the other frames without one join the session of the most
similar (32x32 thumbnail) timestamped frame of their camera. Written to groups.csv.

Protocols, each written as splits/<protocol>_{train,val,test}.txt (one ID per line):

    random   the uploader's test set; val is a random VAL_FRAC of all frames taken from
             the uploader's train set. Adjacent video frames straddle the splits, as in
             every published SeaShips result.
    session  whole sessions go to one split. Sizes match the random protocol; among
             TRIALS random group orders, the one whose per-split class box proportions
             are closest to the overall ones wins. Every split gets passenger ships.

Also writes yolo/<protocol>.yaml and the image lists they point to, for ultralytics.

    python src/data/build_seaships_splits.py
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

POOL = Path(__file__).resolve().parents[2] / "data" / "seaships"
NAMES = ["bulk cargo carrier", "container ship", "fishing boat", "general cargo ship", "ore carrier",
         "passenger ship"]
SESSION_GAP = 600  # seconds
VAL_FRAC = 0.1
TRIALS = 5000
SEED = 0


def thumb(i):
    g = cv2.imread(str(POOL / "images" / (i + ".jpg")), cv2.IMREAD_GRAYSCALE)
    v = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def sessions(d):
    d = d.copy()
    d["t"] = pd.to_datetime(d.timestamp)
    d["session"] = ""
    for cam, x in d[d.t.notna()].sort_values("t").groupby("camera"):
        k = (x.t.diff().dt.total_seconds() > SESSION_GAP).cumsum()
        d.loc[x.index, "session"] = ["%s-%03d" % (cam, v) for v in k]
    for cam, x in d[d.t.isna()].groupby("camera"):
        known = d[(d.camera == cam) & d.t.notna()]
        if known.empty:
            d.loc[x.index, "session"] = cam + "-000"
            continue
        kt = np.stack([thumb(i) for i in known.id])
        for idx, i in zip(x.index, x.id):
            d.loc[idx, "session"] = known.session.iloc[int(np.argmax(kt @ thumb(i)))]
    return d.drop(columns="t")


def class_counts(ids):
    out = np.zeros((len(ids), len(NAMES)), int)
    for k, i in enumerate(ids):
        for line in (POOL / "labels" / (i + ".txt")).read_text().splitlines():
            if line.strip():
                out[k, int(line.split()[0])] += 1
    return out


def score(assign, sizes, counts, target):
    """L1 distance of each split's class proportions from the overall ones, plus size error."""
    total = counts.sum(0) / counts.sum()
    s = 0.0
    for k, n in enumerate(target):
        c = counts[assign == k].sum(0)
        if c[NAMES.index("passenger ship")] == 0:
            return np.inf
        s += np.abs(c / c.sum() - total).sum() + 2 * abs(sizes[assign == k].sum() - n) / sizes.sum()
    return s


def session_split(groups, target):
    """Assign whole groups to test (0), val (1), train (2)."""
    names = groups.index.values
    sizes, counts = groups["n"].values, groups[NAMES].values
    rng = np.random.default_rng(SEED)
    best, best_s = None, np.inf
    for _ in range(TRIALS):
        assign = np.full(len(names), 2)
        filled = [0, 0]
        for g in rng.permutation(len(names)):
            for k in (0, 1):
                if filled[k] + sizes[g] <= target[k] * 1.02:
                    assign[g], filled[k] = k, filled[k] + sizes[g]
                    break
        s = score(assign, sizes, counts, [target[0], target[1], sizes.sum() - target[0] - target[1]])
        if s < best_s:
            best, best_s = assign, s
    return dict(zip(names, best)), best_s


def write_split(name, parts):
    (POOL / "splits").mkdir(exist_ok=True)
    (POOL / "yolo").mkdir(exist_ok=True)
    for split, ids in parts.items():
        ids = sorted(ids)
        (POOL / "splits" / ("%s_%s.txt" % (name, split))).write_text("\n".join(ids) + "\n")
        # ultralytics resolves "./" against the list's folder and maps /images/ to /labels/.
        (POOL / "yolo" / ("%s_%s.txt" % (name, split))).write_text(
            "".join("./../images/%s.jpg\n" % i for i in ids))
    names = "".join("  %d: %s\n" % (k, n) for k, n in enumerate(NAMES))
    (POOL / "yolo" / ("%s.yaml" % name)).write_text(
        "path: %s\ntrain: %s_train.txt\nval: %s_val.txt\ntest: %s_test.txt\nnames:\n%s"
        % ((POOL / "yolo").as_posix(), name, name, name, names))


def report(name, parts, d, counts):
    idx = {i: k for k, i in enumerate(d.id)}
    print("\n%s" % name)
    print("  %-6s %6s %9s  %s" % ("split", "images", "sessions", "  ".join("%6.6s" % n for n in NAMES)))
    for split in ("train", "val", "test"):
        ids = parts[split]
        c = counts[[idx[i] for i in ids]].sum(0)
        print("  %-6s %6d %9d  %s" % (split, len(ids), d[d.id.isin(ids)].session.nunique(),
                                      "  ".join("%6d" % v for v in c)))
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = set(d[d.id.isin(parts[a])].session) & set(d[d.id.isin(parts[b])].session)
        print("  sessions shared by %s and %s: %d" % (a, b, len(shared)))


def main():
    d = pd.read_csv(POOL / "osd.csv", dtype={"id": str})
    d = sessions(d)
    d[["id", "site", "camera", "timestamp", "session"]].to_csv(POOL / "groups.csv", index=False)
    counts = class_counts(d.id)
    print("%d frames, %d cameras, %d sessions (gap > %d s)" % (len(d), d.camera.nunique(), d.session.nunique(),
                                                              SESSION_GAP))

    up_train = (POOL / "splits" / "uploader_train.txt").read_text().split()
    up_test = (POOL / "splits" / "uploader_test.txt").read_text().split()
    rng = np.random.default_rng(SEED)
    n_val = int(round(VAL_FRAC * len(d)))
    val = set(rng.choice(sorted(up_train), n_val, replace=False))
    random = {"train": [i for i in up_train if i not in val], "val": sorted(val), "test": up_test}
    write_split("random", random)
    report("random", random, d, counts)

    g = pd.DataFrame(counts, columns=NAMES)
    g["session"], g["n"] = d.session.values, 1
    groups = g.groupby("session").sum()
    assign, s = session_split(groups, [len(up_test), n_val])
    which = d.session.map(assign)
    session = {split: d.id[which == k].tolist() for k, split in enumerate(("test", "val", "train"))}
    write_split("session", session)
    report("session (score %.3f)" % s, session, d, counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
