"""Train / val / test splits of ABOships by whole day, for in-domain training.

Reads data/aboships/{frames,annotations}.csv (build_aboships.py: 9,041 labelled frames from
202 videos on 13 days; YOLO labels hold the 9 ship types) and writes into data/aboships/:

    splits/aboships_{train,val,test}.txt   frame stems
    yolo/aboships.yaml, yolo/aboships_{train,val,test}.txt
    coco/aboships_{train,val,test}.json    ship boxes, plus the ignore regions as iscrowd
                                           boxes repeated for every category
    ignore.csv                             seamark and miscellaneous boxes (ship = 0): ignore
                                           regions for det_eval (ship_data.gt), as the zero-shot
                                           "lenient" set

Split: whole days, so frames of one trip (one every 15 s) and of one day never straddle two
splits; the waterbus runs the same route every day, so scenes still recur across days, as a
fixed camera's do in SeaShips. Two test days and two val days are chosen among all pairs: the
test pair whose share of frames is nearest 15% and whose ship-type mix is nearest the whole
set (L1 distance of box shares), then the same for val among the remaining days. Frames whose
only boxes are seamarks or miscellaneous stay in, as negatives with ignore regions.

The ABOships papers split frames at random (Iancu et al. 2021: 3,146 training images;
ABOships-PLUS, JMSE 2023: 70/15/15 of 9,880 images), so adjacent frames of a trip are in
both train and test there.

    python src/data/build_aboships_splits.py
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ABO = Path(__file__).resolve().parents[2] / "data" / "aboships"
TARGET = 0.15


def mix(boxes, frames):
    b = boxes[boxes.image.isin(set(frames.image)) & (boxes.ship == 1)]
    return b.type.value_counts()


def choose(frames, boxes, pool, overall):
    best = None
    n_all = len(frames)
    for pair in itertools.combinations(pool, 2):
        sel = frames[frames.date.isin(pair)]
        m = mix(boxes, sel)
        share = (m / m.sum()).reindex(overall.index, fill_value=0)
        cost = abs(len(sel) / n_all - TARGET) * 4 + np.abs(share - overall).sum()
        if best is None or cost < best[0]:
            best = (cost, pair)
    return list(best[1])


def coco(stems, boxes, names, sizes):
    ids = {s: k + 1 for k, s in enumerate(stems)}
    b = boxes[boxes.image.isin(ids)]
    cat = {n: k for k, n in names.items()}
    anns = []
    for r in b.itertuples():
        if r.ship:
            anns.append((ids[r.image], cat[r.type], [r.x, r.y, r.w, r.h], 0))
        else:  # an ignore region for every category, as det_eval
            anns += [(ids[r.image], c, [r.x, r.y, r.w, r.h], 1) for c in names]
    return dict(images=[dict(id=ids[s], file_name=s + ".jpg", height=sizes[1], width=sizes[0]) for s in stems],
                annotations=[dict(id=k + 1, image_id=i, category_id=c, bbox=bb, area=bb[2] * bb[3], iscrowd=cr)
                             for k, (i, c, bb, cr) in enumerate(anns)],
                categories=[dict(id=k, name=n) for k, n in names.items()])


def main():
    frames = pd.read_csv(ABO / "frames.csv", dtype={"image": str, "date": str, "video": str})
    boxes = pd.read_csv(ABO / "annotations.csv", dtype={"image": str, "video": str})
    names = yaml.safe_load((ABO / "yolo" / "all.yaml").read_text())["names"]
    overall = mix(boxes, frames)
    overall = overall / overall.sum()
    dates = sorted(frames.date.unique())
    test = choose(frames, boxes, dates, overall)
    val = choose(frames, boxes, [d for d in dates if d not in test], overall)
    frames["split"] = np.where(frames.date.isin(test), "test", np.where(frames.date.isin(val), "val", "train"))

    for d in ("splits", "coco"):
        (ABO / d).mkdir(exist_ok=True)
    boxes[boxes.ship == 0][["image", "x", "y", "w", "h", "type"]].to_csv(ABO / "ignore.csv", index=False)
    for split in ("train", "val", "test"):
        stems = frames.image[frames.split == split].tolist()
        (ABO / "splits" / ("aboships_%s.txt" % split)).write_text("".join(s + "\n" for s in stems))
        (ABO / "yolo" / ("aboships_%s.txt" % split)).write_text("".join("./../images/%s.jpg\n" % s for s in stems))
        (ABO / "coco" / ("aboships_%s.json" % split)).write_text(json.dumps(coco(stems, boxes, names, (1280, 720))))
    (ABO / "yolo" / "aboships.yaml").write_text(
        "path: %s\ntrain: aboships_train.txt\nval: aboships_val.txt\ntest: aboships_test.txt\nnames:\n%s"
        % ((ABO / "yolo").as_posix(), "".join("  %d: %s\n" % (k, n) for k, n in names.items())))

    print("test days", test, "val days", val)
    print(frames.groupby("split").agg(days=("date", "nunique"), videos=("video", "nunique"), frames=("image", "size"),
                                      ship_boxes=("n_ship", "sum"), ignore_boxes=("n_ignore", "sum")).to_string())
    b = boxes[boxes.ship == 1].merge(frames[["image", "split"]], on="image")
    print(b.pivot_table(index="type", columns="split", values="x", aggfunc="size", fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
