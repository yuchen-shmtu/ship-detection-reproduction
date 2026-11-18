"""Cut one crop per labelled ship and write Valen JSONL for second-stage ship-type recognition.

Inputs: data/seaships (src/data/build_seaships_pool.py, src/data/build_seaships_splits.py) and
data/smd_vis (src/data/build_smd.py). Output, data/crops/:

    seaships/<id>_<k>.jpg              box k of pool image <id>, from the 1138x640 pool image
    smd/<frame image>_<k>.jpg          k-th ship box of an smd_vis frame, from 1920x1080
    seaships_<protocol>_<split>.jsonl  protocol in random/session, split in train/val/test
    smd_all.jsonl                      test only; SMD has its own type set, so its question
                                       has a different ID and candidates from SeaShips
    crops.csv                          one row per crop: box in the source image, class,
                                       group, crop size, whether the box touches the border

Each crop is the box plus PAD of its width/height on each side (at least MIN_PAD px),
clipped to the image, at source resolution. group_id is the camera session for SeaShips and
the video for SMD, so records from one continuous recording share a group. Media paths are
relative to the JSONL directory, as Valen expects; assets carry SHA-256.

    python src/data/build_crops.py
"""
import hashlib
import json
import sys
from pathlib import Path

import cv2
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"
POOL = DATA / "seaships"
SMD = DATA / "smd_vis"
OUT = DATA / "crops"
PAD = 0.125
MIN_PAD = 4
PROMPT = "The image is a crop around one vessel from a maritime surveillance video frame."
QUESTIONS = {
    "seaships_type": {
        "type": "choice",
        "instructions": "Which type of vessel is at the centre of the image?",
        "criteria": {
            "bulk cargo carrier": "Bulk cargo carrier",
            "container ship": "Container ship",
            "fishing boat": "Fishing boat",
            "general cargo ship": "General cargo ship",
            "ore carrier": "Ore carrier",
            "passenger ship": "Passenger ship",
        },
    },
    "smd_type": {
        "type": "choice",
        "instructions": "Which type of vessel is at the centre of the image?",
        "criteria": {
            "ferry": "Ferry",
            "vessel/ship": "Vessel/ship",
            "speed boat": "Speed boat",
            "boat": "Boat",
            "kayak": "Kayak",
            "sail boat": "Sail boat",
        },
    },
}


def crop(im, x1, y1, x2, y2):
    H, W = im.shape[:2]
    px, py = max(PAD * (x2 - x1), MIN_PAD), max(PAD * (y2 - y1), MIN_PAD)
    cx1, cy1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    cx2, cy2 = int(min(W, round(x2 + px))), int(min(H, round(y2 + py)))
    touches = x1 <= 1 or y1 <= 1 or x2 >= W - 1 or y2 >= H - 1
    return im[cy1:cy2, cx1:cx2], int(touches)


def save(im, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), im, [cv2.IMWRITE_JPEG_QUALITY, 95])


def seaships_crops():
    names = list(QUESTIONS["seaships_type"]["criteria"])
    session = pd.read_csv(POOL / "groups.csv", dtype={"id": str}).set_index("id").session
    rows = []
    for lbl in sorted((POOL / "labels").glob("*.txt")):
        i = lbl.stem
        im = cv2.imread(str(POOL / "images" / (i + ".jpg")))
        H, W = im.shape[:2]
        for k, ln in enumerate(v for v in lbl.read_text().splitlines() if v.strip()):
            c, x, y, w, h = [float(v) for v in ln.split()]
            x1, y1, x2, y2 = (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H
            patch, touches = crop(im, x1, y1, x2, y2)
            rel = "seaships/%s_%d.jpg" % (i, k)
            save(patch, OUT / rel)
            rows.append(dict(file=rel, dataset="seaships", image=i, box=k, label=names[int(c)], group=session[i],
                             x1=round(x1, 1), y1=round(y1, 1), x2=round(x2, 1), y2=round(y2, 1),
                             area_frac=round(w * h, 6), crop_w=patch.shape[1], crop_h=patch.shape[0],
                             touches_border=touches))
    return pd.DataFrame(rows)


def smd_crops():
    ann = pd.read_csv(SMD / "annotations.csv")
    ann = ann[ann.ship == 1]
    rows = []
    for name, g in ann.groupby("image", sort=True):
        im = cv2.imread(str(SMD / "images" / (name + ".jpg")))
        H, W = im.shape[:2]
        for k, b in enumerate(g.itertuples()):
            patch, touches = crop(im, b.x, b.y, b.x + b.w, b.y + b.h)
            rel = "smd/%s_%d.jpg" % (name, k)
            save(patch, OUT / rel)
            rows.append(dict(file=rel, dataset="smd", image=name, box=k, label=b.name, group=b.video,
                             x1=b.x, y1=b.y, x2=round(b.x + b.w, 1), y2=round(b.y + b.h, 1),
                             area_frac=round(b.w * b.h / (W * H), 6), crop_w=patch.shape[1], crop_h=patch.shape[0],
                             touches_border=touches))
    return pd.DataFrame(rows)


def record(r, qid, digest):
    keys = list(QUESTIONS[qid]["criteria"])
    return {
        "group_id": r.group,
        "request": {
            "state": {"messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": r.file}}]}]},
            "questions": {qid: QUESTIONS[qid]},
        },
        "targets": {qid: {"probabilities": {k: float(k == r.label) for k in keys}}},
        "assets": [{"path": r.file, "sha256": digest}],
        "meta": {"record_id": Path(r.file).stem, "domain": r.dataset, "modality": "image", "language_bucket": "en",
                 "image": r.image, "box": [r.x1, r.y1, r.x2, r.y2], "area_frac": r.area_frac,
                 "touches_border": r.touches_border},
    }


def write_jsonl(path, crops, qid, digests):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in crops.itertuples():
            f.write(json.dumps(record(r, qid, digests[r.file]), ensure_ascii=False) + "\n")


def main():
    for sub in ("seaships", "smd"):
        if (OUT / sub).exists():
            for p in (OUT / sub).iterdir():
                p.unlink()
    crops = pd.concat([seaships_crops(), smd_crops()], ignore_index=True)
    crops.to_csv(OUT / "crops.csv", index=False)
    digests = {f: hashlib.sha256((OUT / f).read_bytes()).hexdigest() for f in crops.file}
    ss = crops[crops.dataset == "seaships"]
    for protocol in ("random", "session"):
        for split in ("train", "val", "test"):
            ids = set((POOL / "splits" / ("%s_%s.txt" % (protocol, split))).read_text().split())
            part = ss[ss.image.isin(ids)]
            write_jsonl(OUT / ("seaships_%s_%s.jsonl" % (protocol, split)), part, "seaships_type", digests)
            print("seaships %-7s %-5s %5d images %6d crops %4d groups"
                  % (protocol, split, len(ids), len(part), part.group.nunique()))
    sm = crops[crops.dataset == "smd"]
    write_jsonl(OUT / "smd_all.jsonl", sm, "smd_type", digests)
    print("smd     all           %5d images %6d crops %4d groups" % (sm.image.nunique(), len(sm), sm.group.nunique()))
    for d, g in crops.groupby("dataset"):
        side = g[["crop_w", "crop_h"]].min(axis=1)
        print("%s crop short side: median %d px, under 32 px %.1f%%, touching border %.1f%%"
              % (d, side.median(), 100 * (side < 32).mean(), 100 * g.touches_border.mean()))
        print(g.label.value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
