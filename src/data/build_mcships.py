"""Build McShips (lite) for in-domain training, with its own train / val / test lists.

McShips (Zheng & Zhang, "McShips: A Large-Scale Ship Dataset For Detection And Fine-Grained
Categorization in the Wild", ICME 2020; github.com/ZhengYitong2333/Mcships). The lite release
(Google Drive file 1udewXbHCS9WKM-MPpWqouUUGs6Vx5iWf = data/_raw/Mcship_database_lite.zip) has
7,996 web photos (mostly ~500 px wide) with Pascal-VOC XML in two classes, civilianship and
warship, and ImageSets/Main/{train,val,test}.txt (4,048 / 1,637 / 2,311).

Writes data/mcships/:

    images/<stem>.jpg     every image re-encoded as RGB JPEG 95 (web photos can be grayscale or
                          CMYK; the torchvision loaders want 3-channel JPEG)
    labels/<stem>.txt     YOLO boxes, 0 civilianship, 1 warship; boxes clipped to the image
    ignore.csv            boxes flagged difficult (VOC's ignore convention), not in the labels
    annotations.csv       every box with its class, difficult flag and clipping
    splits/mcships_{train,val,test}.txt, yolo/mcships.{yaml,_*.txt}, coco/mcships_*.json

The photos are not video frames, so the official random split has no adjacent-frame leak, but
web collections repeat photos: 116 of the 2,311 test images have a train image whose 32x32
grayscale thumbnail (mean-centred, unit norm) has cosine >= 0.98, most of them the same photo
(checked on a sheet; 98 are >= 0.995). They are kept in mcships_test, for comparison with
published numbers, and left out of mcships_testdedup, the test set we also report.

    python src/data/build_mcships.py
"""
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2] / "data"
ZIP = ROOT / "_raw" / "Mcship_database_lite.zip"
OUT = ROOT / "mcships"
PRE = "Mcship_database_lite/"
NAMES = ["civilianship", "warship"]
DUP_COS = 0.98


def thumb(im):
    a = np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR), np.float32).ravel()
    a -= a.mean()
    return a / (np.linalg.norm(a) + 1e-6)


def main():
    for d in ("images", "labels", "yolo", "splits", "coco"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(ZIP)
    splits = {s: z.read(PRE + "ImageSets/Main/%s.txt" % s).decode().split() for s in ("train", "val", "test")}
    rows, sizes, hashes = [], {}, {}
    for s, stems in splits.items():
        for stem in stems:
            im = Image.open(io.BytesIO(z.read(PRE + "JPEGImages/%s.jpg" % stem))).convert("RGB")
            W, H = im.size
            sizes[stem], hashes[stem] = (H, W), thumb(im)
            im.save(OUT / "images" / (stem + ".jpg"), quality=95)
            t = z.read(PRE + "Annotations/%s.xml" % stem).decode("utf-8", "ignore")
            lines = []
            for o in re.findall(r"<object>(.*?)</object>", t, re.S):
                name = re.search(r"<name>([^<]+)</name>", o).group(1).strip()
                diff = bool(re.search(r"<difficult>1</difficult>", o))
                x, y, x2, y2 = (float(re.search("<%s>([^<]+)</%s>" % (k, k), o).group(1))
                                for k in ("xmin", "ymin", "xmax", "ymax"))
                cx1, cy1, cx2, cy2 = max(x, 0), max(y, 0), min(x2, W), min(y2, H)
                if cx2 - cx1 < 2 or cy2 - cy1 < 2 or name not in NAMES:
                    print("dropped", stem, name, (x, y, x2, y2), (W, H))
                    continue
                c = NAMES.index(name)
                rows.append((stem, s, name, int(diff), cx1, cy1, cx2 - cx1, cy2 - cy1,
                             int((cx1, cy1, cx2, cy2) != (x, y, x2, y2))))
                if not diff:
                    lines.append("%d %.6f %.6f %.6f %.6f" % (c, (cx1 + cx2) / 2 / W, (cy1 + cy2) / 2 / H,
                                                             (cx2 - cx1) / W, (cy2 - cy1) / H))
            (OUT / "labels" / (stem + ".txt")).write_text("".join(ln + "\n" for ln in lines))
    ann = pd.DataFrame(rows, columns=["image", "split", "name", "difficult", "x", "y", "w", "h", "clipped"])
    ann.to_csv(OUT / "annotations.csv", index=False)
    ann[ann.difficult == 1][["image", "x", "y", "w", "h"]].to_csv(OUT / "ignore.csv", index=False)

    for s, stems in splits.items():
        (OUT / "splits" / ("mcships_%s.txt" % s)).write_text("".join(i + "\n" for i in stems))
        (OUT / "yolo" / ("mcships_%s.txt" % s)).write_text("".join("./../images/%s.jpg\n" % i for i in stems))
        ids = {st: k + 1 for k, st in enumerate(stems)}
        a = ann[ann.image.isin(ids)]
        anns = []
        for r in a.itertuples():
            cats = range(len(NAMES)) if r.difficult else [NAMES.index(r.name)]
            anns += [(ids[r.image], c, [r.x, r.y, r.w, r.h], r.difficult) for c in cats]
        d = dict(images=[dict(id=ids[st], file_name=st + ".jpg", height=sizes[st][0], width=sizes[st][1])
                         for st in stems],
                 annotations=[dict(id=k + 1, image_id=i, category_id=c, bbox=bb, area=bb[2] * bb[3], iscrowd=cr)
                              for k, (i, c, bb, cr) in enumerate(anns)],
                 categories=[dict(id=k, name=n) for k, n in enumerate(NAMES)])
        (OUT / "coco" / ("mcships_%s.json" % s)).write_text(json.dumps(d))
    (OUT / "yolo" / "mcships.yaml").write_text(
        "path: %s\ntrain: mcships_train.txt\nval: mcships_val.txt\ntest: mcships_test.txt\nnames:\n%s"
        % ((OUT / "yolo").as_posix(), "".join("  %d: %s\n" % (k, n) for k, n in enumerate(NAMES))))

    tr = np.stack([hashes[s] for s in splits["train"]])
    best = np.stack([hashes[s] for s in splits["test"]]) @ tr.T
    dedup = [s for s, b in zip(splits["test"], best.max(1)) if b < DUP_COS]
    (OUT / "splits" / "mcships_testdedup.txt").write_text("".join(i + "\n" for i in dedup))
    print({s: len(v) for s, v in splits.items()}, "images;", len(ann), "boxes;", int(ann.difficult.sum()), "difficult")
    print(ann.pivot_table(index="name", columns="split", values="x", aggfunc="size", fill_value=0).to_string())
    print("test images with a train photo at thumbnail cosine >= %.2f: %d of %d; mcships_testdedup keeps %d"
          % (DUP_COS, len(splits["test"]) - len(dedup), len(splits["test"]), len(dedup)))
    sz = pd.DataFrame(list(sizes.values()), columns=["h", "w"])
    print("image width median %d, height median %d" % (sz.w.median(), sz.h.median()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
