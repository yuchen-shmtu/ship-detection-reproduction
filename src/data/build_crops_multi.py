"""Cut one crop per labelled box of the in-domain datasets other than SeaShips, for the type stage.

For smdplus, aboships, mcships and wsodd (ship_data.py; or the datasets named as arguments), every
box in the YOLO labels of the images in the dataset's train / val / test splits (ignore regions are
not cut) becomes data/crops/<dataset>/<stem>_<k>.jpg, the box plus PAD of its width / height on
each side (at least MIN_PAD px) clipped to the image, at source resolution, as build_crops.py cuts
SeaShips and as type_rescore.py cuts detector boxes. data/crops/<dataset>/crops.csv has one row per
crop: file (relative to data/crops/), dataset, image, box, label (class name), split, x1, y1, x2,
y2, crop_w, crop_h. type_classifier.py reads it for these protocols.
--pad P and --out NAME (2026-10-20) cut the same boxes with another padding into data/NAME/<dataset>/
(file still relative to data/NAME/), e.g. --pad 0.25 --out crops_p25 for crops with more context.

    python src/data/build_crops_multi.py [wsodd ...] [--pad 0.25 --out crops_p25]
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

SRC_PARTS = [str(Path(__file__).resolve().parents[1] / p) for p in ("data", "models", "evaluation")]
sys.path[:0] = [p for p in SRC_PARTS if p not in sys.path]  # the src/ modules import one another by name
import ship_data  # noqa: E402

OUT = ship_data.DATA / "crops"
PAD, MIN_PAD = 0.125, 4
SPLITS = ("train", "val", "test")


def cut(args):
    ds, protocol, split, stem, pad, out = args
    root = ship_data.root(protocol)
    im = Image.open(root / "images" / (stem + ".jpg")).convert("RGB")
    W, H = im.size
    names = ship_data.names(protocol)
    rows = []
    for k, ln in enumerate((root / "labels" / (stem + ".txt")).read_text().splitlines()):
        c, cx, cy, bw, bh = map(float, ln.split())
        x1, y1, x2, y2 = (cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H
        px, py = max(pad * (x2 - x1), MIN_PAD), max(pad * (y2 - y1), MIN_PAD)
        box = (max(0, round(x1 - px)), max(0, round(y1 - py)), min(W, round(x2 + px)), min(H, round(y2 + py)))
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            continue
        rel = "%s/%s_%d.jpg" % (ds, stem, k)
        im.crop(box).save(out / rel, quality=95)
        rows.append(dict(file=rel, dataset=ds, image=stem, box=k, label=names[int(c)], split=split, x1=round(x1, 1),
                         y1=round(y1, 1), x2=round(x2, 1), y2=round(y2, 1), crop_w=box[2] - box[0],
                         crop_h=box[3] - box[1]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=["smdplus", "aboships", "mcships", "wsodd"])
    ap.add_argument("--pad", type=float, default=PAD)
    ap.add_argument("--out", default="crops", help="directory under data/")
    a = ap.parse_args()
    out = ship_data.DATA / a.out
    for ds in a.datasets:
        protocol = ship_data.DATASETS[ds][0]
        (out / ds).mkdir(parents=True, exist_ok=True)
        jobs = [(ds, protocol, s, stem, a.pad, out) for s in SPLITS for stem in ship_data.split_ids("%s_%s" % (protocol, s))]
        with ThreadPoolExecutor(8) as ex:
            rows = [r for rs in ex.map(cut, jobs) for r in rs]
        c = pd.DataFrame(rows)
        c.to_csv(out / ds / "crops.csv", index=False)
        print(ds, len(c), "crops")
        print(c.pivot_table(index="label", columns="split", values="box", aggfunc="size", fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
