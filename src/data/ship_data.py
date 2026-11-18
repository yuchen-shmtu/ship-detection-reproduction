"""Registry of the detection datasets trained and tested in-domain, and the file and GT helpers
the tools share.

A protocol names one dataset and one split of it. Runs are <model>_<protocol>_s<seed>, splits
<protocol>_{train,val,test}:

  seaships  protocols uploader, random, session (build_seaships_splits.py); also cross_test =
            random_test & session_test
  smdplus   SMD-Plus (Kim et al., JMSE 2022): the 51 SMD VIS videos with its relabelled GT, its
            video-level train / test split, val = whole train videos (build_smdplus.py)
  aboships  ABOships, whole days per split (build_aboships_splits.py); seamark and
            miscellaneous boxes are ignore regions
  mcships   McShips lite, its own ImageSets split (build_mcships.py)
  wsodd     WSODD, whole capture sessions per split (build_wsodd.py)

Each data/<dataset>/ holds images/<stem>.jpg, labels/<stem>.txt (YOLO), splits/<split>.txt,
yolo/<protocol>.yaml (class names), coco/<split>.json (for the official DETR code) and, where
the dataset has them, ignore.csv: image, x, y, w, h in pixels of boxes that count as neither
positives nor negatives. data/smd_vis/ (zero-shot SMD) and the zero-shot use of data/aboships/
are handled by eval_detector.py, not here.
"""
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DATASETS = {"seaships": ("uploader", "random", "session"), "smdplus": ("smdplus",), "aboships": ("aboships",),
            "mcships": ("mcships",), "wsodd": ("wsodd",)}
PROTOCOLS = {p: d for d, ps in DATASETS.items() for p in ps}


def protocol_of(x):
    """'session' -> 'session'; 'smdplus_test' -> 'smdplus'; 'cross_test' -> 'session' (a SeaShips split)."""
    if x in PROTOCOLS:
        return x
    p = x.rsplit("_", 1)[0]
    return "session" if p == "cross" else p


def dataset(x):
    return PROTOCOLS[protocol_of(x)]


def root(x):
    return DATA / dataset(x)


def names(x):
    """{class id: name} of a protocol or split."""
    p = protocol_of(x)
    return yaml.safe_load((root(p) / "yolo" / ("%s.yaml" % ("session" if p == "uploader" else p))).read_text())["names"]


def num_classes(x):
    return len(names(x))


def yolo_yaml(protocol):
    return root(protocol) / "yolo" / (protocol + ".yaml")


def coco_file(split):
    return root(split) / "coco" / (split + ".json")


def split_ids(split):
    if split == "cross_test":
        return sorted(set(split_ids("random_test")) & set(split_ids("session_test")))
    return (root(split) / "splits" / (split + ".txt")).read_text().split()


def split_files(split):
    r = root(split)
    return [str(r / "images" / (i + ".jpg")) for i in split_ids(split)]


def image_sizes(images, x):
    """{stem: (h, w)} of a dataset's images."""
    r = root(x)
    return {im: Image.open(r / "images" / (im + ".jpg")).size[::-1] for im in images}


def ignore_regions(x):
    f = root(x) / "ignore.csv"
    return pd.read_csv(f, dtype={"image": str}) if f.exists() else None


def gt(images, sizes, x, classes=False):
    """GT of a dataset's images in pixels, as det_eval wants it: image, cls, x, y, w, h, ignore.

    The YOLO labels, every box cls 0 unless classes, plus the dataset's ignore regions. det_eval
    matches ignore regions within a class, so with classes each region is repeated for every
    class."""
    r = root(x)
    rows = []
    for im in images:
        h, w = sizes[im]
        for ln in (r / "labels" / (im + ".txt")).read_text().splitlines():
            c, cx, cy, bw, bh = map(float, ln.split())
            rows.append((im, int(c) if classes else 0, (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h, False))
    df = pd.DataFrame(rows, columns=["image", "cls", "x", "y", "w", "h", "ignore"])
    ign = ignore_regions(x)
    if ign is not None:
        ign = ign[ign.image.isin(set(images))][["image", "x", "y", "w", "h"]].assign(ignore=True)
        reps = [ign.assign(cls=c) for c in (names(x) if classes else [0])]
        df = pd.concat([df] + [g[df.columns] for g in reps], ignore_index=True)
    return df
