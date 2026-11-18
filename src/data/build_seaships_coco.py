"""COCO-format annotation files of every SeaShips protocol split, for the official RT-DETR code.

Reads data/seaships/{images,labels,splits}/ (build_seaships_splits.py) and writes
data/seaships/coco/<protocol>_<split>.json for random and session train/val/test, plus
cross_test.json (random_test & session_test, as eval_detector.py).

file_name is relative to data/seaships/images/. Boxes are the YOLO labels in pixels of the
1138x640 pool image, as [x, y, w, h] from the top left; area = w * h, iscrowd 0. Category
ids are the YOLO class ids 0..5, so the RT-DETR config takes num_classes 6 and
remap_mscoco_category False. Image ids are the integer SeaShips IDs.

    python src/data/build_seaships_coco.py
"""
import json
import sys
from pathlib import Path

SRC_PARTS = [str(Path(__file__).resolve().parents[1] / p) for p in ("data", "models", "evaluation")]
sys.path[:0] = [p for p in SRC_PARTS if p not in sys.path]  # the src/ modules import one another by name
from eval_detector import NAMES, SEA, image_sizes, seaships_gt, split_ids  # noqa: E402


def coco(ids):
    sizes = image_sizes(ids)
    gt = seaships_gt(ids, sizes, classes=True)
    images = [dict(id=int(i), file_name=i + ".jpg", height=int(sizes[i][0]), width=int(sizes[i][1])) for i in ids]
    anns = [dict(id=k + 1, image_id=int(r.image), category_id=int(r.cls), bbox=[r.x, r.y, r.w, r.h], area=r.w * r.h,
                 iscrowd=0) for k, r in enumerate(gt.itertuples())]
    cats = [dict(id=int(c), name=n) for c, n in NAMES.items()]
    return dict(images=images, annotations=anns, categories=cats)


def main():
    out = SEA / "coco"
    out.mkdir(exist_ok=True)
    splits = {"%s_%s" % (p, s): sorted(split_ids("%s_%s" % (p, s)))
              for p in ("random", "session") for s in ("train", "val", "test")}
    splits["cross_test"] = sorted(set(splits["random_test"]) & set(splits["session_test"]))
    for name, ids in splits.items():
        d = coco(ids)
        (out / (name + ".json")).write_text(json.dumps(d))
        print("%-14s %5d images %6d boxes" % (name, len(d["images"]), len(d["annotations"])))


if __name__ == "__main__":
    main()
