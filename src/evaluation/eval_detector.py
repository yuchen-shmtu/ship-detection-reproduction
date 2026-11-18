"""Test one trained detector in-domain, and SeaShips runs also on SMD-VIS (and ABOships) zero-shot.

Reads runs/seaships/<run>/weights/best.pt (from train_detector.py; <run> ends in
_<protocol>_s<seed>) and writes runs/seaships/<run>/eval/ (eval_last/ for --weights last):

    val_<split>/              ultralytics val per SeaShips split (per class, as papers report)
    pred6_<split>_<size>.csv  per-class dets (other runs; ultralytics' are in val_<split>/)
    pred_<set>_<size>.csv     class-agnostic dets (agnostic NMS, conf 0.001, max 300 per image)
    summary.json              every number below

Test sets (the run's dataset comes from its protocol, ship_data.py):
  - <protocol>_test: the protocol's own test split, for every dataset; McShips runs also
    mcships_testdedup (its test without photos repeated in train). For runs on another
    dataset than SeaShips the keys below keep their names ("seaships*" = in-domain test) and
    summary.json records the dataset; the zero-shot sets are SeaShips-only.
  - cross_test = random_test & session_test. Neither model trained on these frames, but
    the random model trained on other frames of the same sessions and the session model
    did not, so the gap between the two models on it is what the leak is worth. Its list
    and yaml go to runs/seaships/_lists/.
  - SMD-VIS, class-agnostic (every ship type is one class), scored by det_eval over COCO
    area ranges in native 1920x1080 pixels, for all frames and onshore / onboard, with two
    ignore sets: "lenient" ignores buoy, swimming person, bird/plane and other; "strict"
    ignores only other (clipped ships, barges, lighthouse), so dets on buoys count as FP.
  - ABOships (data/aboships/, build_aboships.py), class-agnostic in native 1280x720 pixels,
    only with --aboships-imgsz: "lenient" ignores seamark and miscellaneous (birds);
    "strict" only miscellaneous. --aboships-only adds it to an existing summary.json.

summary.json keys:
  seaships           ultralytics val, 6 classes (ultralytics models only)
  seaships_coco      det_eval 6-class COCO AP of the per-class-NMS dets, every model: the
                     metric comparable across ultralytics and torchvision models
  seaships_agnostic  det_eval class-agnostic AP, the in-domain reference for SMD
  smd                <ignore set>_<size>
  aboships           <ignore set>_<size>; aboships_n_images, the frames it was scored on

torchvision models (tv_detector.py) and the RT-DETR-family variants (rtdetr_official.py) are
tested the same way, at their own input size: Faster R-CNN at long side 640 (SMD at each
--smd-imgsz), SSD300 at 300x300 only, RT-DETR-R18 at 640x640 only.

--coco-only adds seaships_coco to an existing summary.json from its saved dets, without
the GPU (for runs evaluated before seaships_coco existed).

    python src/evaluation/eval_detector.py --run yolo11n_session_s0
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

os.environ["YOLO_AUTOINSTALL"] = "False"  # save_json val otherwise pip-installs faster-coco-eval into the environment
SRC_PARTS = [str(Path(__file__).resolve().parents[1] / p) for p in ("data", "models", "evaluation")]
sys.path[:0] = [p for p in SRC_PARTS if p not in sys.path]  # the src/ modules import one another by name
import ship_data  # noqa: E402
import tv_detector  # noqa: E402
from det_eval import evaluate, summary  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SEA = ROOT / "data" / "seaships"
SMD = ROOT / "data" / "smd_vis"
ABO = ROOT / "data" / "aboships"
RUNS = ROOT / "runs" / "seaships"
LISTS = RUNS / "_lists"
IGNORE = {"lenient": [2, 8, 9, 10], "strict": [10]}
ABO_IGNORE = {"lenient": ["seamark", "miscellaneous"], "strict": ["miscellaneous"]}
# SeaShips class names (build_seaships_coco.py); None while data/seaships/ is not built
NAMES = yaml.safe_load((SEA / "yolo" / "session.yaml").read_text())["names"] if (SEA / "yolo").exists() else None


def split_ids(name):
    return ship_data.split_ids(name)


def seaships_yaml(split):
    """yaml whose test is the split: the protocol's own for <protocol>_test, else one written to
    runs/seaships/_lists/ (cross_test, mcships_testdedup)."""
    p = ship_data.protocol_of(split)
    if split == p + "_test":
        return ship_data.yolo_yaml(p)
    LISTS.mkdir(parents=True, exist_ok=True)
    lst = LISTS / (split + ".txt")
    lst.write_text("".join("%s\n" % Path(f).as_posix() for f in ship_data.split_files(split)))
    cfg = yaml.safe_load(ship_data.yolo_yaml(p).read_text())
    cfg.update(path=lst.parent.as_posix(), train=lst.as_posix(), val=lst.as_posix(), test=lst.as_posix())
    out = LISTS / ("cross.yaml" if split == "cross_test" else split + ".yaml")
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def seaships_files(split):
    """Image paths of a split of any dataset (the name is kept from SeaShips-only days)."""
    return ship_data.split_files(split)


def predict(model, files, imgsz, chunk=32):
    """Class-agnostic dets as image, x, y, w, h, score, pred_cls; plus image sizes.

    Fed in chunks: ultralytics decodes a whole list source into memory up front, which for
    the 4,002 SMD frames is ~25 GB.
    """
    rows, sizes = [], {}
    for i in range(0, len(files), chunk):
        for r in model.predict(files[i:i + chunk], imgsz=imgsz, conf=0.001, iou=0.7, max_det=300, agnostic_nms=True,
                               batch=16, device=0, verbose=False):
            im = Path(r.path).stem
            sizes[im] = r.orig_shape  # (h, w)
            xyxy, conf, cls = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()
            for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cls):
                rows.append((im, x1, y1, x2 - x1, y2 - y1, s, int(c)))
    dt = pd.DataFrame(rows, columns=["image", "x", "y", "w", "h", "score", "pred_cls"])
    return dt.assign(cls=0), sizes


def seaships_gt(images, sizes, classes=False, split="session_test"):
    """GT of a split's dataset in pixels (YOLO labels, plus its ignore regions); cls 0 unless classes."""
    return ship_data.gt(images, sizes, split, classes)


def image_sizes(images, split="session_test"):
    return ship_data.image_sizes(images, split)


def class_dets(ev, split):
    """Per-class-NMS dets of a run on a SeaShips split, from its eval dir: pred6_<split>_*.csv
    (torchvision runs) or ultralytics val's predictions.json."""
    csv = sorted(ev.glob("pred6_%s_*.csv" % split))
    if csv:
        return pd.read_csv(csv[0], dtype={"image": str})
    p = json.loads((ev / ("val_" + split) / "predictions.json").read_text())
    return pd.DataFrame([(Path(d["file_name"]).stem, d["category_id"] - 1, *d["bbox"], d["score"]) for d in p],
                        columns=["image", "pred_cls", "x", "y", "w", "h", "score"])


def class_ap(dt, images, sizes, split="session_test"):
    """Per-class COCO AP (det_eval, all areas, max 100 dets per image and class) of per-class dets,
    over the split's dataset's classes (6 for SeaShips)."""
    names = ship_data.names(split)
    res = evaluate(seaships_gt(images, sizes, classes=True, split=split), dt.assign(cls=dt.pred_cls), images=images,
                   areas={"all": (0, 1e10)})
    res = res[res.n_gt > 0]
    ap = res[[c for c in res.columns if c.startswith("ap")]]
    return dict(n_images=len(images), map50=float(res.ap50.mean()), map=float(ap.to_numpy().mean()),
                per_class_ap50={names[int(c)]: float(v) for c, v in zip(res.cls, res.ap50)},
                per_class_map={names[int(c)]: float(v) for c, v in zip(res.cls, ap.mean(1))})


def smd_gt(ignore_types):
    ann = pd.read_csv(SMD / "annotations.csv")
    ann = ann[(ann.ship == 1) | ann.type.isin(ignore_types)]
    return ann.assign(cls=0, ignore=ann.ship == 0)[["image", "cls", "x", "y", "w", "h", "ignore"]]


def aboships_gt(ignore_types):
    ann = pd.read_csv(ABO / "annotations.csv", dtype={"image": str})
    ann = ann[(ann.ship == 1) | ann.type.isin(ignore_types)]
    return ann.assign(cls=0, ignore=ann.ship == 0)[["image", "cls", "x", "y", "w", "h", "ignore"]]


def run_val(model, split, out):
    r = model.val(data=str(seaships_yaml(split)), split="test", imgsz=640, batch=16, workers=2, device=0,
                  save_json=True, plots=False, project=str(out), name="val_" + split, exist_ok=True, verbose=False)
    names = r.names
    return dict(n_images=len(seaships_files(split)), map50=float(r.box.map50), map=float(r.box.map),
                per_class_ap50={names[int(c)]: float(v) for c, v in zip(r.box.ap_class_index, r.box.ap50)},
                per_class_map={names[int(c)]: float(v) for c, v in zip(r.box.ap_class_index, r.box.ap)},
                speed_ms=r.speed)


def coco_only(out):
    res = json.loads((out / "summary.json").read_text())
    res["seaships_coco"] = {}
    for split in res["seaships_agnostic"]:
        images = sorted(Path(f).stem for f in seaships_files(split))
        res["seaships_coco"][split] = class_ap(class_dets(out, split), images, image_sizes(images, split), split)
        print(out.parent.name, split, "coco 6-class mAP50 %.4f" % res["seaships_coco"][split]["map50"], flush=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--weights", default="best",
                    help="best / last (the run's own checkpoints) or a path; non-best output goes to eval_<stem>/")
    ap.add_argument("--smd-imgsz", type=int, nargs="*", default=[640], help="none: skip SMD")
    ap.add_argument("--aboships-imgsz", type=int, nargs="*", default=[], help="none (default): skip ABOships")
    ap.add_argument("--aboships-only", action="store_true", help="add ABOships to an existing summary.json")
    ap.add_argument("--coco-only", action="store_true")
    a = ap.parse_args()

    protocol = a.run.split("_")[-2]
    arch = a.run.rsplit("_", 2)[0]
    weights = RUNS / a.run / "weights" / (a.weights + ".pt") if a.weights in ("best", "last") else Path(a.weights)
    out = RUNS / a.run / ("eval" if a.weights == "best" else "eval_" + weights.stem)
    if a.coco_only:
        return coco_only(out)
    out.mkdir(parents=True, exist_ok=True)
    if a.aboships_only:
        res = json.loads((out / "summary.json").read_text())
    else:
        res = dict(run=a.run, protocol=protocol, dataset=ship_data.dataset(protocol), weights=str(weights),
                   seaships={}, seaships_coco={}, seaships_agnostic={}, smd={})

    import rtdetr_official

    detr = arch in rtdetr_official.VARIANTS
    ultra = arch not in tv_detector.ARCHS and not detr
    if detr:
        model, pred = rtdetr_official.load(arch, weights), rtdetr_official.predict
        sea_size = rtdetr_official.SIZE
    elif not ultra:
        cfg = tv_detector.ARCHS[arch]
        model, pred = tv_detector.load(arch, weights), tv_detector.predict
        sea_size = cfg["size"]
    if not ultra:
        def dets(files, size):  # per-class, class-agnostic, image sizes
            dt6, sizes = pred(model, tv_detector.frames_loader(files), size)
            return dt6, tv_detector.agnostic(dt6), sizes
    else:
        from ultralytics import RTDETR, YOLO  # here: spawned dataloader workers re-import this module

        model = (RTDETR if a.run.startswith("rtdetr") else YOLO)(str(weights))
        sea_size = 640

        def dets(files, size):
            return (None, *predict(model, files, size))

    def test_sizes(req):  # SMD / ABOships sizes: SSD300 and RT-DETR-R18 take only their own
        fixed = detr or (not ultra and not tv_detector.ARCHS[arch]["any_size"])
        return ([sea_size] if req else []) if fixed else req

    sea = ship_data.dataset(protocol) == "seaships"
    extra = ["mcships_testdedup"] if protocol == "mcships" else []
    splits = () if a.aboships_only else (protocol + "_test", "cross_test") if sea else (protocol + "_test", *extra)
    smd_sizes = [] if a.aboships_only or not sea else test_sizes(a.smd_imgsz)
    for split in splits:
        if ultra:
            res["seaships"][split] = run_val(model, split, out)
        dt6, dt, sizes = dets(seaships_files(split), sea_size)
        images = sorted(sizes)
        if dt6 is None:
            dt6 = class_dets(out, split)
        else:
            dt6.to_csv(out / ("pred6_%s_%d.csv" % (split, sea_size)), index=False)
        dt.to_csv(out / ("pred_%s_%d.csv" % (split, sea_size)), index=False)
        res["seaships_coco"][split] = class_ap(dt6, images, sizes, split)
        res["seaships_agnostic"][split] = summary(evaluate(seaships_gt(images, sizes, split=split), dt, images=images))
        print(split, "coco %d-class mAP50 %.4f" % (ship_data.num_classes(split), res["seaships_coco"][split]["map50"]),
              res["seaships_agnostic"][split]["all"], flush=True)

    if smd_sizes:  # the zero-shot SMD set is read only when it is tested
        frames = pd.read_csv(SMD / "frames.csv")
        files = [str(SMD / "images" / (im + ".jpg")) for im in frames.image]
    for imgsz in smd_sizes:
        _, dt, _ = dets(files, imgsz)
        dt.to_csv(out / ("pred_smd_%d.csv" % imgsz), index=False)
        for mode, types in IGNORE.items():
            gt = smd_gt(types)
            cell = {}
            for subset in ("all", "onshore", "onboard"):
                ims = frames.image if subset == "all" else frames.image[frames.subset == subset]
                cell[subset] = summary(evaluate(gt[gt.image.isin(ims)], dt, images=list(ims)))
            res["smd"]["%s_%d" % (mode, imgsz)] = cell
            print("smd", imgsz, mode, cell["all"]["all"], flush=True)

    abo_sizes = test_sizes(a.aboships_imgsz) if sea else []
    if abo_sizes:
        frames = pd.read_csv(ABO / "frames.csv", dtype={"image": str})
        files = [str(ABO / "images" / (im + ".jpg")) for im in frames.image]
        res["aboships_n_images"] = len(files)
        res.setdefault("aboships", {})
    for imgsz in abo_sizes:
        _, dt, _ = dets(files, imgsz)
        dt.to_csv(out / ("pred_aboships_%d.csv" % imgsz), index=False)
        for mode, types in ABO_IGNORE.items():
            cell = summary(evaluate(aboships_gt(types), dt, images=list(frames.image)))
            res["aboships"]["%s_%d" % (mode, imgsz)] = cell
            print("aboships", imgsz, mode, cell["all"], flush=True)

    (out / "summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
