"""Paired image bootstrap of the AP gap between two sets of runs on one SeaShips test split.

Each side is one run or several (e.g. the seeds of one protocol). Every run is scored with
det_eval on the same resample of images, drawn with replacement. The gap is mean(a) -
mean(b), recomputed on each resample. The CI covers test-image sampling of that
seed-averaged gap. Seed spread is shown separately as mean ± std over each side's runs.

--preds agnostic (default) reads runs/seaships/<run>/eval/pred_<split>_<size>.csv
(eval_detector.py, class-agnostic NMS) and reports two views:
  - agnostic: every ship is one class;
  - 6-class: each det keeps its predicted class. With agnostic NMS a ship whose type is
    uncertain keeps one box, so type confusion costs full AP.

--preds val reads the per-class-NMS dets: eval/val_<split>/predictions.json of ultralytics
val, i.e. the protocol papers report, or eval/pred6_<split>_<size>.csv of torchvision runs
(eval_detector.class_dets). Only the 6-class view applies. Per-class NMS keeps a
box per candidate type, so it forgives type confusion that agnostic NMS does not. Points
differ slightly from ultralytics' own mAP, whose AP interpolation differs from COCO's.

Matching runs once per run (det_eval.match_images); each resample only re-pools it.

--cluster resamples whole recordings instead of images: SeaShips sessions (groups.csv), SMD-Plus
and ABOships videos and WSODD sessions (frames.csv); McShips photos are their own groups. Frames of one recording
are not independent, so where a class sits in one or two test videos (SMD-Plus kayak, sail
boat) the image bootstrap's CI is too narrow; the cluster CI is the honest one there. Output
names get the suffix _cluster by convention (the caller names them).

    python src/evaluation/compare_runs.py --a yolo11n_random_s{0,1,2} --b yolo11n_session_s{0,1,2}
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_PARTS = [str(Path(__file__).resolve().parents[1] / p) for p in ("data", "models", "evaluation")]
sys.path[:0] = [p for p in SRC_PARTS if p not in sys.path]  # the src/ modules import one another by name
import ship_data  # noqa: E402
from det_eval import average_precision, match_images  # noqa: E402
from eval_detector import RUNS, class_dets, image_sizes, seaships_files, seaships_gt  # noqa: E402

ALL = {"all": (0, 1e10)}


def groups_of(images, split):
    """image -> recording: SeaShips session, SMD-Plus / ABOships video; else the image itself."""
    ds = ship_data.dataset(split)
    root = ship_data.root(split)
    if ds == "seaships":
        g = pd.read_csv(root / "groups.csv", dtype={"id": str}).set_index("id").session
    elif ds in ("smdplus", "aboships", "wsodd"):
        g = pd.read_csv(root / "frames.csv", dtype={"image": str, "video": str}).set_index("image").video
    else:
        return {im: im for im in images}
    return {im: g[im] for im in images}


def mean_ap(per, images):
    """mAP50 and mAP50-95 over classes with GT; images sorted, repeats allowed."""
    aps = np.array([ap for ap, n in (average_precision([by[im] for im in images if im in by])
                                     for by in per.values()) if n])
    return aps[:, 0].mean(), aps.mean()


def agnostic_preds(run, split):
    (f,) = (RUNS / run / "eval").glob("pred_%s_*.csv" % split)  # one input size per SeaShips split
    return pd.read_csv(f, dtype={"image": str})


def fmt(p):
    p = np.asarray(p) * 100
    return "%.1f" % p[0] if len(p) == 1 else "%.1f ± %.1f (%s)" % (p.mean(), p.std(ddof=1), " ".join("%.1f" % x for x in p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True, help="runs on side a, e.g. the seeds of one protocol")
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--split", default="cross_test")
    ap.add_argument("--preds", choices=["agnostic", "val"], default="agnostic")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cluster", action="store_true", help="resample whole sessions / videos, not images")
    ap.add_argument("--seed-boot", action="store_true",
                    help="two-level bootstrap: on each resample of images / recordings also resample each side's runs "
                         "(seeds) with replacement, so the interval covers training variance too")
    ap.add_argument("--save", help="write the per-resample, per-run APs (N x R x 2 per view) to this .npz")
    a = ap.parse_args()
    runs = a.a + a.b

    images = sorted(Path(f).stem for f in seaships_files(a.split))
    sizes = image_sizes(images, a.split)
    gt = seaships_gt(images, sizes, split=a.split)
    gt_cls = seaships_gt(images, sizes, classes=True, split=a.split)  # with ignore regions in every class
    views = {"agnostic": (gt, lambda d: d),
             "%d-class" % ship_data.num_classes(a.split): (gt_cls, lambda d: d.assign(cls=d.pred_cls))}
    if a.preds == "val":
        del views["agnostic"]
        dts = {r: class_dets(RUNS / r / "eval", a.split) for r in runs}
    else:
        dts = {r: agnostic_preds(r, a.split) for r in runs}

    rng = np.random.default_rng(a.seed)
    if a.cluster:
        gid = groups_of(images, a.split)
        members = {}
        for im in images:
            members.setdefault(gid[im], []).append(im)
        keys = sorted(members)
        samples = [sorted(im for g in rng.choice(keys, len(keys)) for im in members[g]) for _ in range(a.n)]
        print("cluster bootstrap over %d recordings" % len(keys))
    else:
        samples = [sorted(rng.choice(images, len(images))) for _ in range(a.n)]
    print("%s, %s NMS dets: %d images, %d boxes; %d paired resamples\n  a = %s\n  b = %s\n  gap = mean(a) - mean(b)"
          % (a.split, a.preds, len(images), len(gt), a.n, " ".join(a.a), " ".join(a.b)))
    if a.seed_boot:
        print("two-level bootstrap: each resample also redraws %d of side a's and %d of side b's runs with replacement"
              % (len(a.a), len(a.b)))
    na = len(a.a)
    pick_a = rng.integers(0, na, size=(a.n, na))  # one seed redraw per resample, shared by the views
    pick_b = rng.integers(0, len(a.b), size=(a.n, len(a.b)))
    saved = {}
    for view, (g, f) in views.items():
        per = {r: match_images(g, f(dts[r]), images, areas=ALL) for r in runs}
        point = np.array([mean_ap(per[r], images) for r in runs])  # R x 2
        boot = np.array([[mean_ap(per[r], s) for r in runs] for s in samples])  # N x R x 2
        saved[view] = boot
        if a.seed_boot:
            ia = np.arange(a.n)[:, None]
            gap = boot[:, :na][ia, pick_a].mean(1) - boot[:, na:][ia, pick_b].mean(1)
        else:
            gap = boot[:, :na].mean(1) - boot[:, na:].mean(1)
        for k, name in enumerate(("mAP50", "mAP50-95")):
            lo, hi = np.percentile(gap[:, k], [2.5, 97.5]) * 100
            print("  %-8s %-8s a %s  b %s  gap %+.1f  95%% CI [%+.1f, %+.1f]  P(gap<=0) %.3f"
                  % (view, name, fmt(point[:na, k]), fmt(point[na:, k]),
                     (point[:na, k].mean() - point[na:, k].mean()) * 100, lo, hi, (gap[:, k] <= 0).mean()))
    if a.save:
        np.savez_compressed(a.save, runs=np.array(runs), na=na, pick_a=pick_a, pick_b=pick_b,
                            **{"boot_" + v.replace("-", "_"): b for v, b in saved.items()})


if __name__ == "__main__":
    main()
