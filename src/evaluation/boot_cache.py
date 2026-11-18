"""Cached paired bootstrap for many comparisons on one test split, optionally with a second level over training seeds.
compare_runs.py recomputes every run's AP on every resample for each pair it compares; the resamples depend
only on the split, --cluster, --n and --seed, so the per-run, per-resample APs can be computed once and reused by every
comparison that involves the run.

  cache    for each run, AP (mAP50, mAP50-95) on the full test split and on each of the --n resamples, from the
           per-class-NMS dets compare_runs.py --preds val reads; written to runs/seaships/<run>/eval/
           boot_<split>_<cl|im>_n<n>_s<seed>.npz
  compare  the gap mean(a) - mean(b) with a 95% percentile interval, printed in compare_runs.py's format (so
           one parser reads both). With --seed-boot, every resample of images / recordings also redraws each side's
           runs (its seeds) with replacement: a two-level bootstrap whose interval covers training variance as well as
           test sampling. With three seeds a side this redraw has only ten distinct multisets, so the interval is coarse.
           Add --paired-seeds when both sides come from the same training runs, listed in the same seed order (a module
           added to a detector, a variant of the type stage, an ablation row that shares the detector): each resample
           then draws one set of seed indices for both sides, so the gap keeps its per-seed pairing and the detector's
           own seed-to-seed spread, which both sides share, does not enter the interval (2026-10-24).

Resamples are identical to compare_runs.py's for the same --split, --cluster, --n and --seed (checked against it).

    python src/evaluation/boot_cache.py cache --split wsodd_test --cluster --runs RUN [RUN ...]
    python src/evaluation/boot_cache.py compare --split wsodd_test --cluster --seed-boot --a A0 A1 A2 --b B0 B1 B2
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

SRC_PARTS = [str(Path(__file__).resolve().parents[1] / p) for p in ("data", "models", "evaluation")]
sys.path[:0] = [p for p in SRC_PARTS if p not in sys.path]  # the src/ modules import one another by name
import ship_data  # noqa: E402
from compare_runs import ALL, fmt, groups_of, mean_ap  # noqa: E402
from det_eval import match_images  # noqa: E402
from eval_detector import RUNS, class_dets, image_sizes, seaships_files, seaships_gt  # noqa: E402


def cache_path(run, split, cluster, n, seed):
    return RUNS / run / "eval" / ("boot_%s_%s_n%d_s%d.npz" % (split, "cl" if cluster else "im", n, seed))


def resamples(images, split, cluster, n, seed):
    """The image lists compare_runs.py draws (same RNG calls in the same order)."""
    rng = np.random.default_rng(seed)
    if cluster:
        gid = groups_of(images, split)
        members = {}
        for im in images:
            members.setdefault(gid[im], []).append(im)
        keys = sorted(members)
        return [sorted(im for g in rng.choice(keys, len(keys)) for im in members[g]) for _ in range(n)], len(keys)
    return [sorted(rng.choice(images, len(images))) for _ in range(n)], len(images)


def cmd_cache(a):
    images = sorted(Path(f).stem for f in seaships_files(a.split))
    sizes = image_sizes(images, a.split)
    gt_cls = seaships_gt(images, sizes, classes=True, split=a.split)  # with ignore regions in every class
    samples, ngroups = resamples(images, a.split, a.cluster, a.n, a.seed)
    for run in a.runs:
        out = cache_path(run, a.split, a.cluster, a.n, a.seed)
        if out.exists() and not a.force:
            print("exists", out.name, run, flush=True)
            continue
        t0 = time.time()
        dts = class_dets(RUNS / run / "eval", a.split)
        per = match_images(gt_cls, dts.assign(cls=dts.pred_cls), images, areas=ALL)
        point = np.array(mean_ap(per, images))
        boot = np.array([mean_ap(per, s) for s in samples])  # N x 2
        np.savez_compressed(out, point=point, boot=boot, split=a.split, cluster=a.cluster, n=a.n, seed=a.seed,
                            ngroups=ngroups, nimages=len(images), made=time.time())
        print("cached %s %s: %.1f / %.1f in %.0f s" % (run, a.split, 100 * point[0], 100 * point[1], time.time() - t0),
              flush=True)


def cmd_compare(a):
    def load(r):
        f = cache_path(r, a.split, a.cluster, a.n, a.seed)
        if not f.exists():
            sys.exit("missing cache: %s" % f)
        return np.load(f)

    za, zb = [load(r) for r in a.a], [load(r) for r in a.b]
    pa, pb = np.array([z["point"] for z in za]), np.array([z["point"] for z in zb])  # R x 2
    ba, bb = np.stack([z["boot"] for z in za], 1), np.stack([z["boot"] for z in zb], 1)  # N x R x 2
    ngroups, nimages = int(za[0]["ngroups"]), int(za[0]["nimages"])
    if a.cluster:
        print("cluster bootstrap over %d recordings" % ngroups)
    print("%s, val NMS dets: %d images; %d paired resamples (cached)\n  a = %s\n  b = %s\n  gap = mean(a) - mean(b)"
          % (a.split, nimages, a.n, " ".join(a.a), " ".join(a.b)))
    if a.seed_boot:
        rng = np.random.default_rng(a.seed + 1)
        ia, ib = rng.integers(0, len(a.a), (a.n, len(a.a))), rng.integers(0, len(a.b), (a.n, len(a.b)))
        if getattr(a, "paired_seeds", False):
            if len(a.a) != len(a.b):
                sys.exit("--paired-seeds needs as many runs on each side")
            ib = ia
        rows = np.arange(a.n)[:, None]
        gap = ba[rows, ia].mean(1) - bb[rows, ib].mean(1)
        if getattr(a, "paired_seeds", False):
            print("two-level bootstrap, seeds paired: each resample also redraws %d seed indices with replacement, the "
                  "same for both sides" % len(a.a))
        else:
            print("two-level bootstrap: each resample also redraws %d of side a's and %d of side b's runs with "
                  "replacement" % (len(a.a), len(a.b)))
    else:
        gap = ba.mean(1) - bb.mean(1)
    view = "%d-class" % ship_data.num_classes(a.split)
    for k, name in enumerate(("mAP50", "mAP50-95")):
        lo, hi = np.percentile(gap[:, k], [2.5, 97.5]) * 100
        # three decimals, so that a bound such as +0.004 is not read back as zero by a parser
        print("  %-8s %-8s a %s  b %s  gap %+.2f  95%% CI [%+.3f, %+.3f]  P(gap<=0) %.3f"
              % (view, name, fmt(pa[:, k]), fmt(pb[:, k]), (pa[:, k].mean() - pb[:, k].mean()) * 100, lo, hi,
                 (gap[:, k] <= 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("cache", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--split", required=True)
        p.add_argument("--cluster", action="store_true")
        p.add_argument("--n", type=int, default=1000)
        p.add_argument("--seed", type=int, default=0)
        if name == "cache":
            p.add_argument("--runs", nargs="+", required=True)
            p.add_argument("--force", action="store_true")
        else:
            p.add_argument("--a", nargs="+", required=True)
            p.add_argument("--b", nargs="+", required=True)
            p.add_argument("--seed-boot", action="store_true")
            p.add_argument("--paired-seeds", action="store_true", help="with --seed-boot: same seed draw for a and b")
    a = ap.parse_args()
    cmd_cache(a) if a.cmd == "cache" else cmd_compare(a)


if __name__ == "__main__":
    main()
