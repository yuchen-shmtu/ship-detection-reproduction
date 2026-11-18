"""Type recognition before and after the type stage, on the boxes a detector finds (2026-10-19).

For a detector run and its fused run (type_rescore.py --save-as), on a protocol's test split:
the detector's per-class dets are grouped into boxes as type_rescore.box_scores does (top 100 per
image); the boxes whose best detector score is >= --thr are matched to the GT class-agnostically
(greedy in score order, IoU >= 0.5, each GT box once; a box on an ignore region is dropped); each
matched box's class is read off the detector (argmax of its class scores) and off the fused run
(argmax of the fused scores of the same box). Both confusion matrices therefore count the same
boxes and differ only in their types. GT boxes left unmatched are "missed": the type stage moves
no box, so it cannot find them. Unmatched boxes above --thr are false positives ("background").

Per protocol, summed over the seeds: confusion counts (GT class x predicted class, plus missed
and the background row), type accuracy on the matched boxes (micro, and the mean over classes),
the boxes the type stage fixes and breaks, in runs/types/confusion/<before>__<after>_<protocol>.json;
a figure of both matrices, rows normalised over each GT class's boxes
(runs/types/confusion/figures/confusion_<protocol>.png); and for seed 0 a sheet of examples drawn at random with a
fixed seed (not picked): fixed, broken, wrong before and after, missed
(runs/types/confusion/figures/cases_<protocol>.jpg). Reads saved predictions only (CPU, no torch).

    python src/evaluation/type_confusion.py --before deim-r18-nomix-wise090 --after deim-r18-nomix-wise090-fw05
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

SRC_PARTS = [str(Path(__file__).resolve().parents[1] / p) for p in ("data", "models", "evaluation")]
sys.path[:0] = [p for p in SRC_PARTS if p not in sys.path]  # the src/ modules import one another by name
import ship_data  # noqa: E402
from det_eval import box_iou  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "seaships"
OUT = ROOT / "runs" / "types" / "confusion"
FIG = OUT / "figures"
KEY = ["image", "x", "y", "w", "h"]
FONT = "C:/Windows/Fonts/arial.ttf"


def class_dets(run, split):
    """Per-class dets of a run, as eval_detector.class_dets (pred6 CSV, else ultralytics' JSON)."""
    ev = RUNS / run / "eval"
    csv = sorted(ev.glob("pred6_%s_*.csv" % split))
    if csv:
        return pd.read_csv(csv[0], dtype={"image": str})
    p = json.loads((ev / ("val_" + split) / "predictions.json").read_text())
    return pd.DataFrame([(Path(d["file_name"]).stem, d["category_id"] - 1, *d["bbox"], d["score"]) for d in p],
                        columns=["image", "pred_cls", "x", "y", "w", "h", "score"])


def boxes_of(dt6, k, top=100):
    """One row per box (image, x, y, w, h) with its class scores s0..s<k-1>, best score and class."""
    dt6 = dt6.assign(**{c: dt6[c].round(3) for c in ("x", "y", "w", "h")})
    s = dt6.pivot_table(index=KEY, columns="pred_cls", values="score", aggfunc="max",
                        fill_value=0.0).reindex(columns=range(k), fill_value=0.0)
    b = s.index.to_frame(index=False)
    b["score"], b["cls"] = s.to_numpy().max(1), s.to_numpy().argmax(1)
    keep = b.sort_values("score", ascending=False).groupby("image").head(top).index
    return b.loc[keep].reset_index(drop=True)


def match(boxes, gt, thr, iou_thr=0.5):
    """Greedy class-agnostic matching per image. Returns the boxes with gt_cls (-1 = false
    positive, -2 = on an ignore region) and gt_index, and the GT with matched (bool)."""
    boxes = boxes[boxes.score >= thr].sort_values("score", ascending=False)
    gt = gt.reset_index(drop=True)
    gt_cls = np.full(len(boxes), -1)
    gt_idx = np.full(len(boxes), -1)
    taken = np.zeros(len(gt), bool)
    pos = boxes.groupby("image", sort=False).indices  # positions, in score order within an image
    for im, g in gt.groupby("image"):
        if im not in pos:
            continue
        rows = pos[im]
        ig = g.ignore.to_numpy(bool)
        iou = box_iou(boxes.iloc[rows][["x", "y", "w", "h"]].to_numpy(float), g[["x", "y", "w", "h"]].to_numpy(float), ig)
        gi = g.index.to_numpy()
        for r, row in zip(rows, iou):
            ok = (~ig) & (~taken[gi]) & (row >= iou_thr)
            if ok.any():
                j = np.flatnonzero(ok)[row[ok].argmax()]
                taken[gi[j]] = True
                gt_cls[r], gt_idx[r] = g.cls.iloc[j], gi[j]
            elif (ig & (row >= iou_thr)).any():
                gt_cls[r] = -2
    return boxes.assign(gt_cls=gt_cls, gt_idx=gt_idx), gt.assign(matched=taken)


def analyse(before, after, protocol, seed, thr):
    split = protocol + "_test"
    names = ship_data.names(split)
    k = len(names)
    images = sorted(ship_data.split_ids(split))
    sizes = ship_data.image_sizes(images, split)
    gt = ship_data.gt(images, sizes, split, classes=True)
    gt = pd.concat([gt[~gt.ignore], gt[gt.ignore].drop_duplicates(KEY)], ignore_index=True)  # one copy per region
    b = boxes_of(class_dets("%s_%s_s%d" % (before, protocol, seed), split), k)
    a = boxes_of(class_dets("%s_%s_s%d" % (after, protocol, seed), split), k, top=10 ** 9)
    b = b.merge(a[KEY + ["cls", "score"]].rename(columns={"cls": "after_cls", "score": "after_score"}), on=KEY, how="left")
    lost = int(b.after_cls.isna().sum())  # boxes the fused run dropped (all fused scores < 1e-4)
    b["after_cls"] = b.after_cls.fillna(-1).astype(int)
    m, g = match(b, gt, thr)
    return dict(split=split, names=names, k=k, boxes=m, gt=g[~g.ignore], lost=lost)


def counts(res):
    k = res["k"]
    m, g = res["boxes"], res["gt"]
    mt = m[m.gt_cls >= 0]
    cm = {}
    for tag, col in (("before", "cls"), ("after", "after_cls")):
        c = np.zeros((k + 1, k + 1), int)  # rows: GT classes + background; cols: classes + missed
        np.add.at(c, (mt.gt_cls.to_numpy(), mt[col].clip(0).to_numpy()), 1)
        fp = m[m.gt_cls == -1]
        np.add.at(c, (np.full(len(fp), k), fp[col].clip(0).to_numpy()), 1)
        miss = g[~g.matched].cls.astype(int).to_numpy()
        np.add.at(c, (miss, np.full(len(miss), k)), 1)
        cm[tag] = c
    ok_b, ok_a = mt.cls.to_numpy() == mt.gt_cls.to_numpy(), mt.after_cls.to_numpy() == mt.gt_cls.to_numpy()
    return cm, dict(matched=int(len(mt)), missed=int((~g.matched).sum()), false_pos=int((m.gt_cls == -1).sum()),
                    fixed=int((~ok_b & ok_a).sum()), broken=int((ok_b & ~ok_a).sum()),
                    still_wrong=int((~ok_b & ~ok_a).sum()), lost_boxes=res["lost"])


def accuracy(c, k):
    d = c[:k, :k]
    n = d.sum(1)
    rec = np.divide(np.diag(d), n, out=np.full(k, np.nan), where=n > 0)
    return float(np.trace(d) / max(d.sum(), 1)), float(np.nanmean(rec)), rec


def plot(cm, names, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = len(names)
    labels = [names[i] for i in range(k)]
    fig, axes = plt.subplots(1, 2, figsize=(4 + 0.62 * (k + 1) * 2, 2.2 + 0.5 * k), squeeze=False)
    for ax, tag, head in zip(axes[0], ("before", "after"), ("detector", "detector + type stage")):
        c = cm[tag][:k].astype(float)
        r = c / np.maximum(c.sum(1, keepdims=True), 1)
        ax.imshow(r, cmap="Blues", vmin=0, vmax=1)
        acc, macc, _ = accuracy(cm[tag], k)
        ax.set_title("%s\ntype accuracy %.1f%% (class mean %.1f%%)" % (head, 100 * acc, 100 * macc), fontsize=10)
        ax.set_xticks(range(k + 1))
        ax.set_xticklabels(labels + ["missed"], rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(k))
        ax.set_yticklabels(["%s (%d)" % (labels[i], c[i].sum()) for i in range(k)], fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("ground truth (boxes)")
        for i in range(k):
            for j in range(k + 1):
                if r[i, j] >= 0.005:
                    ax.text(j, i, "%d" % round(100 * r[i, j]), ha="center", va="center", fontsize=7,
                            color="white" if r[i, j] > 0.6 else "black")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def groups_of(images, split):
    """image -> recording, as compare_runs.groups_of (SeaShips session, video; else the image)."""
    ds, root = ship_data.dataset(split), ship_data.root(split)
    if ds == "seaships":
        g = pd.read_csv(root / "groups.csv", dtype={"id": str}).set_index("id").session
    elif ds in ("smdplus", "aboships", "wsodd"):
        g = pd.read_csv(root / "frames.csv", dtype={"image": str, "video": str}).set_index("image").video
    else:
        return {im: im for im in images}
    return {im: g[im] for im in images}


def tile(im, row, label_lines, font, h=120, max_w=330, pad=0.25):
    x, y, w, hh = row.x, row.y, row.w, row.h
    px, py = max(pad * w, 8), max(pad * hh, 8)
    box = (max(0, x - px), max(0, y - py), min(im.width, x + w + px), min(im.height, y + hh + py))
    crop = im.crop(tuple(round(v) for v in box)).convert("RGB")
    sc = min(h / crop.height, max_w / crop.width)
    crop = crop.resize((max(1, round(crop.width * sc)), max(1, round(crop.height * sc))))
    ImageDraw.Draw(crop).rectangle([sc * (x - box[0]), sc * (y - box[1]), sc * (x - box[0] + w), sc * (y - box[1] + hh)],
                                   outline=(255, 40, 40), width=2)
    out = Image.new("RGB", (max_w, h + 16 * len(label_lines) + 4), "white")
    out.paste(crop, ((max_w - crop.width) // 2, 0))
    d = ImageDraw.Draw(out)
    for i, (txt, col) in enumerate(label_lines):
        d.text((4, h + 2 + 16 * i), txt, fill=col, font=font)
    return out


def cases(res, path, per=6, rng=0):
    names, root = res["names"], ship_data.root(res["split"])
    m, g = res["boxes"], res["gt"]
    mt = m[m.gt_cls >= 0]
    ok_b, ok_a = mt.cls == mt.gt_cls, mt.after_cls == mt.gt_cls
    cats = [("fixed by the type stage", mt[~ok_b & ok_a]), ("broken by the type stage", mt[ok_b & ~ok_a]),
            ("wrong before and after", mt[~ok_b & ~ok_a]), ("missed (no box)", g[~g.matched])]
    r = np.random.default_rng(rng)
    font = ImageFont.truetype(FONT, 13)
    head = ImageFont.truetype(FONT, 16)
    rows = []
    for title, df in cats:
        # random order (fixed seed), then at most one box per recording, so that consecutive frames
        # of one ship do not fill the row
        df = df.iloc[r.permutation(len(df))] if len(df) else df
        rec = df.image.map(groups_of(df.image.unique(), res["split"]))
        pick = df[~rec.duplicated()].head(per)
        tiles = []
        for row in pick.itertuples():
            im = Image.open(root / "images" / (row.image + ".jpg"))
            if title.startswith("missed"):
                lines = [("GT: %s" % names[int(row.cls)], (0, 0, 0)), ("no detection", (150, 0, 0))]
            else:
                lines = [("GT: %s" % names[int(row.gt_cls)], (0, 0, 0)),
                         ("det: %s %.2f" % (names[int(row.cls)], row.score), (0, 110, 0) if row.cls == row.gt_cls else (180, 0, 0)),
                         ("+type: %s" % names[int(row.after_cls)], (0, 110, 0) if row.after_cls == row.gt_cls else (180, 0, 0))]
            tiles.append(tile(im, row, lines, font))
        th = max([t.height for t in tiles], default=40)
        band = Image.new("RGB", (330 * per, th + 26), "white")
        ImageDraw.Draw(band).text((6, 4), "%s (%d boxes, one per recording shown)" % (title, len(df)), fill=(0, 0, 0),
                                  font=head)
        for i, t in enumerate(tiles):
            band.paste(t, (330 * i, 26))
        rows.append(band)
    sheet = Image.new("RGB", (330 * per, sum(b.height for b in rows)), "white")
    y = 0
    for b in rows:
        sheet.paste(b, (0, y))
        y += b.height
    sheet.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="detector variant, e.g. deim-r18-nomix-wise090")
    ap.add_argument("--after", required=True, help="its fused variant, e.g. deim-r18-nomix-wise090-fw05")
    ap.add_argument("--protocols", nargs="+", default=["session", "smdplus", "aboships", "mcships", "wsodd"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--thr", type=float, default=0.25, help="detector score a box needs to count")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    for p in a.protocols:
        total, stats, first = None, [], None
        for s in a.seeds:
            if not all((RUNS / ("%s_%s_s%d" % (v, p, s)) / "eval").exists() for v in (a.before, a.after)):
                print("%s s%d: runs missing, skipped" % (p, s), flush=True)
                continue
            res = analyse(a.before, a.after, p, s, a.thr)
            cm, st = counts(res)
            first = first or res
            total = cm if total is None else {t: total[t] + cm[t] for t in cm}
            stats.append(dict(seed=s, **st))
        if total is None:
            continue
        k, names = first["k"], first["names"]
        acc_b, macc_b, rec_b = accuracy(total["before"], k)
        acc_a, macc_a, rec_a = accuracy(total["after"], k)
        summ = {key: int(sum(x[key] for x in stats)) for key in stats[0] if key != "seed"}
        out = dict(before=a.before, after=a.after, protocol=p, seeds=[x["seed"] for x in stats], thr=a.thr,
                   classes=[names[i] for i in range(k)], type_acc=dict(before=acc_b, after=acc_a),
                   type_acc_class_mean=dict(before=macc_b, after=macc_a),
                   class_recall=dict(before=dict(zip([names[i] for i in range(k)], rec_b.tolist())),
                                     after=dict(zip([names[i] for i in range(k)], rec_a.tolist()))),
                   totals=summ, per_seed=stats, confusion={t: total[t].tolist() for t in total})
        (OUT / ("%s__%s_%s.json" % (a.before, a.after, p))).write_text(json.dumps(out, indent=1))
        print("%-9s matched %6d  type acc %.1f -> %.1f (class mean %.1f -> %.1f)  fixed %d broken %d still wrong %d"
              "  missed %d  fp %d  lost %d" % (p, summ["matched"], 100 * acc_b, 100 * acc_a, 100 * macc_b, 100 * macc_a,
                                               summ["fixed"], summ["broken"], summ["still_wrong"], summ["missed"],
                                               summ["false_pos"], summ["lost_boxes"]), flush=True)
        if not a.no_figures:
            plot(total, names, "%s, test, %d seeds: boxes with detector score >= %.2f matched at IoU 0.5"
                 % (p, len(stats), a.thr), FIG / ("confusion_%s.png" % p))
            cases(first, FIG / ("cases_%s.jpg" % p))


if __name__ == "__main__":
    main()
