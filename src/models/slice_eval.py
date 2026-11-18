"""Sliced inference for the RT-DETR-family runs: the full frame plus a grid of overlapping tiles,
each warped to the model's 640x640 input, merged into one set of per-class dets.

Small ships dominate SMD-Plus and ABOships (~80% of boxes under 1% of the frame), and the
640x640 warp of a 1920x1080 frame shrinks them 3x horizontally. Tiles that each cover FRAC of
the frame's width and height (a GRID x GRID grid, adjacent tiles overlapping) show them FRAC^-1
times larger. The rule is fixed for every dataset, model and seed (default 2 x 2 tiles at 0.6:
1.67x zoom, 20% overlap, 5 forward passes per image); it is not tuned on any test split.

Per image and source (full frame or tile), every query of the model's last decoder layer gives
a box and a score per class (sigmoid of its logits, as the official postprocessor). Merging:
  - tile queries whose box touches an interior tile edge (within EDGE of the tile size) are
    dropped: they are cut objects, which a neighbouring tile or the full frame sees whole;
  - each source keeps its TOPQ queries by best class score;
  - greedily by best score, a query absorbs every query of another source with IoU >= IOU and
    takes the element-wise maximum of their class scores (queries of one source are not merged,
    as the model's own output is not);
  - the top MAX_PAIRS (query, class) pairs of the merged queries form the per-class dets, the
    same budget as the official postprocessor's top 300, so sliced and plain runs output equally
    many dets.
With --grid 0 only the full frame is used, which reproduces rtdetr_official.predict exactly
(checked on CPU, 2026-10-16). --size S warps every source to S x S instead of 640 x 640 (the encoder's
position embeddings and the decoder's anchors are then built for the input, as in training, whose
multi-scale range is 480-800); with --grid 0 that is plain inference at a larger input (2026-10-20).

For a run <variant>_<protocol>_s<seed> this writes the run <variant>-sl<grid>_<protocol>_s<seed>
(-r<size> appended for another input size; <variant>-r<size>_... with --grid 0) with eval/pred6_<split>_<size>.csv (per-class dets), eval/pred_<split>_<size>.csv (class-agnostic, NMS
0.7 as eval_detector) and eval/summary.json (seaships_coco and seaships_agnostic per test split,
the slicing settings and the ms per image of the sliced forward + merge), for the protocol's val
and test splits and the test subsets cross_test (SeaShips) / mcships_testdedup (McShips). So
tabulate_runs.py, compare_runs.py and type_rescore.py --fuse treat it like any run.

    python src/models/slice_eval.py --run deim-r18-nomix-wise090_smdplus_s0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SRC_PARTS = [str(Path(__file__).resolve().parents[1] / p) for p in ("data", "models", "evaluation")]
sys.path[:0] = [p for p in SRC_PARTS if p not in sys.path]  # the src/ modules import one another by name
import rtdetr_official  # noqa: E402
import ship_data  # noqa: E402
import tv_detector  # noqa: E402
from det_eval import evaluate, summary  # noqa: E402
from eval_detector import RUNS, class_ap, seaships_gt  # noqa: E402

SIZE = rtdetr_official.SIZE
SUBSETS = {"session_test": ["cross_test"], "mcships_test": ["mcships_testdedup"]}


def tiles(W, H, grid, frac):
    """(x1, y1, x2, y2) of a grid x grid set of tiles, each frac of the frame, evenly spread."""
    if grid < 1:
        return []
    tw, th = round(W * frac), round(H * frac)
    xs = [round(i * (W - tw) / (grid - 1)) for i in range(grid)] if grid > 1 else [(W - tw) // 2]
    ys = [round(j * (H - th) / (grid - 1)) for j in range(grid)] if grid > 1 else [(H - th) // 2]
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def box_iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]).clip(0) * (x[:, 3] - x[:, 1]).clip(0)  # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def merge(boxes, scores, src, iou_thr):
    """Greedy cross-source merge; boxes N x 4 xyxy, scores N x C, src N. Returns kept boxes, scores."""
    order = np.argsort(-scores.max(1), kind="stable")
    b, s, src = boxes[order], scores[order].copy(), src[order]
    iou = box_iou(b, b)
    absorbed = np.zeros(len(b), bool)
    keep = []
    for i in range(len(b)):
        if absorbed[i]:
            continue
        keep.append(i)
        m = ~absorbed & (iou[i] >= iou_thr) & (src != src[i])
        m[: i + 1] = False
        if m.any():
            s[i] = np.maximum(s[i], s[m].max(0))
            absorbed |= m
    return b[keep], s[keep]


@torch.no_grad()
def sliced_dets(model, im, grid, frac, iou_thr=0.5, topq=150, max_pairs=300, edge=0.005, size=SIZE):
    """Per-class dets of one uint8 3 x H x W frame: (x1, y1, x2, y2, score, cls) rows, as numpy."""
    dev = next(model.parameters()).device
    H, W = im.shape[-2:]
    regions = [(0, 0, W, H)] + tiles(W, H, grid, frac)
    x = torch.cat([F.interpolate(im[:, y1:y2, x1:x2].to(dev)[None].float().div_(255), size=(size, size),
                                 mode="bilinear", antialias=True) for x1, y1, x2, y2 in regions])
    out = model.model(x)
    logits, cxcywh = out["pred_logits"].float(), out["pred_boxes"].float()
    sc = torch.sigmoid(logits) if getattr(model.post, "use_focal_loss", True) else F.softmax(logits, -1)[..., :-1]
    sc, cxcywh = sc.cpu().numpy(), cxcywh.cpu().numpy()
    if grid < 1:  # the official postprocessor: top max_pairs (query, class) pairs of the one source
        flat = torch.from_numpy(sc[0]).flatten()
        top, idx = torch.topk(flat, max_pairs)
        q, c = (idx // sc.shape[2]).numpy(), (idx % sc.shape[2]).numpy()
        bx = cxcywh[0][q]
        xyxy = np.stack([bx[:, 0] - bx[:, 2] / 2, bx[:, 1] - bx[:, 3] / 2, bx[:, 0] + bx[:, 2] / 2,
                         bx[:, 1] + bx[:, 3] / 2], 1) * np.array([W, H, W, H], np.float32)
        return np.column_stack([xyxy, top.numpy(), c])
    allb, alls, allsrc = [], [], []
    for k, (x1, y1, x2, y2) in enumerate(regions):
        tw, th = x2 - x1, y2 - y1
        bx = cxcywh[k]
        xyxy = np.stack([bx[:, 0] - bx[:, 2] / 2, bx[:, 1] - bx[:, 3] / 2, bx[:, 0] + bx[:, 2] / 2,
                         bx[:, 1] + bx[:, 3] / 2], 1) * np.array([tw, th, tw, th], np.float32)
        ok = np.ones(len(bx), bool)
        if k:  # drop boxes cut by an interior tile edge
            ex, ey = max(2.0, edge * tw), max(2.0, edge * th)
            if x1 > 0:
                ok &= xyxy[:, 0] > ex
            if x2 < W:
                ok &= xyxy[:, 2] < tw - ex
            if y1 > 0:
                ok &= xyxy[:, 1] > ey
            if y2 < H:
                ok &= xyxy[:, 3] < th - ey
        xyxy += np.array([x1, y1, x1, y1], np.float32)
        s = sc[k]
        idx = np.nonzero(ok)[0]
        idx = idx[np.argsort(-s[idx].max(1), kind="stable")[:topq]]
        allb.append(xyxy[idx])
        alls.append(s[idx])
        allsrc.append(np.full(len(idx), k))
    b, s = merge(np.concatenate(allb), np.concatenate(alls), np.concatenate(allsrc), iou_thr)
    flat = s.ravel()
    top = np.argsort(-flat, kind="stable")[:max_pairs]
    q, c = top // s.shape[1], top % s.shape[1]
    return np.column_stack([b[q], flat[top], c])


def predict(model, files, grid, frac, workers=2, size=SIZE):
    """Per-class dets (image, x, y, w, h, score, pred_cls), image sizes, mean ms per image."""
    rows, sizes, ms = [], {}, []
    for ims, stems in tv_detector.frames_loader(files, workers):
        for im, stem in zip(ims, stems):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            d = sliced_dets(model, im, grid, frac, size=size)
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - t0) * 1000)
            sizes[stem] = tuple(im.shape[-2:])
            rows += [(stem, x1, y1, x2 - x1, y2 - y1, s, int(c)) for x1, y1, x2, y2, s, c in d]
    return pd.DataFrame(rows, columns=["image", "x", "y", "w", "h", "score", "pred_cls"]), sizes, float(np.mean(ms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="an RT-DETR-family run, e.g. deim-r18-nomix-wise090_smdplus_s0")
    ap.add_argument("--grid", type=int, default=2, help="tiles per side; 0 = full frame only")
    ap.add_argument("--frac", type=float, default=0.6, help="tile size as a fraction of the frame")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--size", type=int, default=SIZE, help="input size of every source (default 640)")
    a = ap.parse_args()
    variant, protocol, seed = a.run.rsplit("_", 2)
    tag = ("-sl%d" % a.grid if a.grid or a.size == SIZE else "") + ("-r%d" % a.size if a.size != SIZE else "")
    out = RUNS / ("%s%s_%s_%s" % (variant, tag, protocol, seed)) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    model = rtdetr_official.load(variant, RUNS / a.run / "weights" / "best.pt")
    if a.size != SIZE:  # build position embeddings and anchors for the input instead of the cached 640 ones
        for part in (model.model.encoder, model.model.decoder):
            part.eval_spatial_size = None
    res = dict(run=out.parent.name, protocol=protocol, dataset=ship_data.dataset(protocol),
               weights=str(RUNS / a.run / "weights" / "best.pt"), seaships={}, seaships_coco={}, seaships_agnostic={},
               smd={}, slicing=dict(base_run=a.run, grid=a.grid, frac=a.frac, iou=0.5, topq=150, max_pairs=300, size=a.size))
    for split in (protocol + "_val", protocol + "_test"):
        files = ship_data.split_files(split)
        dt6, sizes, ms = predict(model, files, a.grid, a.frac, a.workers, a.size)
        res["slicing"]["ms_per_image_" + split] = ms
        for sub in [split] + SUBSETS.get(split, []):
            images = sorted(Path(f).stem for f in ship_data.split_files(sub))
            d6 = dt6[dt6.image.isin(set(images))].reset_index(drop=True)
            dt = tv_detector.agnostic(d6).reset_index(drop=True)
            d6.to_csv(out / ("pred6_%s_%d.csv" % (sub, a.size)), index=False)
            dt.to_csv(out / ("pred_%s_%d.csv" % (sub, a.size)), index=False)
            sz = {im: sizes[im] for im in images}
            res["seaships_coco"][sub] = class_ap(d6, images, sz, sub)
            res["seaships_agnostic"][sub] = summary(evaluate(seaships_gt(images, sz, split=sub), dt, images=images))
            print(sub, "%d-class mAP50 %.4f mAP %.4f" % (ship_data.num_classes(sub), res["seaships_coco"][sub]["map50"],
                                                         res["seaships_coco"][sub]["map"]),
                  "agnostic", res["seaships_agnostic"][sub]["all"], "%.1f ms/image" % ms, flush=True)
    for split in (protocol + "_val",):  # val is for the type stage's w sweep; keep test-only keys like eval_detector
        res["seaships_coco"].pop(split, None)
        res["seaships_agnostic"].pop(split, None)
    (out / "summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
