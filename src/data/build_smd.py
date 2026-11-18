"""Build a frame-level detection set from the raw Singapore Maritime Dataset (VIS, annotated).

Reads data/smd_raw/VIS_{Onshore,Onboard}/{Videos,ObjectGT} (see src/data/fetch_smd.py) and
writes data/smd_vis/:

    images/<video>_<frame>.jpg   every STEP-th annotated frame, native 1920x1080
    labels/<video>_<frame>.txt   YOLO boxes for the ship-like types, class = index in SHIPS
    annotations.csv              every GT box on those frames, all types, pixel x/y/w/h
                                 clipped to the frame, with motion and distance codes
    frames.csv                   image, video, subset, frame, box counts
    yolo/{all,onshore,onboard}.{txt,yaml}

GT notes, checked on overlay sheets:

  - GT and video agree from frame 0. Three GT files run past the end of their video
    (MVI_0790_VIS_OB 1010 vs 600 frames, MVI_1584_VIS 550 vs 539, MVI_0799_VIS_OB 601 vs
    600); the boxes still fit the last video frame, so the surplus is at the tail and only
    frames < min(video, GT) are used.
  - A GT frame with no boxes means unannotated, not empty: MVI_0790_VIS_OB frames ~350-550
    show vessels with no boxes. Such frames are skipped.
  - 'Other' (10) mixes ships cut off at the frame edge, a lighthouse, a shore tank farm and a
    person's head in front of the lens. It stays out of the labels, like Buoy, Swimming
    person and Flying bird/plane; annotations.csv keeps them (ship=0) for use as ignore
    regions.
  - One box (MVI_1613_VIS frame 0) has type, motion and distance 0 and zero size; boxes
    under 2 px after clipping are dropped.

The yaml files point train/val/test at the same list: SMD is a test-only set here.

    python src/data/build_smd.py
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[2] / "data"
RAW = ROOT / "smd_raw"
OUT = ROOT / "smd_vis"
STEP = 5
WORKERS = 4
TYPES = {1: "ferry", 2: "buoy", 3: "vessel/ship", 4: "speed boat", 5: "boat", 6: "kayak", 7: "sail boat",
         8: "swimming person", 9: "flying bird/plane", 10: "other"}
SHIPS = [1, 3, 4, 5, 6, 7]


def load_gt(path):
    """Per frame: array of rows (type, motion, distance, x, y, w, h), empty if unannotated."""
    frames = []
    for fr in np.atleast_1d(sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)["structXML"]):
        bb = np.atleast_2d(np.asarray(fr.BB, float))
        if bb.size == 0:
            frames.append(np.zeros((0, 7)))
            continue
        cols = [np.atleast_1d(np.asarray(getattr(fr, f), float)) for f in ("Object", "Motion", "Distance")]
        frames.append(np.column_stack(cols + [bb]))
    return frames


def build_video(gt_path):
    stem = gt_path.name[:-len("_ObjectGT.mat")]
    subset = "onboard" if gt_path.parents[1].name == "VIS_Onboard" else "onshore"
    gt = load_gt(gt_path)
    cap = cv2.VideoCapture(str(gt_path.parents[1] / "Videos" / (stem + ".avi")))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, boxes = [], []
    for f in range(min(n_video, len(gt))):
        if not cap.grab():
            print(stem, "decode stopped at frame", f)
            break
        if f % STEP or not len(gt[f]):
            continue
        ok, im = cap.retrieve()
        if not ok:
            print(stem, "cannot decode frame", f)
            continue
        H, W = im.shape[:2]
        name = "%s_%04d" % (stem, f)
        rows, lines = [], []
        for t, m, d, x, y, w, h in gt[f]:
            x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            t = int(t)
            ship = t in SHIPS
            rows.append((name, stem, f, t, TYPES.get(t, "invalid"), int(ship), int(m), int(d),
                         round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1),
                         int((x1, y1, x2, y2) != (x, y, x + w, y + h))))
            if ship:
                lines.append("%d %.6f %.6f %.6f %.6f" % (SHIPS.index(t), (x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                                                         (x2 - x1) / W, (y2 - y1) / H))
        if not rows:
            continue
        cv2.imwrite(str(OUT / "images" / (name + ".jpg")), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (OUT / "labels" / (name + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        frames.append((name, stem, subset, f, len(lines), len(rows) - len(lines)))
        boxes.extend(rows)
    cap.release()
    print("%-18s %-7s video %4d GT %4d -> %3d frames, %5d boxes" % (stem, subset, n_video, len(gt), len(frames),
                                                                     len(boxes)))
    return frames, boxes


def main():
    for d in ("images", "labels", "yolo"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    for d in ("images", "labels"):
        for p in (OUT / d).iterdir():
            p.unlink()
    gts = sorted(RAW.glob("VIS_*/ObjectGT/*_ObjectGT.mat"))
    with ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(build_video, gts))
    frames = pd.DataFrame([r for f, _ in results for r in f],
                          columns=["image", "video", "subset", "frame", "n_ship", "n_ignore"])
    boxes = pd.DataFrame([r for _, b in results for r in b],
                         columns=["image", "video", "frame", "type", "name", "ship", "motion", "distance",
                                  "x", "y", "w", "h", "clipped"])
    frames.to_csv(OUT / "frames.csv", index=False)
    boxes.to_csv(OUT / "annotations.csv", index=False)
    names = "".join("  %d: %s\n" % (k, TYPES[t]) for k, t in enumerate(SHIPS))
    for subset in ("all", "onshore", "onboard"):
        sel = frames if subset == "all" else frames[frames.subset == subset]
        (OUT / "yolo" / (subset + ".txt")).write_text("".join("./../images/%s.jpg\n" % i for i in sel.image))
        (OUT / "yolo" / (subset + ".yaml")).write_text(
            "path: %s\ntrain: %s.txt\nval: %s.txt\ntest: %s.txt\nnames:\n%s"
            % ((OUT / "yolo").as_posix(), subset, subset, subset, names))
    print("\n%d videos, %d frames (%s), %d boxes" % (frames.video.nunique(), len(frames),
                                                     frames.subset.value_counts().to_dict(), len(boxes)))
    print(boxes.groupby("name").agg(boxes=("ship", "size"), clipped=("clipped", "sum")).to_string())
    s = boxes[boxes.ship == 1]
    area = s.w * s.h / (1920 * 1080)
    print("ship boxes: %d; under 1%% of the image: %.1f%%; median height %d px"
          % (len(s), 100 * (area < 0.01).mean(), s.h.median()))
    print("frames with no ship box: %d" % (frames.n_ship == 0).sum())
    return 0


if __name__ == "__main__":
    sys.exit(main())
