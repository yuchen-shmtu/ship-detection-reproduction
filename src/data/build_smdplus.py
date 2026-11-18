"""Build the SMD-Plus detection set: train / val / test by whole video.

SMD-Plus (Kim et al., "Object Detection and Classification Based on YOLO-V5 with Improved
Maritime Dataset", JMSE 2022, 10, 377; github.com/kjunhwa/SMD-Plus, Google Drive file
1yokFvx_cJu-Fl5ti1wgF1WPHao_2kcKE = data/_raw/smd_plus.zip) relabels the 51 VIS videos of the
Singapore Maritime Dataset: tightened boxes, Speed boat merged into Boat, Swimming person and
Flying bird/plane removed, Ferry / Other relabelled, missing Kayak boxes added. Its ObjectGT
.mat files have the original SMD layout (structXML, BB = [x, y, w, h]) with Object codes 1..7 =
Ferry, Buoy, Vessel/ship, Boat, Kayak, Sail boat, Other (counts match the paper's Table 1).

Reads data/smdplus_raw/VIS_{Onshore,Onboard}/{Videos,ObjectGT} (the two inner zips, unpacked)
and writes data/smdplus/:

    images/<gt stem>_<frame>.jpg  every STEP-th GT frame that has boxes, native 1920x1080, JPEG 95
    labels/<stem>.txt             YOLO boxes, all 7 classes (class = code - 1), as the paper
    annotations.csv               every box: image, video, frame, code, name, x, y, w, h, clipped
    frames.csv                    image, video, subset, frame, n_box, split
    splits/smdplus_{train,val,test}.txt, yolo/smdplus.{yaml,_*.txt}, coco/smdplus_*.json

Split: the paper's Table 2 (train 28 onshore + 9 onboard videos, test 12 onshore + 2 onboard);
val is five whole train videos (VAL), chosen so that every class but Kayak still has train
videos (Kayak is in MVI_1609, train, and MVI_1592, test, only). Frames of one video never
straddle two splits.

GT notes: onboard GT stems lack the video file's _OB suffix (MVI_0788_VIS vs
MVI_0788_VIS_OB.avi); a GT frame with no boxes means unannotated (MVI_0790: 97 of 600 frames
annotated) and is skipped, as in build_smd.py; boxes are clipped to the frame and dropped under
2 px; only frames < min(video, GT) are used.

    python src/data/build_smdplus.py
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[2] / "data"
RAW = ROOT / "smdplus_raw"
OUT = ROOT / "smdplus"
STEP = 5
WORKERS = 4
NAMES = ["ferry", "buoy", "vessel/ship", "boat", "kayak", "sail boat", "other"]
TEST_VIDEOS = {"MVI_1448_VIS_Haze", "MVI_1469_VIS", "MVI_1474_VIS", "MVI_1587_VIS", "MVI_1592_VIS", "MVI_1613_VIS",
               "MVI_1614_VIS", "MVI_1615_VIS", "MVI_1640_VIS", "MVI_1644_VIS", "MVI_1645_VIS", "MVI_1646_VIS",
               "MVI_0799_VIS", "MVI_0804_VIS"}
VAL_VIDEOS = {"MVI_1470_VIS", "MVI_1478_VIS", "MVI_1612_VIS", "MVI_1622_VIS", "MVI_0801_VIS"}


def load_gt(path):
    """Per frame: array of rows (code, x, y, w, h), empty if unannotated."""
    frames = []
    for fr in np.atleast_1d(sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)["structXML"]):
        bb = np.atleast_2d(np.asarray(fr.BB, float))
        if bb.size == 0:
            frames.append(np.zeros((0, 5)))
            continue
        frames.append(np.column_stack([np.atleast_1d(np.asarray(fr.Object, float)), bb]))
    return frames


def build_video(gt_path):
    stem = gt_path.name[:-len("_ObjectGT.mat")]
    subset = "onboard" if gt_path.parents[1].name == "VIS_Onboard" else "onshore"
    video = sorted((gt_path.parents[1] / "Videos").glob(stem + "*.avi"))
    if len(video) != 1:
        sys.exit("%s: %d matching videos" % (stem, len(video)))
    gt = load_gt(gt_path)
    cap = cv2.VideoCapture(str(video[0]))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    split = "test" if stem in TEST_VIDEOS else "val" if stem in VAL_VIDEOS else "train"
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
        lines = []
        for code, x, y, w, h in gt[f]:
            x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
            c = int(code)
            if x2 - x1 < 2 or y2 - y1 < 2 or not 1 <= c <= len(NAMES):
                continue
            boxes.append((name, stem, f, c, NAMES[c - 1], round(x1, 1), round(y1, 1), round(x2 - x1, 1),
                          round(y2 - y1, 1), int((x1, y1, x2, y2) != (x, y, x + w, y + h))))
            lines.append("%d %.6f %.6f %.6f %.6f" % (c - 1, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W,
                                                     (y2 - y1) / H))
        if not lines:
            continue
        cv2.imwrite(str(OUT / "images" / (name + ".jpg")), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (OUT / "labels" / (name + ".txt")).write_text("\n".join(lines) + "\n")
        frames.append((name, stem, subset, f, len(lines), split, H, W))
    cap.release()
    print("%-18s %-7s %-5s video %4d GT %4d -> %3d frames" % (stem, subset, split, n_video, len(gt), len(frames)))
    return frames, boxes


def write_coco(name, fr, boxes):
    ids = {im: k + 1 for k, im in enumerate(fr.image)}
    b = boxes[boxes.image.isin(ids)]
    d = dict(images=[dict(id=ids[r.image], file_name=r.image + ".jpg", height=int(r.H), width=int(r.W))
                     for r in fr.itertuples()],
             annotations=[dict(id=k + 1, image_id=ids[r.image], category_id=int(r.code) - 1, bbox=[r.x, r.y, r.w, r.h],
                               area=r.w * r.h, iscrowd=0) for k, r in enumerate(b.itertuples())],
             categories=[dict(id=k, name=n) for k, n in enumerate(NAMES)])
    (OUT / "coco" / (name + ".json")).write_text(json.dumps(d))


def main():
    for d in ("images", "labels", "yolo", "splits", "coco"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    for d in ("images", "labels"):
        for p in (OUT / d).iterdir():
            p.unlink()
    gts = sorted(RAW.glob("VIS_*/ObjectGT/*_ObjectGT.mat"))
    stems = {g.name[:-len("_ObjectGT.mat")] for g in gts}
    if len(gts) != 51 or not (TEST_VIDEOS | VAL_VIDEOS) <= stems:
        sys.exit("expected the 51 SMD-Plus videos incl. every test / val video; found %d" % len(gts))
    with ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(build_video, gts))
    frames = pd.DataFrame([r for f, _ in results for r in f],
                          columns=["image", "video", "subset", "frame", "n_box", "split", "H", "W"])
    boxes = pd.DataFrame([r for _, b in results for r in b],
                         columns=["image", "video", "frame", "code", "name", "x", "y", "w", "h", "clipped"])
    frames.drop(columns=["H", "W"]).to_csv(OUT / "frames.csv", index=False)
    boxes.to_csv(OUT / "annotations.csv", index=False)
    names = "".join("  %d: %s\n" % (k, n) for k, n in enumerate(NAMES))
    for split in ("train", "val", "test"):
        fr = frames[frames.split == split]
        (OUT / "splits" / ("smdplus_%s.txt" % split)).write_text("".join(i + "\n" for i in fr.image))
        (OUT / "yolo" / ("smdplus_%s.txt" % split)).write_text("".join("./../images/%s.jpg\n" % i for i in fr.image))
        write_coco("smdplus_" + split, fr, boxes)
    (OUT / "yolo" / "smdplus.yaml").write_text("path: %s\ntrain: smdplus_train.txt\nval: smdplus_val.txt\n"
                                               "test: smdplus_test.txt\nnames:\n%s" % ((OUT / "yolo").as_posix(), names))
    b = boxes.merge(frames[["image", "split"]], on="image")
    print("\n%d videos, %d frames, %d boxes" % (frames.video.nunique(), len(frames), len(boxes)))
    print(frames.groupby("split").agg(videos=("video", "nunique"), frames=("image", "size")).to_string())
    print(b.pivot_table(index="name", columns="split", values="x", aggfunc="size", fill_value=0).to_string())
    area = boxes.w * boxes.h / (1920 * 1080)
    print("boxes under 1%% of the image: %.1f%%; median height %d px" % (100 * (area < 0.01).mean(), boxes.h.median()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
