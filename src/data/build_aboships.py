"""Build a test-only detection set from the original ABOships dataset.

Reads data/aboships_raw/ (src/data/fetch_aboships.py): Labels/Vesibussi_Labels.csv and
Seaships/<yyyymmdd>/<stem>.png, and writes data/aboships/:

    images/<stem>.jpg     every labelled frame that has been fetched, native 1280x720 (JPEG 95,
                          as data/smd_vis/, so the torchvision loaders read it)
    labels/<stem>.txt     YOLO boxes for the ship types, class = index in SHIPS
    annotations.csv       every GT box on those frames, all types, pixel x/y/w/h clipped to
                          the frame; ship = 0 for the non-ship types
    frames.csv            image, video, date, box counts
    yolo/all.{txt,yaml}

<stem> is <yyyymmddHHMM>_<k>: one video (202 of them) and the k-th frame taken from it, one
every 15 s. The waterbus camera moves, so frames of one video are not near-duplicates the
way SeaShips' fixed-camera frames are.

CSV notes:

  - width/height are the box's, xmax - xmin and ymax - ymin; no duplicate rows.
  - 527 boxes run past the frame (x down to -178, up to 1398): clipped here, flagged in
    annotations.csv.
  - 839 of the 9,880 frames have no row. Whether they are empty or unannotated is not
    stated, so they are left out, as SMD's box-less frames.
  - Seamark (buoys and navigation marks) and Miscellaneous (200 boxes, median 12x10 px;
    birds on the water on a crop sheet) are not ships: they stay out of the labels and are
    kept in annotations.csv as ignore regions.

Frame notes, from an overlay sheet of the first fetched frames: the waterbus's own orange
bow and railing fill the bottom third of every frame, unlabelled; moored boats in marinas
and under bridges are often left without a box, or share one box as a group. So some
correct detections count as false positives, for every model alike.

A partial fetch builds a partial set; the counts printed say how many labelled frames are
still missing. Re-run after the fetch completes.

    python src/data/build_aboships.py
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "data"
RAW = ROOT / "aboships_raw"
OUT = ROOT / "aboships"
W, H = 1280, 720
WORKERS = 4
SHIPS = ["boat", "cargoship", "cruiseship", "ferry", "militaryship", "miscboat", "motorboat", "passengership",
         "sailboat"]
NON_SHIPS = ["seamark", "miscellaneous"]


def build_frame(item):
    stem, boxes = item
    png = RAW / "Seaships" / stem[:8] / (stem + ".png")
    if not png.exists():
        return None
    im = cv2.imread(str(png))
    if im is None or im.shape[:2] != (H, W):
        raise RuntimeError("%s: cannot read or not %dx%d" % (png, W, H))
    rows, lines = [], []
    for b in boxes.itertuples():
        x1, y1, x2, y2 = max(b.xmin, 0), max(b.ymin, 0), min(b.xmax, W), min(b.ymax, H)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        t = b.type.lower()
        ship = t in SHIPS
        rows.append((stem, stem.split("_")[0], t, int(ship), x1, y1, x2 - x1, y2 - y1,
                     int((x1, y1, x2, y2) != (b.xmin, b.ymin, b.xmax, b.ymax))))
        if ship:
            lines.append("%d %.6f %.6f %.6f %.6f" % (SHIPS.index(t), (x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                                                     (x2 - x1) / W, (y2 - y1) / H))
    cv2.imwrite(str(OUT / "images" / (stem + ".jpg")), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    (OUT / "labels" / (stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    return (stem, stem.split("_")[0], stem[:8], len(lines), len(rows) - len(lines)), rows


def main():
    csv = pd.read_csv(RAW / "Labels" / "Vesibussi_Labels.csv", dtype={"filename": str})
    csv = csv.rename(columns={"class": "type"})
    unknown = set(csv.type.str.lower()) - set(SHIPS) - set(NON_SHIPS)
    if unknown:
        sys.exit("unknown types: %s" % unknown)
    for d in ("images", "labels", "yolo"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    for d in ("images", "labels"):
        for p in (OUT / d).iterdir():
            p.unlink()
    with ThreadPoolExecutor(WORKERS) as ex:
        results = [r for r in ex.map(build_frame, csv.groupby("filename", sort=True)) if r]
    frames = pd.DataFrame([f for f, _ in results], columns=["image", "video", "date", "n_ship", "n_ignore"])
    boxes = pd.DataFrame([r for _, b in results for r in b],
                         columns=["image", "video", "type", "ship", "x", "y", "w", "h", "clipped"])
    frames.to_csv(OUT / "frames.csv", index=False)
    boxes.to_csv(OUT / "annotations.csv", index=False)
    (OUT / "yolo" / "all.txt").write_text("".join("./../images/%s.jpg\n" % i for i in frames.image))
    (OUT / "yolo" / "all.yaml").write_text(
        "path: %s\ntrain: all.txt\nval: all.txt\ntest: all.txt\nnames:\n%s"
        % ((OUT / "yolo").as_posix(), "".join("  %d: %s\n" % (k, t) for k, t in enumerate(SHIPS))))

    n_lab = csv.filename.nunique()
    print("%d of %d labelled frames built (%d not fetched yet), %d videos, %d boxes"
          % (len(frames), n_lab, n_lab - len(frames), frames.video.nunique(), len(boxes)))
    print(boxes.groupby("type").agg(boxes=("ship", "size"), clipped=("clipped", "sum")).to_string())
    s = boxes[boxes.ship == 1]
    area = s.w * s.h
    print("ship boxes: %d; COCO small (< 32^2 px): %.1f%%; under 1%% of the image: %.1f%%; median height %d px"
          % (len(s), 100 * (area < 32 ** 2).mean(), 100 * (area < 0.01 * W * H).mean(), s.h.median()))
    print("frames with no ship box: %d" % (frames.n_ship == 0).sum())
    return 0


if __name__ == "__main__":
    sys.exit(main())
