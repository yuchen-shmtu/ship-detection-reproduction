"""Build the canonical SeaShips pool in data/seaships from the Roboflow export in data/seaships7k.

Roboflow stretched every 1920x1080 frame to 640x640. This undoes the stretch by resizing
back to 16:9 at 1138x640 (no vertical detail is lost; the horizontal axis is upsampled),
so ship shapes match native-resolution datasets such as SMD. YOLO labels are normalised,
so they carry over unchanged. Files are renamed to the original SeaShips ID (000002.jpg)
and the uploader's split is recorded in splits/uploader_{train,test}.txt.
"""
import re
import sys
from pathlib import Path

import cv2
import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data"
SRC = DATA / "seaships7k"
DST = DATA / "seaships"
SIZE = (1138, 640)  # (w, h), 640 * 16 / 9 rounded
NC = 6


def check_labels(text, name):
    rows = [ln.split() for ln in text.splitlines() if ln.strip()]
    out, problems = [], []
    for r in rows:
        if len(r) != 5:
            problems.append("%s: bad row %r" % (name, r))
            continue
        c, x, y, w, h = int(r[0]), *map(float, r[1:])
        if not 0 <= c < NC:
            problems.append("%s: class %d" % (name, c))
            continue
        # Clip to the image; drop boxes that end up empty.
        x1, y1, x2, y2 = max(0.0, x - w / 2), max(0.0, y - h / 2), min(1.0, x + w / 2), min(1.0, y + h / 2)
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            problems.append("%s: empty box %r" % (name, r))
            continue
        if (x1, y1, x2, y2) != (x - w / 2, y - h / 2, x + w / 2, y + h / 2):
            problems.append("%s: clipped box %r" % (name, r))
        out.append("%d %.6f %.6f %.6f %.6f" % (c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return out, problems


def main():
    (DST / "images").mkdir(parents=True, exist_ok=True)
    (DST / "labels").mkdir(parents=True, exist_ok=True)
    (DST / "splits").mkdir(parents=True, exist_ok=True)
    seen, problems, empty = {}, [], []
    for split in ("train", "test"):
        ids = []
        for img in sorted((SRC / split / "images").glob("*.jpg")):
            sid = "%06d" % int(re.match(r"(\d+)_jpg", img.name).group(1))
            if sid in seen:
                sys.exit("duplicate ID %s in %s and %s" % (sid, seen[sid], split))
            seen[sid] = split
            ids.append(sid)
            lbl = SRC / split / "labels" / (img.stem + ".txt")
            rows, p = check_labels(lbl.read_text() if lbl.exists() else "", sid)
            problems += p
            if not rows:
                empty.append(sid)
            (DST / "labels" / (sid + ".txt")).write_text("\n".join(rows) + ("\n" if rows else ""))
            out = DST / "images" / (sid + ".jpg")
            if not out.exists():
                im = cv2.imread(str(img))
                assert im.shape[:2] == (640, 640), (img, im.shape)
                im = cv2.resize(im, SIZE, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(str(out), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (DST / "splits" / ("uploader_%s.txt" % split)).write_text("\n".join(ids) + "\n")
        print("%s: %d images" % (split, len(ids)))

    ids = np.array(sorted(int(s) for s in seen))
    missing = sorted(set(range(ids.min(), ids.max() + 1)) - set(ids))
    print("pool: %d images, IDs %d-%d, missing IDs: %s" % (len(ids), ids.min(), ids.max(), missing))
    print("images without boxes: %d %s" % (len(empty), empty[:20]))
    print("label problems: %d" % len(problems))
    for p in problems[:20]:
        print("  " + p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
