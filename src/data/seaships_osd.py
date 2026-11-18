"""Read the camera name and capture timestamp burned into the SeaShips frames by the OSD.

Camera: the name sits in the bottom-left corner ("北山咀东HQ211球机"). Two stages:

1. Site. The bottom-band edge map is matched against src/data/osd_camera_templates.npz
   (mean edge maps of k-means clusters of this band, labelled by eye). Each template
   belongs to a site: the place-name prefix and where the camera ID sits after it.
   Cameras at one site share the prefix, so this stage cannot tell them apart.
2. ID. The three ID digits are read at the site's position with the timestamp digit
   templates (same OSD font and pitch), the letter before them against J/Q templates.

Checked by eye: every frame where the camera changes in ID order is a real change, and
the 10 lowest-scoring reads are right. The PTZ preset overlay ("P074:T00") that some
cameras draw over the name does not reach the ID row.

Timestamp: the top-left line comes in one of two layouts, both starting with the year at
the same place:

    dash: "YYYY-MM-DD <weekday> HH:MM:SS"
    cjk:  "YYYY年MM月DD日 <weekday> HH:MM:SS"

The line is located per frame by matching the "2017" edge pattern, then each digit cell is
matched against src/data/osd_digit_templates.npz (built the same way as the camera templates)
with normalised cross-correlation over a few pixels of shift. The layout whose digits
parse to a valid time with the higher mean score wins. Checked by eye on the 20 reads
with the smallest margin: 3 had the last seconds digit wrong, the rest were right.

Writes data/seaships/osd.csv: id, site, camera, site margin over the best other site,
worst ID character score, timestamp (ISO, empty if unread), layout, anchor score, worst
digit score, worst digit margin, raw digits.
"""
import csv
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

POOL = Path(__file__).resolve().parents[2] / "data" / "seaships"
TEMPLATES = Path(__file__).resolve().parent / "osd_digit_templates.npz"
CAMERAS = Path(__file__).resolve().parent / "osd_camera_templates.npz"
CAM_SHIFT = 4
ID_SHIFT = 4
X0, PITCH, Y0, H, PAD = 52.5, 25.8, 37, 30, 3  # nominal cell grid, pool (1138x640) pixels
LAYOUTS = {
    "dash": [5, 6, 8, 9, 18, 19, 21, 22, 24, 25],
    "cjk": [6, 7, 10, 11, 22, 23, 25, 26, 28, 29],
}
SEARCH_Y = (-15, 15)
SEARCH_X = (-6, 6)
MIN_ANCHOR = 0.5


def edge(g):
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return np.sqrt(gx * gx + gy * gy)


def cell_x(k):
    return int(round(X0 + PITCH * k))


def year_template(tmpl):
    """Place the 2, 0, 1, 7 digit templates side by side on the nominal grid."""
    w = cell_x(4) - cell_x(0)
    out = np.zeros((H, w + 2 * PAD), np.float32)
    for k, d in enumerate((2, 0, 1, 7)):
        x = cell_x(k) - cell_x(0)
        out[:, x:x + tmpl.shape[2]] = np.maximum(out[:, x:x + tmpl.shape[2]], tmpl[d])
    return out


def best_match(window, templates):
    """Scores of each template at its best position in the window, and that position."""
    res = [cv2.minMaxLoc(cv2.matchTemplate(window, t, cv2.TM_CCOEFF_NORMED))[1:4:2] for t in templates]
    return np.array([r[0] for r in res]), [r[1] for r in res]


def pick_site(m, cams):
    """Index of the best-matching band template, and its margin over the best other site."""
    t, y0, y1 = cams["templates"], int(cams["y0"]), int(cams["y1"])
    s = CAM_SHIFT
    region = np.pad(m[y0 - s:], ((0, max(0, y1 - y0 + 2 * s - (m.shape[0] - y0 + s))), (0, 0)))
    region = region[:y1 - y0 + 2 * s, :t.shape[2] + 2 * s]
    scores, _ = best_match(region, t[:, s:-s, s:-s])
    k = int(np.argmax(scores))
    other = scores[cams["sites"] != cams["sites"][k]].max()
    return k, float(scores[k] - other)


def read_digits(m, x0, y0, inner, n=3):
    """Read n digits on the OSD pitch starting at cell (x0, y0). Returns the digits, the
    worst score, the worst margin over the runner-up digit, and where the first one sat."""
    digits, scores, margins, first = "", [], [], None
    h, w = inner.shape[1:]
    for k in range(n):
        x = x0 + int(round(PITCH * k))
        s, locs = best_match(m[y0 - ID_SHIFT:y0 + h + ID_SHIFT, x - ID_SHIFT:x + w + ID_SHIFT], inner)
        order = np.argsort(s)[::-1]
        digits += str(order[0])
        scores.append(s[order[0]])
        margins.append(s[order[0]] - s[order[1]])
        if first is None:
            first = (x - ID_SHIFT + locs[order[0]][0], y0 - ID_SHIFT + locs[order[0]][1])
    return digits, min(scores), min(margins), first


def read_camera(m, cams, inner):
    """Site, camera ID ("HQ010"), site margin, and worst ID-character score."""
    k, site_margin = pick_site(m, cams)
    digits, score, _, (x, y) = read_digits(m, int(cams["id_x"][k]), int(cams["id_y"][k]), inner)
    lx = x - int(round(PITCH))
    s, _ = best_match(m[y - ID_SHIFT:y + inner.shape[1] + ID_SHIFT, lx - ID_SHIFT:lx + inner.shape[2] + ID_SHIFT],
                      cams["letters"])
    letter = str(cams["letter_names"][int(np.argmax(s))])
    return str(cams["sites"][k]), "H" + letter + digits, site_margin, float(min(score, s.max()))


def main():
    tmpl = np.load(TEMPLATES)["templates"]  # (10, H, cell width + 2*PAD), edge maps
    cams = np.load(CAMERAS)
    inner = tmpl[:, 2:-2, PAD:-PAD].copy()
    year = year_template(tmpl)
    rows, n_ok = [], 0
    for p in sorted((POOL / "images").glob("*.jpg")):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        m = edge(g.astype(np.float32))
        site, cam, site_margin, cam_score = read_camera(m, cams, inner)
        # Locate the line by the year.
        y_lo, x_lo = Y0 + SEARCH_Y[0], cell_x(0) - PAD + SEARCH_X[0]
        region = m[y_lo:Y0 + H + SEARCH_Y[1], x_lo:cell_x(0) - PAD + year.shape[1] + SEARCH_X[1]]
        r = cv2.matchTemplate(region, year, cv2.TM_CCOEFF_NORMED)
        _, anchor, _, (bx, by) = cv2.minMaxLoc(r)
        dy, dx = y_lo + by - Y0, x_lo + bx - (cell_x(0) - PAD)
        best = None
        if anchor >= MIN_ANCHOR:
            for name, cells in LAYOUTS.items():
                digits, scores, margins = "", [], []
                for k in cells:
                    x, y = cell_x(k) - PAD + dx, Y0 + dy
                    cell = m[y - 2:y + H + 2, x - 2:x + tmpl.shape[2] + 2]
                    s = np.array([cv2.matchTemplate(cell, t, cv2.TM_CCOEFF_NORMED).max() for t in inner])
                    order = np.argsort(s)[::-1]
                    digits += str(order[0])
                    scores.append(s[order[0]])
                    margins.append(s[order[0]] - s[order[1]])
                try:
                    ts = datetime.strptime("2017" + digits, "%Y%m%d%H%M%S").isoformat()
                except ValueError:
                    ts = ""
                cand = (bool(ts), float(np.mean(scores)), name, ts, min(scores), min(margins), digits)
                if best is None or cand[:2] > best[:2]:
                    best = cand
        head = (p.stem, site, cam, "%.3f" % site_margin, "%.3f" % cam_score)
        if best and best[0]:
            n_ok += 1
            rows.append(head + (best[3], best[2], "%.3f" % anchor, "%.3f" % best[4], "%.3f" % best[5], best[6]))
        else:
            rows.append(head + ("", best[2] if best else "", "%.3f" % anchor, "", "", best[6] if best else ""))
    with open(POOL / "osd.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "site", "camera", "site_margin", "camera_score", "timestamp", "layout", "anchor", "score",
                    "margin", "raw"])
        w.writerows(rows)
    print("timestamps read: %d / %d" % (n_ok, len(rows)))
    print("cameras:", dict(Counter((r[1], r[2]) for r in rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
