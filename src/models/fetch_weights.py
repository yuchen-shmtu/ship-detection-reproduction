"""Fetch COCO-pretrained ultralytics weights into weights/, checked against the official sha256.

Files are fetched through api.github.com, which also works where github.com itself is not
reachable: a release asset's API url with "Accept: application/octet-stream" redirects to
release-assets.githubusercontent.com. Slow transfers stall and get cut off, so each file is
fetched in resumed pieces (curl -C -, a fresh signed redirect per piece) until its size
matches, then its sha256 is checked against the digest the release lists. Only then is it
moved to weights/<name>; a .part file is kept for resuming.

The default release is ultralytics/assets v8.4.0, which lists a sha256 for every asset.
--release picks another; lyuwenyu/storage (the RT-DETR weights) lists no digests, so its files
are checked by size only and printed with their sha256 (compare weights/README.md).

Unauthenticated api.github.com allows 60 requests an hour; one per piece plus one for the
release, so a few files per hour at most get retried hard.

    python src/models/fetch_weights.py yolov8n.pt yolov5nu.pt yolov10n.pt yolo12n.pt rtdetr-l.pt
    python src/models/fetch_weights.py --release lyuwenyu/storage@v0.1 rtdetr_r18vd_dec3_6x_coco_from_paddle.pth
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "weights"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(asset, attempts):
    name, size = asset["name"], asset["size"]
    digest = asset["digest"].split(":", 1)[1] if asset.get("digest") else None
    dst, part = WEIGHTS / name, WEIGHTS / (name + ".part")
    if dst.exists() and (sha256(dst) == digest if digest else dst.stat().st_size == size):
        print(name, "present,", "sha256 ok" if digest else "size ok (no digest listed)", flush=True)
        return True
    for i in range(attempts):
        have = part.stat().st_size if part.exists() else 0
        if have >= size:
            break
        # a stall under 2 KB/s for 60 s ends the piece; the next one resumes from the .part size
        subprocess.run(["curl", "-s", "-L", "-C", "-", "--speed-limit", "2000", "--speed-time", "60", "-m", "1800",
                        "-H", "Accept: application/octet-stream", "-o", str(part), asset["url"]])
        now = part.stat().st_size if part.exists() else 0
        print("%s piece %d: %d -> %d of %d bytes" % (name, i + 1, have, now, size), flush=True)
        if now == have:
            time.sleep(30)
    if not part.exists() or part.stat().st_size != size:
        print(name, "FAILED: incomplete after %d pieces" % attempts, flush=True)
        return False
    got = sha256(part)
    if digest is None:
        part.replace(dst)
        print(name, "ok by size; no digest listed; sha256", got, flush=True)
        return True
    if got != digest:
        print(name, "FAILED: sha256 %s != %s; .part removed" % (got, digest), flush=True)
        part.unlink()
        return False
    part.replace(dst)
    print(name, "ok, sha256", digest, flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--release", default="ultralytics/assets@v8.4.0", help="owner/repo@tag")
    ap.add_argument("--attempts", type=int, default=40)
    a = ap.parse_args()
    repo, tag = a.release.split("@")
    url = "https://api.github.com/repos/%s/releases/tags/%s" % (repo, tag)
    rel = json.loads(subprocess.run(["curl", "-s", "-m", "120", url], capture_output=True, check=True).stdout)
    assets = {x["name"]: x for x in rel["assets"]}
    missing = [n for n in a.names if n not in assets]
    if missing:
        sys.exit("not in release %s: %s" % (rel["tag_name"], missing))
    ok = [fetch(assets[n], a.attempts) for n in a.names]
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
