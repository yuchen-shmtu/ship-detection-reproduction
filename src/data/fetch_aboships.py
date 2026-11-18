"""Fetch the original ABOships dataset (Zenodo 4736931, CC-BY-4.0) as a cross-dataset test set.

ABOships (Iancu et al., Remote Sens. 2021): 9,880 PNG frames at 1280x720, one every 15 s
from videos shot on a waterbus in the Finnish archipelago, 41,967 boxes in 11 types. The
zip is 8.2 GB. Zenodo honours HTTP range requests, so, as fetch_smd.py, this reads the
zip's central directory remotely and downloads single members into data/aboships_raw/:

    Labels/Vesibussi_Labels.csv   filename, box width, height, class, xmin, xmax, ymin, ymax
    ABOshipDescription.pdf        the dataset's own description
    parse_seaships.py             its annotation viewer
    Seaships/<yyyymmdd>/<stem>.png  only the 9,041 frames that have a box in the CSV

Why not ABOships-PLUS (Zenodo 10469672, the same 9,880 frames, COCO JSON, 4 superclasses):
it drops every box under 16x16 px without marking it ignore, and the images that are left
with none (8,377 frames remain), so a detector's hits on those small objects would count as
false positives; its "images" entries also carry a box's width/height, not the frame's.
The original keeps every box and the fine types (seamark and miscellaneous can be ignore
regions, as for SMD).

Zenodo gives each connection only ~10 KB/s but scales with parallel connections (8 gave
~86 KB/s), so members are fetched by WORKERS threads, one range request per member. The
frames are ordered so that every prefix is spread evenly over the frames (stride order),
which leaves a usable subset if the transfer is stopped. Each member is inflated and checked
against the CRC-32 and size in the zip directory; complete files are skipped, so re-running
resumes.

    python src/data/fetch_aboships.py
"""
import csv
import sys
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fetch_smd import RangeFile, fetch_range

URL = "https://zenodo.org/api/records/4736931/files/ABOshipsDataset.zip/content"
SIZE = 8168648505
OUT = Path(__file__).resolve().parents[2] / "data" / "aboships_raw"
WORKERS = 16
STRIDE = 8
EXTRA = 256  # read past the local header's name for its extra field, in the same request


def target(info):
    # "ABOshipsDataset/ABOshipDescription.pdf.pdf" -> "ABOshipDescription.pdf"
    rel = info.filename.split("/", 1)[1]
    return OUT / (rel[:-4] if rel.endswith(".pdf.pdf") else rel)


def fetch(info):
    dest = target(info)
    if dest.exists() and dest.stat().st_size == info.file_size:
        return None
    name = len(info.filename.encode())
    raw = fetch_range(URL, info.header_offset, info.header_offset + 30 + name + EXTRA + info.compress_size - 1)
    n, e = int.from_bytes(raw[26:28], "little"), int.from_bytes(raw[28:30], "little")
    data = raw[30 + n + e:30 + n + e + info.compress_size]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    if zlib.crc32(data) != info.CRC or len(data) != info.file_size:
        raise RuntimeError("%s: CRC/size mismatch" % info.filename)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return info.file_size


def main():
    infos = zipfile.ZipFile(RangeFile(URL, SIZE)).infolist()
    meta = [i for i in infos if not i.filename.endswith((".png", "/"))]
    for i in meta:
        fetch(i)
    with open(OUT / "Labels" / "Vesibussi_Labels.csv", newline="") as f:
        labelled = {r["filename"] for r in csv.DictReader(f)}
    frames = sorted((i for i in infos if i.filename.endswith(".png") and Path(i.filename).stem in labelled),
                    key=lambda i: i.filename)
    frames = [i for k in range(STRIDE) for i in frames[k::STRIDE]]
    print("%d of %d frames labelled, %.2f GB compressed" % (len(frames), sum(i.filename.endswith(".png") for i in infos),
                                                           sum(i.compress_size for i in frames) / 1e9), flush=True)
    done = failed = got = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futures = {ex.submit(fetch, i): i for i in frames}
        for f in as_completed(futures):
            try:
                got += f.result() or 0
                done += 1
            except Exception as err:  # keep going; a re-run resumes
                failed += 1
                print("FAILED", futures[f].filename, err, flush=True)
            if (done + failed) % 200 == 0:
                print("%d done, %d failed, %.2f GB fetched" % (done, failed, got / 1e9), flush=True)
    print("done, %d failed" % failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
