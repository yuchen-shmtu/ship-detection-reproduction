"""Fetch the annotated visible-light part of the raw Singapore Maritime Dataset.

The Kaggle mirror mmichelli/singapore-maritime-dataset is one 5.5 GB zip, mostly NIR video
and VIS video without object annotations. Kaggle redirects to a Google Cloud Storage URL
that honours HTTP range requests, so this reads the zip's central directory remotely and
downloads only:

    Explanation of GT files.pdf
    VIS_Onshore/ and VIS_Onboard/ ObjectGT/*.mat
    the Videos/*.avi those .mat files annotate

(40 videos, ~3.5 GB), several at a time. Each member is inflated and checked against the
CRC-32 and size in the zip directory. Already-complete files are skipped, so it can be
re-run after an interruption.

    python src/data/fetch_smd.py
"""
import io
import os
import re
import subprocess
import sys
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/mmichelli/singapore-maritime-dataset"
OUT = Path(__file__).resolve().parents[2] / "data" / "smd_raw"
WORKERS = 4
CHUNK = 1 << 20


def signed_url():
    """Kaggle hands out a short-lived storage URL; ask for a fresh one per member."""
    r = subprocess.run(["curl", "-s", "-A", "Mozilla/5.0", "-o", os.devnull, "-w", "%{redirect_url}", KAGGLE],
                       capture_output=True, text=True, timeout=60)
    if not r.stdout.startswith("https://"):
        raise RuntimeError("no redirect from Kaggle: %r" % r.stdout)
    return r.stdout.strip()


def fetch_range(url, start, end, dest=None):
    """Bytes [start, end] inclusive, into memory or appended to dest (resuming if partial)."""
    if dest is None:
        r = subprocess.run(["curl", "-sf", "--retry", "5", "-r", "%d-%d" % (start, end), url], capture_output=True,
                           timeout=600)
        if r.returncode or len(r.stdout) != end - start + 1:
            raise RuntimeError("range %d-%d: curl exit %d, got %d bytes" % (start, end, r.returncode, len(r.stdout)))
        return r.stdout
    have = dest.stat().st_size if dest.exists() else 0
    if have < end - start + 1:
        with open(dest, "ab") as f:
            subprocess.run(["curl", "-sf", "--retry", "5", "-r", "%d-%d" % (start + have, end), url], stdout=f,
                           timeout=7200, check=True)
    if dest.stat().st_size != end - start + 1:
        raise RuntimeError("%s: %d of %d bytes" % (dest.name, dest.stat().st_size, end - start + 1))


class RangeFile(io.RawIOBase):
    """Just enough of a seekable file for zipfile to read the central directory."""

    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.size, self.pos + n)
        if end <= self.pos:
            return b""
        data = fetch_range(self.url, self.pos, end - 1)
        self.pos += len(data)
        return data


def wanted(infos):
    """The GT explanation, VIS ObjectGT files, and the videos they annotate."""
    gt = [i for i in infos if re.match(r"VIS_On(shore|board)/.*/ObjectGT/.*_ObjectGT\.mat$", i.filename)]
    stems = {Path(i.filename).name[:-len("_ObjectGT.mat")] for i in gt}
    videos = [i for i in infos if i.filename.endswith(".avi") and "/Videos/" in i.filename
              and Path(i.filename).stem in stems]
    assert len(videos) == len(stems), "ObjectGT without a video"
    pdf = [i for i in infos if i.filename == "Explanation of GT files.pdf"]
    return pdf + gt + videos


def target(info):
    # "VIS_Onshore/VIS_Onshore/Videos/x.avi" -> "VIS_Onshore/Videos/x.avi"
    parts = info.filename.split("/")
    if len(parts) > 2 and parts[0] == parts[1]:
        parts = parts[1:]
    return OUT.joinpath(*parts)


def fetch(info):
    dest = target(info)
    if dest.exists() and dest.stat().st_size == info.file_size:
        return info.filename, "skip"
    url = signed_url()
    head = fetch_range(url, info.header_offset, info.header_offset + 29)
    n, e = int.from_bytes(head[26:28], "little"), int.from_bytes(head[28:30], "little")
    start = info.header_offset + 30 + n + e
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".deflate")
    fetch_range(url, start, start + info.compress_size - 1, part)
    d, crc, size = zlib.decompressobj(-15), 0, 0
    with open(part, "rb") as src, open(dest, "wb") as out:
        while chunk := src.read(CHUNK):
            data = d.decompress(chunk) if info.compress_type == zipfile.ZIP_DEFLATED else chunk
            out.write(data)
            crc, size = zlib.crc32(data, crc), size + len(data)
        if info.compress_type == zipfile.ZIP_DEFLATED:
            data = d.flush()
            out.write(data)
            crc, size = zlib.crc32(data, crc), size + len(data)
    if crc != info.CRC or size != info.file_size:
        dest.unlink()
        raise RuntimeError("%s: CRC/size mismatch" % info.filename)
    part.unlink()
    return info.filename, "%.1f MB" % (info.file_size / 1e6)


def main():
    url = signed_url()
    h = subprocess.run(["curl", "-sI", url], capture_output=True, text=True, timeout=60).stdout
    size = int(re.search(r"(?im)^content-length:\s*(\d+)", h).group(1))
    infos = wanted(zipfile.ZipFile(RangeFile(url, size)).infolist())
    print("%d members, %.2f GB compressed" % (len(infos), sum(i.compress_size for i in infos) / 1e9), flush=True)
    failed = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futures = [ex.submit(fetch, i) for i in infos]
        for f in futures:
            try:
                print("%s  %s" % f.result(), flush=True)
            except Exception as err:  # keep going; a re-run resumes
                failed += 1
                print("FAILED", err, flush=True)
    print("done, %d failed" % failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
