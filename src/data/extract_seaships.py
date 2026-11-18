"""Extract the SeaShips7k Kaggle mirror into data/seaships7k (images + labels + data.yaml only)."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data"
SRC = ROOT / "_raw" / "seaships7k.zip"
DST = ROOT / "seaships7k"
PREFIX = "Seaships7000-1/"


def main():
    n = 0
    with zipfile.ZipFile(SRC) as z:
        for info in z.infolist():
            if not info.filename.startswith(PREFIX) or info.is_dir():
                continue
            rel = info.filename[len(PREFIX):]
            if rel.endswith(".cache"):
                continue
            out = DST / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(info))
            n += 1
    print("extracted", n, "files to", DST)
    for split in ("train", "test"):
        print(split, len(list((DST / split / "images").glob("*.jpg"))), "images,",
              len(list((DST / split / "labels").glob("*.txt"))), "labels")


if __name__ == "__main__":
    main()
