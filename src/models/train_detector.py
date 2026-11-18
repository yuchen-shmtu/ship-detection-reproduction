"""Train one detector on one protocol: a SeaShips split or another dataset (ship_data.py).

Reads the protocol's yolo/<protocol>.yaml (data/seaships/yolo/ for SeaShips) and
weights/<model> (COCO-pretrained), writes
runs/seaships/<model stem>_<protocol>_s<seed>/ (ultralytics output; weights/best.pt is
picked on the protocol's own val split). Testing is done separately by eval_detector.py.

One recipe for every ultralytics baseline: ultralytics defaults, 100 epochs, imgsz 640
(long side, so the 1138x640 pool trains at 640x360), batch 16, 4 workers, deterministic.
Run from the repo root: the AMP check loads weights/yolo26n.pt through the relative
weights_dir setting.

rtdetr-* weights go through ultralytics' RTDETR class (its own trainer, loss and square
640 letterbox); everything else through YOLO. optimizer=auto picks AdamW lr 0.001 for
every model here, RT-DETR included (the official RT-DETR recipe uses lr 1e-4).

RT-DETR-l trains at batch 8: at 16 it ran out of GPU memory at the start of epoch 2 (12.7 GB
allocated on the 16 GB card, where Windows keeps part of it). Ultralytics accumulates
gradients to a nominal batch of 64 (nbs), so an optimizer step still sees 64 images, and
lr, weight decay and warmup are unchanged; only BatchNorm sees 8 images instead of 16.

--model faster-rcnn-r50 / ssd300-vgg16 (no .pt) trains a torchvision model with its own
reference recipe instead (tv_detector.py), and the RT-DETR-family variants (rtdetr-r18,
rtdetr-r18-o365, rtdetrv2-s, deim-r18) their official code and recipe (rtdetr_official.py); --epochs and --batch override the recipe,
--imgsz does not apply.

Workers are part of the recipe (each worker has its own augmentation RNG stream) and are
capped by memory, not speed. The val loader gets 2x workers, and on Windows every spawned
worker commits ~0.7-0.8 GB. With 8 workers one run committed ~25 GB of the 36 GB commit
limit, and a second job next to it killed the run with an allocation error. Run one
training at a time.

    python src/models/train_detector.py --model yolo11n.pt --protocol session --seed 0
"""
import argparse
import os
import sys
from pathlib import Path

os.environ["YOLO_AUTOINSTALL"] = "False"  # never let ultralytics pip-install into the environment
SRC_PARTS = [str(Path(__file__).resolve().parents[1] / p) for p in ("data", "models", "evaluation")]
sys.path[:0] = [p for p in SRC_PARTS if p not in sys.path]  # the src/ modules import one another by name
import ship_data  # noqa: E402
import tv_detector  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "seaships"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--protocol", required=True, choices=[p for p in ship_data.PROTOCOLS if p != "uploader"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, help="default 100; the reference schedule for torchvision models")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, help="default 16; 8 for rtdetr-*; the reference batch for torchvision models")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="non-ultralytics models only: first N train and val images (smoke tests)")
    a = ap.parse_args()

    name = "%s_%s_s%d" % (Path(a.model).stem, a.protocol, a.seed)
    if a.model in tv_detector.ARCHS:
        return tv_detector.train(a.model, a.protocol, a.seed, a.workers, RUNS / name, a.epochs, a.batch, a.limit)
    import rtdetr_official

    if a.model in rtdetr_official.VARIANTS:
        return rtdetr_official.train(a.model, a.protocol, a.seed, a.workers, RUNS / name, a.epochs, a.batch, a.limit)
    from ultralytics import RTDETR, YOLO  # here: spawned dataloader workers re-import this module

    cls = RTDETR if name.startswith("rtdetr") else YOLO
    batch = a.batch or (8 if cls is RTDETR else 16)
    cls(str(ROOT / "weights" / a.model)).train(
        data=str(ship_data.yolo_yaml(a.protocol)), epochs=a.epochs or 100,
        imgsz=a.imgsz, batch=batch, workers=a.workers, seed=a.seed, deterministic=True, device=0, project=str(RUNS),
        name=name, exist_ok=False)


if __name__ == "__main__":
    main()
