"""torchvision detectors on SeaShips: Faster R-CNN R50-FPN and SSD300-VGG16, COCO-pretrained.

The two classic baselines of the benchmark (Faster R-CNN is in 5 of the 15 JMSE result
tables, SSD in 6). train_detector.py and eval_detector.py call this for --model
faster-rcnn-r50 / ssd300-vgg16. Runs go to runs/seaships/<arch>_<protocol>_s<seed>/ as for
ultralytics: weights/{best,last}.pt (state dicts), results.csv (one row per epoch),
args.yaml.

Models are torchvision's own builders with their COCO_V1 weights (fasterrcnn_resnet50_fpn,
COCO box AP 37.0; ssd300_vgg16, 25.1), read from weights/torch_hub/checkpoints/ (the
sha256 prefix is in the file name), with a new 7-way classifier (6 types + background).
The builders freeze what torchvision's COCO fine-tune freezes: FrozenBatchNorm and the
stem + layer1 of ResNet-50; the first block of VGG-16 (which has no BatchNorm).

Recipe: torchvision's detection reference recipe per model (references/detection/README),
on one GPU by the linear lr rule, with AMP, and a linear warmup from 1/1000 over the first
min(1000, iters - 1) iterations of epoch 0:
  faster-rcnn-r50  SGD momentum 0.9, batch 8, lr 0.01 (reference: 0.02 at batch 16),
                   wd 1e-4, 26 epochs, x0.1 at 16 and 22; horizontal flip. Input long side
                   640, as the ultralytics runs: the 1138x640 pool goes in at 640x360,
                   padded to 384
  ssd300-vgg16     SGD momentum 0.9, batch 32 (reference: 8 GPUs x 4), lr 0.002, wd 5e-4,
                   120 epochs, x0.1 at 80 and 110; the reference "ssd" augmentation
                   (photometric distort, zoom-out, IoU crop, flip). Input always warped to
                   300x300
best.pt is picked on the protocol's own val split by 6-class COCO mAP50-95 (det_eval), as
ultralytics picks by its own mAP50-95.

Test time: score threshold 0.001 and at most 300 dets per image, as in the ultralytics
evals. Each model keeps its own per-class NMS (IoU 0.5 Faster R-CNN, 0.45 SSD); the
class-agnostic dets are those merged across classes by NMS at 0.7, as ultralytics'
agnostic NMS. Faster R-CNN takes any long side (SMD at 640 and 1280); SSD300 only 300x300.

Determinism as ultralytics' deterministic=True: seeded, cudnn deterministic,
torch.use_deterministic_algorithms(warn_only=True). roi_align's backward has no
deterministic CUDA kernel, so Faster R-CNN runs are not bit-reproducible. train() keeps that
kernel on Linux too, where torchvision would otherwise swap in a torch.compile'd roi_align.

--limit N (train_detector.py) keeps the first N train and val images, for smoke tests.
"""
import math
import os
import random
import re
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # for use_deterministic_algorithms
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import tv_tensors  # noqa: E402
from torchvision.io import decode_jpeg, read_file  # noqa: E402
from torchvision.models import detection  # noqa: E402
from torchvision.models.detection import _utils as det_utils  # noqa: E402
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # noqa: E402
from torchvision.models.detection.ssd import SSDClassificationHead  # noqa: E402
from torchvision.ops import nms  # noqa: E402
from torchvision.transforms import v2  # noqa: E402

SRC_PARTS = [str(Path(__file__).resolve().parents[1] / p) for p in ("data", "models", "evaluation")]
sys.path[:0] = [p for p in SRC_PARTS if p not in sys.path]  # the src/ modules import one another by name

ROOT = Path(__file__).resolve().parents[2]
torch.hub.set_dir(str(ROOT / "weights" / "torch_hub"))
ARCHS = {
    "faster-rcnn-r50": dict(batch=8, lr=0.01, wd=1e-4, epochs=26, steps=[16, 22], size=640, any_size=True),
    "ssd300-vgg16": dict(batch=32, lr=0.002, wd=5e-4, epochs=120, steps=[80, 110], size=300, any_size=False),
}
MEAN = (123, 117, 104)  # the reference's zoom-out fill
# s per batch. A worker whose queue feeder fails (error 1455, commit limit: 2026-10-03, SSD
# session s2) stays alive, so without a timeout the main process waits for its batch forever
LOADER_TIMEOUT = 600


def build(arch, n_cls=6):
    """COCO-pretrained torchvision model with a new n_cls-type (+ background) classifier."""
    if arch == "faster-rcnn-r50":
        m = detection.fasterrcnn_resnet50_fpn(weights="COCO_V1", min_size=640, max_size=640)
        m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, n_cls + 1)
    else:
        m = detection.ssd300_vgg16(weights="COCO_V1")
        m.head.classification_head = SSDClassificationHead(
            det_utils.retrieve_out_channels(m.backbone, (300, 300)), m.anchor_generator.num_anchors_per_location(),
            n_cls + 1)
    return m


def load(arch, weights):
    """A trained run's model; the number of types is read off its classifier (6 for SeaShips)."""
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    if arch == "faster-rcnn-r50":
        n_cls = sd["roi_heads.box_predictor.cls_score.weight"].shape[0] - 1
    else:  # SSD300's first head layer predicts (n_cls + 1) scores for each of its 4 anchors
        n_cls = sd["head.classification_head.module_list.0.weight"].shape[0] // 4 - 1
    m = build(arch, n_cls)
    m.load_state_dict(sd)
    return m.cuda().eval()


def read_image(path):
    return decode_jpeg(read_file(str(path)))  # uint8, 3 x H x W


class Pool(Dataset):
    """A dataset's images with their boxes: xyxy pixels, labels 1..n (0 is background)."""

    def __init__(self, ids, tf, root):
        self.ids, self.tf, self.root = ids, tf, root

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        im = tv_tensors.Image(read_image(self.root / "images" / (self.ids[i] + ".jpg")))
        h, w = im.shape[-2:]
        lab = np.array([ln.split() for ln in (self.root / "labels" / (self.ids[i] + ".txt")).read_text().splitlines()],
                       float).reshape(-1, 5)
        cx, cy, bw, bh = lab[:, 1] * w, lab[:, 2] * h, lab[:, 3] * w, lab[:, 4] * h
        boxes = torch.as_tensor(np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1), dtype=torch.float32)
        tgt = dict(boxes=tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(h, w)),
                   labels=torch.as_tensor(lab[:, 0], dtype=torch.int64) + 1)
        im, tgt = self.tf(im, tgt)
        return im, dict(boxes=tgt["boxes"].as_subclass(torch.Tensor), labels=tgt["labels"])


class Frames(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return read_image(self.files[i]), Path(self.files[i]).stem


def collate(batch):
    return tuple(zip(*batch))


def seed_worker(_):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def transforms(arch, train):
    tail = [v2.ClampBoundingBoxes(), v2.SanitizeBoundingBoxes(min_size=1), v2.ToDtype(torch.float32, scale=True)]
    if arch == "ssd300-vgg16":
        aug = [v2.RandomPhotometricDistort(p=0.5), v2.RandomZoomOut(fill={tv_tensors.Image: MEAN, "others": 0}),
               v2.RandomIoUCrop(), v2.RandomHorizontalFlip(0.5)]
    else:
        aug = [v2.RandomHorizontalFlip(0.5)]
    return v2.Compose((aug if train else []) + tail)


def frames_loader(files, workers=2, batch=8):
    return DataLoader(Frames(files), batch_size=batch, num_workers=workers, persistent_workers=workers > 0,
                      collate_fn=collate, timeout=LOADER_TIMEOUT if workers else 0)


@torch.no_grad()
def predict(model, loader, size):
    """Per-class dets as image, x, y, w, h, score, pred_cls (0..5), plus image sizes (h, w).

    size is the long side for Faster R-CNN; SSD300 ignores it.
    """
    model.eval()
    if isinstance(model, detection.FasterRCNN):
        model.transform.min_size, model.transform.max_size = (size,), size
        model.roi_heads.score_thresh, model.roi_heads.detections_per_img = 0.001, 300
    else:
        model.score_thresh, model.detections_per_img = 0.001, 300
    rows, sizes = [], {}
    for ims, stems in loader:
        out = model([im.cuda(non_blocking=True).float().div_(255) for im in ims])
        for im, stem, o in zip(ims, stems, out):
            sizes[stem] = tuple(im.shape[-2:])
            b, s, c = o["boxes"].cpu().numpy(), o["scores"].cpu().numpy(), o["labels"].cpu().numpy() - 1
            rows += [(stem, x1, y1, x2 - x1, y2 - y1, sc, int(k)) for (x1, y1, x2, y2), sc, k in zip(b, s, c)]
    return pd.DataFrame(rows, columns=["image", "x", "y", "w", "h", "score", "pred_cls"]), sizes


def agnostic(dt, iou=0.7, max_det=300):
    """Per-class dets merged across classes by NMS, as ultralytics' agnostic NMS; cls 0."""
    keep = []
    for _, d in dt.groupby("image", sort=False):
        b = torch.as_tensor(d[["x", "y", "w", "h"]].to_numpy(), dtype=torch.float32)
        b[:, 2:] += b[:, :2]
        k = nms(b, torch.as_tensor(d.score.to_numpy(), dtype=torch.float32), iou)[:max_det]
        keep.append(d.index.to_numpy()[k.numpy()])
    return dt.loc[np.concatenate(keep) if keep else []].assign(cls=0)


def train(arch, protocol, seed, workers, out, epochs=None, batch=None, limit=None):
    import ship_data
    from eval_detector import class_ap, split_ids  # here, so dataloader workers do not import it

    cfg = ARCHS[arch]
    epochs, batch = epochs or cfg["epochs"], batch or cfg["batch"]
    lr = cfg["lr"] * batch / cfg["batch"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.use_deterministic_algorithms(True, warn_only=True)
    warnings.filterwarnings("ignore", message="roi_align_backward_kernel does not have a deterministic")  # every epoch
    # with deterministic algorithms on, torchvision 0.21 replaces roi_align's CUDA kernel by a
    # torch.compile'd Python version wherever compile is supported (Linux with triton, never
    # Windows); keep the CUDA kernel everywhere, as on Windows (on a Linux machine with gcc 4.8 triton
    # could not build its helper module); import_module, since torchvision.ops.roi_align as
    # an attribute is the function, not the module
    import importlib

    importlib.import_module("torchvision.ops.roi_align").is_compile_supported = lambda device_type: False
    # with AMP, GradScaler skips the first steps while it calibrates, so warmup steps lead optimizer steps
    warnings.filterwarnings("ignore", message=re.escape("Detected call of `lr_scheduler.step()` before"))

    out.mkdir(parents=True)  # fails if the run exists
    (out / "weights").mkdir()
    args = dict(arch=arch, protocol=protocol, seed=seed, workers=workers, epochs=epochs, batch=batch, lr=lr,
                momentum=0.9, weight_decay=cfg["wd"], lr_steps=cfg["steps"], warmup="linear 1e-3, epoch 0",
                amp=True, size=cfg["size"], limit=limit)
    (out / "args.yaml").write_text(yaml.safe_dump(args, sort_keys=False))
    train_ids, val_ids = split_ids(protocol + "_train"), split_ids(protocol + "_val")
    if limit:
        train_ids, val_ids = train_ids[:limit], val_ids[:limit]
    g = torch.Generator()
    g.manual_seed(seed)
    root = ship_data.root(protocol)
    loader = DataLoader(Pool(train_ids, transforms(arch, True), root), batch_size=batch, shuffle=True, drop_last=True,
                        num_workers=workers, persistent_workers=workers > 0, collate_fn=collate,
                        worker_init_fn=seed_worker, generator=g, timeout=LOADER_TIMEOUT if workers else 0)
    val_loader = frames_loader([str(root / "images" / (i + ".jpg")) for i in val_ids], min(2, workers))

    model = build(arch, ship_data.num_classes(protocol)).cuda()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9,
                          weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, cfg["steps"], 0.1)
    scaler = torch.amp.GradScaler("cuda")
    print("%s: %d train / %d val images, %d iters per epoch, %d epochs" % (out.name, len(train_ids), len(val_ids),
                                                                            len(loader), epochs), flush=True)
    log, best = [], -1.0
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        warm = torch.optim.lr_scheduler.LinearLR(opt, 1e-3, total_iters=min(1000, len(loader) - 1)) if ep == 0 else None
        sums = {}
        for it, (ims, tgts) in enumerate(loader):
            ims = [im.cuda(non_blocking=True) for im in ims]
            tgts = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in tgts]
            with torch.autocast("cuda"):
                losses = model(ims, tgts)
            loss = sum(losses.values())
            if not math.isfinite(loss.item()):
                sys.exit("loss %s at epoch %d iter %d: %s" % (loss.item(), ep + 1, it, losses))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if warm:
                warm.step()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.item()
        lr_ep = opt.param_groups[0]["lr"]  # at the end of the epoch, after warmup
        sched.step()
        t1 = time.time()
        dt, sizes = predict(model, val_loader, cfg["size"])
        ap = class_ap(dt, sorted(sizes), sizes, protocol + "_val")
        log.append(dict(epoch=ep + 1, lr=lr_ep, **{k: v / len(loader) for k, v in sums.items()},
                        val_map50=ap["map50"], val_map=ap["map"], train_min=(t1 - t0) / 60,
                        val_min=(time.time() - t1) / 60))
        pd.DataFrame(log).to_csv(out / "results.csv", index=False)
        torch.save(model.state_dict(), out / "weights" / "last.pt")
        if ap["map"] > best:
            best = ap["map"]
            torch.save(model.state_dict(), out / "weights" / "best.pt")
        print(" ".join("%s %.4g" % kv for kv in log[-1].items()), flush=True)
    print("best val mAP50-95 %.4f at epoch %d" % (best, max(log, key=lambda r: r["val_map"])["epoch"]), flush=True)
