"""RT-DETR-family detectors on SeaShips with their official code: RT-DETR v1 / v2
(lyuwenyu/RT-DETR, rtdetrv2_pytorch) and DEIM (Intellindust-AI-Lab/DEIM, a fork of it).

RT-DETR is in 7 of the 15 JMSE result tables, mostly as R18, the lightweight size;
ultralytics ships only -l/-x. train_detector.py, eval_detector.py and model_cost.py call this
for the model names in VARIANTS. Runs go to runs/seaships/<variant>_<protocol>_s<seed>/: the
official output (log.txt, one JSON line per epoch with the val COCO stats; summary/,
tensorboard) plus weights/{best,last}.pt, the official checkpoints moved there after training
(full solver state; the model is state["ema"]["module"]).

Variants — all a PResNet-18vd backbone with 3 decoder layers (~20 M params), each fine-tuned
from COCO weights with its own official COCO recipe, as the torchvision baselines keep theirs:
  rtdetr-r18       RT-DETR v1 R18 (configs/rtdetr/rtdetr_r18vd_6x_coco.yml), from
                   rtdetr_r18vd_dec3_6x_coco_from_paddle.pth (lyuwenyu/storage v0.1, AP 46.5);
                   72 epochs, the recipe below
  rtdetr-r18-o365  the same model and recipe from the Objects365-pretrained weights
                   rtdetr_r18vd_5x_coco_objects365_from_paddle.pth (v0.1, COCO AP 49.2)
  rtdetrv2-s       RT-DETRv2-S (configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml), from
                   rtdetrv2_r18vd_120e_coco_rerun_48.1.pth (v0.2, AP 48.1); 120 epochs,
                   augmentation and multi-scale off from epoch 117
  deim-r18         DEIM on the RT-DETRv2-S model (DEIM configs/deim_rtdetrv2/
                   deim_r18vd_120e_coco.yml), from DEIM's own COCO weights
                   deim_rtdetrv2_r18vd_120e_coco.pth (README "S" row, Google Drive, AP 49.0):
                   Mosaic and MixUp in epochs 4-64, the MAL loss, flat-cosine lr 2e-4, SiLU
                   decoder, 120 epochs. At epoch 117 DEIM reloads its best stage-1 checkpoint
                   and keeps a stage-2 one only if that beats it; best.pt is the later of the two.
  deim-r18-nomix   deim-r18 with Mosaic and MixUp off (src/models/configs/deim_r18vd_nomix_seaships.yml)
  deim-r18-nomix-fdi, -mpd
                   deim-r18-nomix plus one of our modules (src/models/boat_detr_modules.py):
                   fine-detail injection, or the MPDIoU box loss
  deim-r18-nomix-blr  deim-r18-nomix with the backbone at 0.1x the lr, as RT-DETR v1/v2
  deim-r18-nomix-v2init  deim-r18-nomix's recipe on RT-DETRv2-S's architecture and COCO weights
  deim-r18-nomix-vfl  deim-r18-nomix with RT-DETRv2's VFL loss instead of MAL
  deim-r18-nomix-cp   deim-r18-nomix plus box-level copy-paste of small objects (SmallCopyPaste in
                   src/models/boat_detr_modules.py; src/models/configs/deim_r18vd_nomix_cp_seaships.yml), 2026-10-22
DEIM's code is patched by src/models/patch_deim.py (calflops optional, torchvision 0.21 transforms).

Model (rtdetr-r18): the official RT-DETR v1 R18 (PResNet-18vd, 3 decoder layers) fine-tuned
from its COCO weights. -t (tuning) loads every tensor whose shape matches, i.e. all but the
80-class heads.

Recipe: the official 6x recipe unchanged, run by the official tools/train.py and solver:
AdamW lr 1e-4 (backbone 1e-5), wd 1e-4 (none on norms and biases), 72 epochs at constant lr
after a 2,000-iteration linear warmup, EMA 0.9999, grad clip 0.1, AMP, batch 16. Train
augmentation: photometric distort, zoom-out, IoU crop, flip, then a square 640x640 warp and
a per-batch random rescale to 480-800. Val/test: warped to 640x640, so the 1138x640 pool
is squashed and not letterboxed. best.pt is picked by the solver on the protocol's own val
split by COCO mAP50-95 (faster-coco-eval), as ultralytics picks by its mAP50-95.

Changed from the official config (src/models/configs/rtdetr_r18vd_seaships.yml): 6 classes without
the COCO category remap; the backbone's ImageNet download is off (the COCO checkpoint
carries it); checkpoint_freq 1000 (the official 1 keeps a ~250 MB checkpoint every epoch);
loader workers 4 train / 2 val (8 val workers commit ~6 GB of host memory on Windows).
The official code seeds but sets no deterministic flags, so runs are not reproducible.

Test time: the official postprocessor keeps the top 300 (query, class) pairs by sigmoid
score, without NMS, so a query can give boxes of several types: these are the per-class
dets. The class-agnostic dets merge them by NMS at 0.7, as for the torchvision models.
SMD is tested at 640x640 only.

Needs faster-coco-eval (the official dataset and evaluator import it). The official
dataset also builds torchvision's CocoDetection, which imports pycocotools; that is not
installed, so the official train.py is launched after faster_coco_eval.init_as_pycocotools(),
its drop-in replacement (the official evaluator uses faster-coco-eval anyway).

--limit N keeps the first N train and val images (annotation files written into the run
dir), for smoke tests.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

SRC_PARTS = [str(Path(__file__).resolve().parents[1] / p) for p in ("data", "models", "evaluation")]
sys.path[:0] = [p for p in SRC_PARTS if p not in sys.path]  # the src/ modules import one another by name

ROOT = Path(__file__).resolve().parents[2]
RTDETR_CODE = ROOT / "third_party" / "RT-DETR-main" / "rtdetrv2_pytorch"
DEIM_CODE = ROOT / "third_party" / "DEIM-main"
TRAIN_PY = {RTDETR_CODE: RTDETR_CODE / "tools" / "train.py", DEIM_CODE: DEIM_CODE / "train.py"}
# model name -> (code, config in src/models/configs/, COCO weights in weights/); see the docstring
VARIANTS = {
    "rtdetr-r18": (RTDETR_CODE, "rtdetr_r18vd_seaships.yml", "rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"),
    "rtdetr-r18-o365": (RTDETR_CODE, "rtdetr_r18vd_seaships.yml", "rtdetr_r18vd_5x_coco_objects365_from_paddle.pth"),
    "rtdetrv2-s": (RTDETR_CODE, "rtdetrv2_r18vd_seaships.yml", "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"),
    "deim-r18": (DEIM_CODE, "deim_r18vd_seaships.yml", "deim_rtdetrv2_r18vd_120e_coco.pth"),
    "deim-r18-nomix": (DEIM_CODE, "deim_r18vd_nomix_seaships.yml", "deim_rtdetrv2_r18vd_120e_coco.pth"),
    "deim-r18-nomix-fdi": (DEIM_CODE, "deim_r18vd_nomix_fdi_seaships.yml", "deim_rtdetrv2_r18vd_120e_coco.pth"),
    "deim-r18-nomix-mpd": (DEIM_CODE, "deim_r18vd_nomix_mpd_seaships.yml", "deim_rtdetrv2_r18vd_120e_coco.pth"),
    "deim-r18-nomix-blr": (DEIM_CODE, "deim_r18vd_nomix_blr_seaships.yml", "deim_rtdetrv2_r18vd_120e_coco.pth"),
    "deim-r18-nomix-v2init": (DEIM_CODE, "deim_r18vd_nomix_v2arch_seaships.yml", "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"),
    "deim-r18-nomix-vfl": (DEIM_CODE, "deim_r18vd_nomix_vfl_seaships.yml", "deim_rtdetrv2_r18vd_120e_coco.pth"),
    "deim-r18-nomix-cp": (DEIM_CODE, "deim_r18vd_nomix_cp_seaships.yml", "deim_rtdetrv2_r18vd_120e_coco.pth"),
}
# WiSE-FT runs (src/models/wise_ft.py): <variant>-wise<NNN> is <variant>'s model with interpolated weights
VARIANTS.update({"%s-wise%03d" % (v, a): VARIANTS[v] for v in list(VARIANTS) for a in range(5, 100, 5)})
SIZE = 640
LOADER_TIMEOUT = 600  # s per batch, as tv_detector.LOADER_TIMEOUT
# python -c LAUNCH train.py args...: runs the official train.py with faster_coco_eval as pycocotools
LAUNCH = ("import runpy, sys, faster_coco_eval; faster_coco_eval.init_as_pycocotools(); "
          "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')")
# DEIM code: the same, after registering our modules (src/models/boat_detr_modules.py) into its workspace
LAUNCH_DEIM = LAUNCH.replace("sys.argv = ", "sys.path.append(%r); import boat_detr_modules; sys.argv = "
                             % str(Path(__file__).resolve().parent))


def limited(src, n, dst):
    d = json.loads(src.read_text())
    d["images"] = d["images"][:n]
    keep = {im["id"] for im in d["images"]}
    d["annotations"] = [a for a in d["annotations"] if a["image_id"] in keep]
    dst.write_text(json.dumps(d))
    return dst


def train(variant, protocol, seed, workers, out, epochs=None, batch=None, limit=None, extra=()):
    """extra: more -u overrides (smoke tests, e.g. an early DEIM stage switch). The protocol's
    dataset (ship_data.py) gives the COCO files, the image folder and num_classes."""
    import ship_data

    code, config, weights = VARIANTS[variant]
    out.mkdir(parents=True)  # fails if the run exists
    ann = {s: ship_data.coco_file("%s_%s" % (protocol, s)) for s in ("train", "val")}
    if limit:
        ann = {s: limited(f, limit, out / ("ann_%s.json" % s)) for s, f in ann.items()}
    images = ship_data.root(protocol) / "images"
    upd = ["train_dataloader.dataset.img_folder=" + images.as_posix(),
           "train_dataloader.dataset.ann_file=" + ann["train"].as_posix(),
           "val_dataloader.dataset.img_folder=" + images.as_posix(),
           "val_dataloader.dataset.ann_file=" + ann["val"].as_posix(),
           "num_classes=%d" % ship_data.num_classes(protocol),
           "train_dataloader.num_workers=%d" % workers, "val_dataloader.num_workers=%d" % min(2, workers)]
    if workers:  # without it a failed worker hangs the run: see tv_detector.LOADER_TIMEOUT
        upd += ["train_dataloader.timeout=%d" % LOADER_TIMEOUT, "val_dataloader.timeout=%d" % LOADER_TIMEOUT]
    if epochs:
        upd.append("epoches=%d" % epochs)
    if batch:
        upd += ["train_dataloader.total_batch_size=%d" % batch, "val_dataloader.total_batch_size=%d" % batch]
    upd += list(extra)
    cmd = [sys.executable, "-c", LAUNCH_DEIM if code == DEIM_CODE else LAUNCH, str(TRAIN_PY[code]), "-c", str(ROOT / "src" / "models" / "configs" / config),
           "-t", str(ROOT / "weights" / weights), "--seed", str(seed), "--use-amp", "--output-dir", str(out), "-u", *upd]
    print(" ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=code).returncode
    if rc:
        sys.exit(rc)
    (out / "weights").mkdir()
    # DEIM writes best_stg1.pth and, if stage 2 beats it, best_stg2.pth; RT-DETR writes best.pth
    best = next(out / f for f in ("best.pth", "best_stg2.pth", "best_stg1.pth") if (out / f).exists())
    shutil.move(str(best), str(out / "weights" / "best.pt"))
    shutil.move(str(out / "last.pth"), str(out / "weights" / "last.pt"))
    stats = [json.loads(ln) for ln in (out / "log.txt").read_text().splitlines()]
    best = max(stats, key=lambda s: s["test_coco_eval_bbox"][0])
    print("best val mAP50-95 %.4f at epoch %d" % (best["test_coco_eval_bbox"][0], best["epoch"] + 1), flush=True)


class Deployed(torch.nn.Module):
    def __init__(self, model, post):
        super().__init__()
        self.model, self.post = model.deploy(), post.deploy()

    def forward(self, images, orig_sizes):
        return self.post(self.model(images), orig_sizes)


def load(variant, weights, device="cuda"):
    """The official model + postprocessor (deploy mode) with the EMA weights of a run; the number
    of classes is read off the checkpoint's class head (6 for SeaShips runs).

    One code base per process: RT-DETR's package is `src`, DEIM's is `engine`."""
    code, config, _ = VARIANTS[variant]
    sys.path.insert(0, str(code))
    if code == DEIM_CODE:
        from engine.core import YAMLConfig

        import boat_detr_modules  # noqa: F401  registers our modules
    else:
        from src.core import YAMLConfig

    state = torch.load(weights, map_location="cpu", weights_only=False)["ema"]["module"]
    cfg = YAMLConfig(str(ROOT / "src" / "models" / "configs" / config), num_classes=state["decoder.enc_score_head.weight"].shape[0])
    cfg.model.load_state_dict(state)
    return Deployed(cfg.model, cfg.postprocessor).to(device).eval()


@torch.no_grad()
def predict(model, loader, size=SIZE):
    """Per-class dets as image, x, y, w, h, score, pred_cls (0..5), plus image sizes (h, w).

    Images are warped to size x size, as the official val transform (bilinear, antialiased;
    the official one resizes the PIL image).
    """
    dev = next(model.parameters()).device
    rows, sizes = [], {}
    for ims, stems in loader:
        x = torch.cat([F.interpolate(im.to(dev, non_blocking=True)[None].float().div_(255), size=(size, size),
                                     mode="bilinear", antialias=True) for im in ims])
        orig = torch.tensor([[im.shape[-1], im.shape[-2]] for im in ims], device=dev)  # (w, h)
        labels, boxes, scores = model(x, orig)
        for im, stem, lab, box, sco in zip(ims, stems, labels.cpu().numpy(), boxes.cpu().numpy(), scores.cpu().numpy()):
            sizes[stem] = tuple(im.shape[-2:])
            rows += [(stem, x1, y1, x2 - x1, y2 - y1, s, int(k)) for (x1, y1, x2, y2), s, k in zip(box, sco, lab)]
    return pd.DataFrame(rows, columns=["image", "x", "y", "w", "h", "score", "pred_cls"]), sizes
