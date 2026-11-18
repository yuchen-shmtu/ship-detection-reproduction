"""Params, GFLOPs and batch-1 latency per detector architecture, as the lightweight papers report.

Reads the best.pt of one run per model (session_s0 if it exists, else the first run found;
the 6-class head, and cost does not depend on protocol or seed) and writes
runs/seaships/cost.csv:

  params_m, gflops  ultralytics' fused model summary (Conv+BN folded; NMS-free models such as
                    YOLOv10 and YOLO26 drop their training-only one-to-many head), GFLOPs at
                    640x640, i.e. the numbers ultralytics prints after training
  *_ms, fps         end-to-end model.predict on the 284 cross_test images (32 per call, see
                    latency()), batch 1, imgsz 640,
                    conf 0.25: preprocess + inference + postprocess (NMS where the model has
                    it), ms per image from ultralytics' own timers; the first pass is warm-up
                    and the mean of the next --passes passes is kept. fp32 and fp16 (half=True).
                    YOLO letterboxes the 1138x640 frames to 384x640; RT-DETR always takes 640x640

The torchvision models (tv_detector.py) and the RT-DETR-family variants (rtdetr_official.py)
are measured the same way where they allow it:
  params_m          parameters of the model as tested: RT-DETR-R18 after the official
                    deploy(), which folds each RepVGG block into one conv (deploy() keeps the
                    unfused branches, which are not counted; 20.09 M before deploy);
                    torchvision's FrozenBatchNorm holds buffers, not parameters
  gflops            torch.utils.flop_counter at 640x640 (SSD300 at 300x300, its only size)
                    on a real frame, 2 FLOPs per multiply-add as ultralytics; counts conv and
                    matmul only, so Faster R-CNN's RoIAlign and NMS are left out
  *_ms, fps         the same 284 frames at the size the evals use (Faster R-CNN long side 640,
                    SSD 300x300, RT-DETR-R18 640x640), timed with CUDA syncs:
                    pre = upload + scaling (+ the warp for RT-DETR-R18), inf = the model's
                    forward, which for torchvision includes its own resize and per-class NMS,
                    post = copy of the dets to the host. Score threshold 0.25 as above. fp16 is
                    autocast (ultralytics' half=True casts the weights instead)

Latency depends on the GPU and on what else runs: run it with the GPU idle (not next to
training). Measured on the RTX 4070 Ti SUPER. On 2026-10-04 it also depended on the order:
with all 284 frames in one predict call, RT-DETR-l reserved 28.7 GB of GPU memory and ran at
137 ms, and the nano YOLOs measured after it in the same process took 6-10 ms fp32 instead
of 0.5-0.8 (a later RT-DETR-l load failed with CUDA out of memory). Hence the chunks in
latency(). And YOLO26n fp16 measured after fp32 in the same process drifted from 0.5 to
1.1 ms over three passes, against a steady 0.29 ms alone, with memory flat. So each model
and precision runs in its own process.

    python src/evaluation/model_cost.py --models yolo11n yolo26n rtdetr-l
    python src/evaluation/model_cost.py --models faster-rcnn-r50 ssd300-vgg16 rtdetr-r18
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

os.environ["YOLO_AUTOINSTALL"] = "False"
from ultralytics import RTDETR, YOLO  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_num_params  # noqa: E402

SRC_PARTS = [str(Path(__file__).resolve().parents[1] / p) for p in ("data", "models", "evaluation")]
sys.path[:0] = [p for p in SRC_PARTS if p not in sys.path]  # the src/ modules import one another by name
import rtdetr_official  # noqa: E402
import tv_detector  # noqa: E402
from eval_detector import RUNS, seaships_files  # noqa: E402


def latency(weights, cls, files, half, passes, chunk=32):
    """Per-image pre / inference / post ms. Frames go to predict `chunk` at a time: within one
    call ultralytics holds GPU memory per frame until the call ends (~12 MB YOLOv8n, ~85 MB
    RT-DETR-l), so all 284 at once overflowed the 16 GB for RT-DETR-l (28.7 GB reserved) and
    WDDM paged it to host memory: 137 ms instead of ~9."""
    model = cls(str(weights))
    out = []
    for k in range(passes + 1):
        sp = [r.speed for i in range(0, len(files), chunk)
              for r in model.predict(files[i:i + chunk], imgsz=640, conf=0.25, batch=1, half=half, device=0,
                                     verbose=False, stream=True)]
        if k:
            out.append([np.mean([s[x] for s in sp]) for x in ("preprocess", "inference", "postprocess")])
    return np.mean(out, 0)


def torch_model(m, weights):
    """Model, per-frame preprocessing and the forward call of a non-ultralytics model."""
    if m in rtdetr_official.VARIANTS:
        model, size = rtdetr_official.load(m, weights), rtdetr_official.SIZE

        def pre(im):
            x = F.interpolate(im.cuda()[None].float().div_(255), size=(size, size), mode="bilinear", antialias=True)
            return x, torch.tensor([[im.shape[-1], im.shape[-2]]], device="cuda")

        return model, pre, lambda x: model(*x)
    model = tv_detector.load(m, weights)
    if isinstance(model, tv_detector.detection.FasterRCNN):
        model.roi_heads.score_thresh = 0.25
    else:
        model.score_thresh = 0.25
    return model, lambda im: [im.cuda().float().div_(255)], model


def deployed_params(model):
    """Parameters the forward uses: RT-DETR's RepVGG blocks keep their unfused conv1/conv2 after
    deploy(); DEIM's VGGBlock drops them, so only existing branches are excluded."""
    unused = {id(p) for mod in model.modules() if hasattr(mod, "convert_to_deploy") and hasattr(mod, "conv")
              for branch in (getattr(mod, "conv1", None), getattr(mod, "conv2", None)) if branch is not None
              for p in branch.parameters()}
    return sum(p.numel() for p in model.parameters() if id(p) not in unused)


def torch_flops(m, fwd, pre, im):
    size = 300 if m == "ssd300-vgg16" else 640
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        fwd(pre(F.interpolate(im[None].float(), size=(size, size), mode="bilinear")[0].round().byte()))
    return fc.get_total_flops() / 1e9


@torch.no_grad()
def torch_latency(fwd, pre, ims, half, passes):
    out = []
    for k in range(passes + 1):
        t = []
        for im in ims:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            x = pre(im)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            with torch.autocast("cuda", enabled=half):
                y = fwd(x)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            torch.utils._pytree.tree_map(lambda v: v.cpu().numpy() if torch.is_tensor(v) else v, y)
            t.append((t1 - t0, t2 - t1, time.perf_counter() - t2))
        if k:
            out.append(np.mean(t, 0) * 1000)
    return np.mean(out, 0)


def run_weights(m):
    runs = sorted(RUNS.glob(m + "_*_s*"), key=lambda r: (r.name != m + "_session_s0", r.name))
    return next(r / "weights" / "best.pt" for r in runs if (r / "weights" / "best.pt").exists())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="model stems, e.g. yolo11n rtdetr-l")
    ap.add_argument("--precision", nargs="+", choices=["fp32", "fp16"], default=["fp32", "fp16"])
    ap.add_argument("--passes", type=int, default=3)
    a = ap.parse_args()
    if len(a.models) > 1 or len(a.precision) > 1:  # a fresh process per model and precision: see the docstring
        jobs = [(m, p) for m in a.models for p in a.precision]
        failed = [j for j in jobs if subprocess.run([sys.executable, __file__, "--models", j[0], "--precision", j[1],
                                                     "--passes", str(a.passes)]).returncode]
        print("failed:", failed or "none", flush=True)
        sys.exit(1 if failed else 0)
    files = seaships_files("cross_test")
    (m,), (tag,) = a.models, a.precision
    weights = run_weights(m)
    if m in tv_detector.ARCHS or m in rtdetr_official.VARIANTS:
        ims = [tv_detector.read_image(f) for f in files]
        model, pre_fn, fwd = torch_model(m, weights)
        row = dict(model=m, params_m=deployed_params(model) / 1e6, gflops=torch_flops(m, fwd, pre_fn, ims[0]))
        pre, inf, post = torch_latency(fwd, pre_fn, ims, tag == "fp16", a.passes)
    else:
        cls = RTDETR if m.startswith("rtdetr") else YOLO
        net = cls(str(weights))
        net.fuse()
        row = dict(model=m, params_m=get_num_params(net.model) / 1e6, gflops=get_flops(net.model, 640))
        pre, inf, post = latency(weights, cls, files, tag == "fp16", a.passes)
    row.update({"weights": weights.parent.parent.name, "%s_pre_ms" % tag: pre, "%s_inf_ms" % tag: inf,
                "%s_post_ms" % tag: post, "%s_fps" % tag: 1000 / (pre + inf + post)})
    print({k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    df = pd.DataFrame([row]).set_index("model")
    out = RUNS / "cost.csv"
    if out.exists():  # keep other models' rows and this model's other precision
        df = df.combine_first(pd.read_csv(out).set_index("model"))
    cols = ["params_m", "gflops", "weights"] + ["%s_%s" % (t, k) for t in ("fp32", "fp16")
                                                for k in ("pre_ms", "inf_ms", "post_ms", "fps")]
    df = df.reindex(columns=cols).round(3)
    df.to_csv(out)
    print(df.to_string(), flush=True)


if __name__ == "__main__":
    main()
