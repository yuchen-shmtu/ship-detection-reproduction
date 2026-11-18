"""End-to-end latency of a detector plus the type stage, batch 1, frames already decoded (2026-10-20).

One process per configuration (DEIM's and RT-DETR's code bases cannot share a process). For a
detector run and its dataset's crop arm, on --images test frames drawn at random (seed 0) and kept in
memory, after --warmup frames, each frame is timed with CUDA syncs in two parts:
  - detector: for a DETR-family run the deploy-mode model as rtdetr_official.predict runs it (warp to
    640 x 640, forward, the official top-300 postprocessor) or, with --slice, slice_eval.sliced_dets
    (full frame + 2 x 2 tiles at 0.6, merge); for a YOLO run ultralytics' predict (letterbox 640, its
    NMS) at conf --thr; fp32 throughout, as src/evaluation/model_cost.py;
  - type stage: the boxes whose best class score is >= --thr (the deployment threshold, see
    src/evaluation/type_threshold.py) are cropped as type_rescore.DetCrops does, letterboxed, classified by
    the arm (fp16 autocast, one batch per frame), softmaxed with the arm's temperature and fused
    with the detector's scores at w = 0.5; frames without such boxes cost only the grouping.
Written to runs/types/speed/<run>[-sl2]__<arm>.json (mean and median ms per frame, boxes re-typed per
frame, the python processes running alongside). Run with the GPU idle.

Before timing, the process asks Windows for full speed (full_speed): HIGH priority class, power
throttling of execution speed and timer resolution off, and affinity to the performance cores, as a
latency-critical deployment would. Without it, Windows 11 throttles the bursty CPU side of batch-1
inference: launched hidden through WMI our detector took 22.5 ms per frame instead of 9.3, and even
from the interactive shell 10.2 instead of 9.5 (2026-10-20; priority,
affinity or the throttling opt-out each removes most of it, the 1 ms timer does not). On other
systems full_speed is skipped.

    python src/evaluation/e2e_speed.py --run deim-r18-nomix-wise090_smdplus_s0 --arm convnext-t-ft_smdplus_s0
"""
import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SRC_PARTS = [str(Path(__file__).resolve().parents[1] / p) for p in ("data", "models", "evaluation")]
sys.path[:0] = [p for p in SRC_PARTS if p not in sys.path]  # the src/ modules import one another by name
import ship_data  # noqa: E402
import type_classifier as tc  # noqa: E402
import type_rescore as tr  # noqa: E402
from type_threshold import crop_box  # noqa: E402

RUNS = Path(__file__).resolve().parents[2] / "runs" / "seaships"


def full_speed():
    """HIGH priority, no power throttling, performance cores only (Windows); returns what was set."""
    from ctypes import wintypes

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]

    class Throttling(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    h = k.GetCurrentProcess()
    # performance cores = logical processors of the highest efficiency class (SYSTEM_CPU_SET_INFORMATION:
    # Size at 0, LogicalProcessorIndex at 14, EfficiencyClass at 18)
    n = wintypes.ULONG(0)
    k.GetSystemCpuSetInformation(None, 0, ctypes.byref(n), None, 0)
    buf = (ctypes.c_byte * n.value)()
    k.GetSystemCpuSetInformation(buf, n, ctypes.byref(n), None, 0)
    raw, off, cls = bytes(buf), 0, {}
    while off < n.value:
        cls[raw[off + 14]] = raw[off + 18]
        off += int.from_bytes(raw[off:off + 4], "little")
    top = max(cls.values())
    mask = sum(1 << lp for lp, c in cls.items() if c == top)
    st = Throttling(1, 0x1 | 0x4, 0)  # execution speed and timer resolution: controlled, not throttled
    return dict(priority_high=bool(k.SetPriorityClass(h, 0x80)), affinity=hex(mask) if k.SetProcessAffinityMask(h, mask) else None,
                no_throttling=bool(k.SetProcessInformation(h, 4, ctypes.byref(st), ctypes.sizeof(st))))


def group(xyxy, score, cls, k):
    """(query, class) or (anchor, class) dets -> boxes (x, y, w, h) with a class-score vector."""
    uniq, inv = np.unique(np.round(xyxy, 2), axis=0, return_inverse=True)
    s = np.zeros((len(uniq), k), np.float32)
    np.maximum.at(s, (inv.ravel(), cls.astype(int)), score)
    return np.column_stack([uniq[:, :2], uniq[:, 2:] - uniq[:, :2]]), s


class Row:
    __slots__ = ("x", "y", "w", "h")

    def __init__(self, b):
        self.x, self.y, self.w, self.h = map(float, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="detector run, e.g. deim-r18-nomix-wise090_smdplus_s0 or yolov8n_wsodd_s0")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--slice", action="store_true", help="DETR-family: sliced inference (2 x 2 tiles at 0.6)")
    ap.add_argument("--thr", type=float, default=0.05)
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--images", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    a = ap.parse_args()
    speed_settings = full_speed() if sys.platform == "win32" else {}
    protocol = a.run.split("_")[-2]
    split = protocol + "_test"
    k = ship_data.num_classes(split)
    variant = a.run.rsplit("_", 2)[0]
    weights = RUNS / a.run / "weights" / "best.pt"
    files = sorted(tr.seaships_files(split))
    pick = np.random.default_rng(0).choice(len(files), min(len(files), a.images + a.warmup), replace=False)
    frames = [Image.open(files[i]).convert("RGB") for i in pick]
    for im in frames:
        im.load()
    yolo = variant.startswith("yolo")
    if yolo:
        import os

        os.environ["YOLO_AUTOINSTALL"] = "False"
        from ultralytics import YOLO

        model = YOLO(str(weights))
        bgr = [np.ascontiguousarray(np.asarray(im)[:, :, ::-1]) for im in frames]
    else:
        import rtdetr_official
        import slice_eval

        model = rtdetr_official.load(variant, weights)
        chw = [torch.from_numpy(np.asarray(im).transpose(2, 0, 1).copy()) for im in frames]
    logits, canvas, t = tr.load_arm(a.arm)
    dev = torch.device("cuda")

    @torch.no_grad()
    def detect(i):
        if yolo:
            r = model.predict(bgr[i], imgsz=640, conf=a.thr, verbose=False, device=0)[0].boxes
            return r.xyxy.cpu().numpy(), r.conf.cpu().numpy(), r.cls.cpu().numpy()
        if a.slice:
            d = slice_eval.sliced_dets(model, chw[i], 2, 0.6)
            return d[:, :4], d[:, 4], d[:, 5]
        im = chw[i]
        x = F.interpolate(im.to(dev)[None].float().div_(255), size=(640, 640), mode="bilinear", antialias=True)
        lab, box, sco = model(x, torch.tensor([[im.shape[-1], im.shape[-2]]], device=dev))
        return box[0].cpu().numpy(), sco[0].cpu().numpy(), lab[0].cpu().numpy()

    @torch.no_grad()
    def type_stage(i, xyxy, score, cls):
        boxes, s = group(xyxy, score, cls, k)
        sel = s.max(1) >= a.thr
        if not sel.any():
            return 0
        crops = [tc.letterbox(crop_box(frames[i], Row(b)), *canvas) for b in boxes[sel]]
        with torch.autocast("cuda"):
            lg = logits(torch.stack(crops).to(dev))
        p = torch.softmax(torch.as_tensor(lg, dtype=torch.float32) / t, 1).cpu().numpy()[:, :k]  # drop a background class
        smax = s[sel].max(1, keepdims=True)
        _ = np.maximum(s[sel], 1e-3 * smax) ** (1 - a.w) * (smax * p) ** a.w  # the fused class scores
        return int(sel.sum())

    det_ms, type_ms, n_boxes = [], [], []
    for i in range(len(frames)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        d = detect(i)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        n = type_stage(i, *d)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if i >= a.warmup:
            det_ms.append(1000 * (t1 - t0))
            type_ms.append(1000 * (t2 - t1))
            n_boxes.append(n)
    det_ms, type_ms = np.array(det_ms), np.array(type_ms)
    others = subprocess.run(["powershell", "-NoProfile", "-Command",
                             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { $_.CommandLine }"],
                            capture_output=True, text=True).stdout.splitlines() if sys.platform == "win32" else []
    res = dict(run=a.run, arm=a.arm, slice=a.slice, thr=a.thr, w=a.w, split=split, n_images=len(det_ms),
               process=speed_settings,
               det_ms=float(det_ms.mean()), det_ms_median=float(np.median(det_ms)), type_ms=float(type_ms.mean()),
               type_ms_median=float(np.median(type_ms)), total_ms=float((det_ms + type_ms).mean()),
               fps=float(1000 / (det_ms + type_ms).mean()), boxes_per_image=float(np.mean(n_boxes)),
               gpu=torch.cuda.get_device_name(0), torch=torch.__version__, time=time.strftime("%Y-%m-%d %H:%M"),
               other_python=[c.strip() for c in others if c.strip() and "e2e_speed" not in c][:20])
    out = tc.OUT / "speed"
    out.mkdir(parents=True, exist_ok=True)
    name = "%s__%s.json" % (a.run.replace("_" + protocol, ("-sl2" if a.slice else "") + "_" + protocol, 1), a.arm)
    (out / name).write_text(json.dumps(res, indent=1))
    print(json.dumps({k2: (round(v, 2) if isinstance(v, float) else v) for k2, v in res.items() if k2 != "other_python"}),
          flush=True)


if __name__ == "__main__":
    main()
