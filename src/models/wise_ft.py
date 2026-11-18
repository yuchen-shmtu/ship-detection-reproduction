"""Weight-space interpolation between a detector's COCO weights and its SeaShips fine-tune
(WiSE-FT, Wortsman et al., CVPR 2022) for the RT-DETR-family runs of rtdetr_official.py.

For a run <variant>_<protocol>_s<seed> and each alpha, writes the run
runs/seaships/<variant>-wise<100*alpha:03d>_<protocol>_s<seed>/ whose weights/best.pt holds
the EMA weights

    alpha * fine-tuned + (1 - alpha) * COCO

for every floating-point tensor the two checkpoints share. The 6-class heads (enc_score_head,
dec_score_head, denoising_class_embed) have no COCO counterpart of their shape and stay
fine-tuned, as do integer buffers. The COCO weights are those the run was fine-tuned from
(rtdetr_official.VARIANTS), so the model and its cost are the base run's. wise.json records
the base run, the COCO file, alpha and the tensors kept. Such a run is evaluated like any
other (eval_detector.py, type_rescore.py --fuse, compare_runs.py, tabulate_runs.py);
rtdetr_official.VARIANTS serves the -wise<NNN> names.

--val scores each new run, and the base run as alpha 1, on the protocol's val split: 6-class
COCO mAP50 / mAP50-95 with det_eval (cached as eval/pred6_<split>_640.csv), written to
wise.json and runs/seaships/wise_val_<run>.json. Alpha is chosen on this in-domain val only,
never on SMD or ABOships.

    python src/models/wise_ft.py --run deim-r18-nomix_session_s0 --alphas 0.7 0.8 0.9 --val
"""
import argparse
import json
import sys
from pathlib import Path

import torch

SRC_PARTS = [str(Path(__file__).resolve().parents[1] / p) for p in ("data", "models", "evaluation")]
sys.path[:0] = [p for p in SRC_PARTS if p not in sys.path]  # the src/ modules import one another by name
import rtdetr_official  # noqa: E402
from eval_detector import RUNS, class_ap, image_sizes, seaships_files  # noqa: E402


def coco_state(variant):
    s = torch.load(rtdetr_official.ROOT / "weights" / rtdetr_official.VARIANTS[variant][2], map_location="cpu",
                   weights_only=False)
    return s["ema"]["module"] if "ema" in s else s["model"]


def interpolate(ft, coco, alpha):
    sd, kept = {}, []
    for k, v in ft.items():
        c = coco.get(k)
        if c is not None and c.shape == v.shape and v.is_floating_point():
            sd[k] = alpha * v + (1 - alpha) * c.to(v.dtype)
        else:
            sd[k] = v
            kept.append(k)
    return sd, kept


def val_score(run, protocol):
    import type_rescore

    split = protocol + "_val"
    images = sorted(Path(f).stem for f in seaships_files(split))
    r = class_ap(type_rescore.detector_pred6(run, split), images, image_sizes(images, split), split)
    torch.cuda.empty_cache()
    return dict(map50=100 * r["map50"], map=100 * r["map"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="fine-tuned run, e.g. deim-r18-nomix_session_s0")
    ap.add_argument("--alphas", type=float, nargs="+", required=True)
    ap.add_argument("--val", action="store_true", help="score each run on the protocol's val split")
    a = ap.parse_args()
    variant, protocol, seed = a.run.rsplit("_", 2)
    ft = torch.load(RUNS / a.run / "weights" / "best.pt", map_location="cpu", weights_only=False)["ema"]["module"]
    coco = coco_state(variant)
    val = {}
    if a.val:
        (RUNS / a.run / "eval").mkdir(exist_ok=True)
        val["1.0"] = val_score(a.run, protocol)
        print(a.run, "alpha 1.0 val", val["1.0"], flush=True)
    for alpha in a.alphas:
        run = "%s-wise%03d_%s_%s" % (variant, round(100 * alpha), protocol, seed)
        w = RUNS / run / "weights"
        info = dict(base=a.run, coco=rtdetr_official.VARIANTS[variant][2], alpha=alpha)
        if not (w / "best.pt").exists():
            w.mkdir(parents=True, exist_ok=True)
            sd, kept = interpolate(ft, coco, alpha)
            torch.save({"ema": {"module": sd}}, w / "best.pt")
            info["kept_finetuned"] = kept
        else:
            info = json.loads((RUNS / run / "wise.json").read_text())
        if a.val:
            (RUNS / run / "eval").mkdir(exist_ok=True)
            info["val"] = val[str(alpha)] = val_score(run, protocol)
            print(run, "val", info["val"], flush=True)
        (RUNS / run / "wise.json").write_text(json.dumps(info, indent=1))
    if a.val:
        (RUNS / ("wise_val_%s.json" % a.run)).write_text(json.dumps(val, indent=1))


if __name__ == "__main__":
    main()
