"""Second-stage ship-type recognizer: crop classifiers trained on a protocol's GT crops and
compared on its val/test crops by accuracy, macro-F1 and calibration.
SeaShips crops come from data/crops/crops.csv (src/data/build_crops.py), the
other datasets' from data/crops/<dataset>/crops.csv (src/data/build_crops_multi.py); the classes are
the protocol's (ship_data.py).

Arms (backbones from weights/torch_hub/, fetched 2026-10-04):
  dinov2-s-probe, dinov2-b-probe
                frozen DINOv2 ViT-S/14 or ViT-B/14 (official torch.hub code and weights);
                features = CLS token and mean patch token, standardised; a multinomial
                logistic regression (sklearn), C picked on val by NLL
  convnext-t-ft ImageNet ConvNeXt-T (torchvision) fine-tuned end to end
  dinov2-s-ft   DINOv2 ViT-S/14 fine-tuned end to end
  dinov2-b-ft   DINOv2 ViT-B/14 fine-tuned end to end (2026-10-20; use --lr-body 1e-5)
Fine-tuning: AdamW (lr 1e-4 backbone, 1e-3 head, wd 0.05), cosine over --epochs, batch 32,
AMP, label smoothing 0.1, horizontal flip and mild colour jitter; the epoch with the lowest
val NLL is kept. Every arm's logits then get one temperature fitted on val (Guo et al. 2017).

Input: the crop letterboxed (aspect kept, centred, zero padding) into a wide canvas, since
SeaShips crops are wide (w/h median 4.4, 5-95% 2.0-9.1): 448x126 for DINOv2 (multiples of
its 14-px patch), 448x128 for ConvNeXt; ImageNet mean/std.

Output: runs/types/<arm>_<protocol>_s<seed>/
    metrics.json        val and test: accuracy, macro-F1, NLL and ECE (15 bins) before and
                        after temperature scaling; the temperature; per-class recall
    probs_<split>.csv   crop file, label, the 6 calibrated probabilities
    model.pt / probe.pkl
Only the GT crops: applying an arm to a detector's boxes is src/models/type_rescore.py.

Options added 2026-10-20 for the type-stage pilots (recorded in metrics.json, which type_rescore.py
reads): --gt-crops NAME takes the GT crops from data/NAME/ (e.g. crops_p25, cut with --pad 0.25 by
build_crops_multi.py; --pad must match); --det-crops RUN adds the crops of a detector's own boxes
(src/data/mine_det_crops.py) to train and val, positives with their GT class and, with neg in
--det-kinds, negatives as an extra last class "background"; test stays GT crops only; --lr-body sets
the backbone's lr; --tag is appended to the arm in the run name.

    python src/models/type_classifier.py --arm dinov2-s-probe --protocol session
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SRC_PARTS = [str(Path(__file__).resolve().parents[1] / p) for p in ("data", "models", "evaluation")]
sys.path[:0] = [p for p in SRC_PARTS if p not in sys.path]  # the src/ modules import one another by name
import ship_data  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CROPS = ROOT / "data" / "crops"
SPLITS = ROOT / "data" / "seaships" / "splits"
OUT = ROOT / "runs" / "types"
HUB = ROOT / "weights" / "torch_hub"
CLASSES = ["bulk cargo carrier", "container ship", "fishing boat", "general cargo ship", "ore carrier",
           "passenger ship"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CANVAS = {"dinov2": (448, 126), "convnext": (448, 128)}  # (w, h)
torch.hub.set_dir(str(HUB))


def letterbox(im, w, h):
    """PIL image -> float tensor (3, h, w): aspect kept, centred, zero padding, normalised."""
    s = min(w / im.width, h / im.height)
    nw, nh = max(1, round(im.width * s)), max(1, round(im.height * s))
    canvas = Image.new("RGB", (w, h))
    canvas.paste(im.convert("RGB").resize((nw, nh), Image.BILINEAR), ((w - nw) // 2, (h - nh) // 2))
    x = torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]


class Crops(Dataset):
    """Crop files (paths under data/crops/ or absolute) and labels (class index, -1 if none)."""

    def __init__(self, files, labels, canvas, train=False):
        self.files, self.labels, self.canvas, self.train = list(files), list(labels), canvas, train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        im = Image.open(CROPS / self.files[i] if not Path(self.files[i]).is_absolute() else self.files[i])
        if self.train:
            if np.random.rand() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            arr = np.asarray(im.convert("RGB"), dtype=np.float32)
            arr = arr * np.random.uniform(0.8, 1.2) + np.random.uniform(-20, 20)  # contrast, brightness
            im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        return letterbox(im, *self.canvas), self.labels[i]


def classes(protocol):
    return list(ship_data.names(protocol).values())


def split_crops(protocol, split, gt_dir="crops", names=None):
    ids = set(ship_data.split_ids("%s_%s" % (protocol, split)))
    ds = ship_data.dataset(protocol)
    root = ROOT / "data" / gt_dir
    c = pd.read_csv(root / ("crops.csv" if ds == "seaships" else ds + "/crops.csv"), dtype={"image": str})
    c = c[(c.dataset == ds) & c.image.isin(ids)].reset_index(drop=True)
    names = names or classes(protocol)
    prefix = "" if gt_dir == "crops" else "../%s/" % gt_dir  # Crops opens files relative to data/crops/
    return [prefix + f for f in c.file], [names.index(lab) for lab in c.label]


def det_crops(run, split, kinds, names):
    """Crops of a detector's boxes (src/data/mine_det_crops.py) for one split: files, labels."""
    c = pd.read_csv(CROPS / "_det" / run / "crops.csv", dtype={"image": str})
    c = c[(c.split == split) & c.kind.isin(kinds)]
    return c.file.tolist(), [names.index(lab) for lab in c.label]


def convnext_tiny():
    import torchvision

    m = torchvision.models.convnext_tiny(weights="IMAGENET1K_V1")
    m.classifier[2] = nn.Identity()
    return m


def dinov2_local(name):
    """The official code and weights, as fetched into weights/torch_hub/ (no network)."""
    return torch.hub.load(str(HUB / "facebookresearch_dinov2_main"), name, source="local")


class Net(nn.Module):
    """Backbone features -> linear head. DINOv2: CLS and mean patch token, concatenated."""

    def __init__(self, arm, n_classes=len(CLASSES)):
        super().__init__()
        self.dino = arm.startswith("dinov2")
        if self.dino:
            self.body = dinov2_local("dinov2_vits14" if arm.startswith("dinov2-s") else "dinov2_vitb14")
            dim = 2 * self.body.embed_dim
        else:
            self.body = convnext_tiny()
            dim = 768
        self.head = nn.Linear(dim, n_classes)

    def features(self, x):
        if self.dino:
            f = self.body.forward_features(x)
            return torch.cat([f["x_norm_clstoken"], f["x_norm_patchtokens"].mean(1)], 1)
        return self.body(x)

    def forward(self, x):
        return self.head(self.features(x))


def ece(probs, labels, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == labels[m]).mean() - conf[m].mean())
    return float(e)


def scores(logits, labels, t=1.0, names=CLASSES):
    from sklearn.metrics import f1_score
    p = torch.softmax(torch.as_tensor(logits) / t, 1).numpy()
    lab = np.asarray(labels)
    return dict(accuracy=float((p.argmax(1) == lab).mean()), macro_f1=float(f1_score(lab, p.argmax(1), average="macro")),
                nll=float(-np.log(p[np.arange(len(lab)), lab] + 1e-12).mean()), ece=ece(p, lab),
                recall={c: float((p.argmax(1)[lab == k] == k).mean()) for k, c in enumerate(names)
                        if (lab == k).any()}), p


def fit_temperature(logits, labels):
    lg, lab = torch.as_tensor(logits, dtype=torch.float32), torch.as_tensor(labels)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / log_t.exp(), lab)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


@torch.no_grad()
def extract(net, files, labels, canvas, logits=False, batch=64, workers=2):
    """Backbone features (probe arms) or logits (fine-tuned arms) of every crop, in order."""
    net.eval()
    out = []
    loader = DataLoader(Crops(files, labels, canvas), batch_size=batch, num_workers=workers,
                        timeout=600 if workers else 0)
    for x, _ in loader:
        with torch.autocast("cuda"):
            out.append((net if logits else net.features)(x.cuda()).float().cpu())
    return torch.cat(out).numpy()


def run_probe(arm, data, out, n_classes=len(CLASSES)):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    net = Net(arm, n_classes).cuda()
    canvas = CANVAS["dinov2"]
    feats = {s: extract(net, *data[s], canvas) for s in data}
    sc = StandardScaler().fit(feats["train"])
    best = None
    for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        lr = LogisticRegression(C=c, max_iter=3000).fit(sc.transform(feats["train"]), data["train"][1])
        nll = scores(lr.decision_function(sc.transform(feats["val"])), data["val"][1])[0]["nll"]
        print("probe C=%g val NLL %.4f" % (c, nll), flush=True)
        if best is None or nll < best[0]:
            best = (nll, c, lr)
    _, c, lr = best
    (out / "probe.pkl").write_bytes(pickle.dumps(dict(scaler=sc, probe=lr, C=c)))
    return {s: lr.decision_function(sc.transform(feats[s])) for s in ("val", "test")}, dict(C=c)


def run_finetune(arm, data, out, epochs, seed, workers, n_classes=len(CLASSES), lr_body=1e-4):
    net = Net(arm, n_classes).cuda()
    canvas = CANVAS["dinov2" if arm.startswith("dinov2") else "convnext"]
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(Crops(*data["train"], canvas, train=True), batch_size=32, shuffle=True, drop_last=True,
                        num_workers=workers, persistent_workers=workers > 0, generator=g,
                        timeout=600 if workers else 0)
    opt = torch.optim.AdamW([{"params": net.body.parameters(), "lr": lr_body}, {"params": net.head.parameters(), "lr": 1e-3}],
                            weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(loader))
    scaler = torch.amp.GradScaler("cuda")
    best, log = None, []
    for ep in range(epochs):
        net.train()
        t0, tot = time.time(), 0.0
        for x, y in loader:
            with torch.autocast("cuda"):
                loss = F.cross_entropy(net(x.cuda()), y.cuda(), label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        val = extract(net, *data["val"], canvas, logits=True, workers=workers)
        m = scores(val, data["val"][1])[0]
        log.append(dict(epoch=ep + 1, train_loss=tot / len(loader), val_acc=m["accuracy"], val_nll=m["nll"],
                        minutes=(time.time() - t0) / 60))
        print(log[-1], flush=True)
        if best is None or m["nll"] < best:
            best = m["nll"]
            torch.save(net.state_dict(), out / "model.pt")
    pd.DataFrame(log).to_csv(out / "results.csv", index=False)
    net.load_state_dict(torch.load(out / "model.pt", map_location="cuda"))
    return {s: extract(net, *data[s], canvas, logits=True, workers=workers) for s in ("val", "test")}, dict(epochs=epochs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["dinov2-s-probe", "dinov2-b-probe", "convnext-t-ft", "dinov2-s-ft",
                                                     "dinov2-b-ft"])
    ap.add_argument("--protocol", default="session", choices=[p for p in ship_data.PROTOCOLS if p != "uploader"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gt-crops", default="crops", help="directory under data/ holding the GT crops")
    ap.add_argument("--pad", type=float, default=0.125, help="padding the GT crops were cut with (for type_rescore)")
    ap.add_argument("--det-crops", help="detector run whose box crops (src/data/mine_det_crops.py) join train and val")
    ap.add_argument("--det-kinds", default="pos,neg", help="kinds of detector crops to add: pos, neg")
    ap.add_argument("--lr-body", type=float, default=1e-4)
    ap.add_argument("--tag", default="", help="appended to the arm in the run name, e.g. -bg")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = OUT / ("%s%s_%s_s%d" % (a.arm, a.tag, a.protocol, a.seed))
    out.mkdir(parents=True)  # fails if the run exists
    kinds = a.det_kinds.split(",") if a.det_crops else []
    names = classes(a.protocol) + (["background"] if "neg" in kinds else [])
    data = {s: split_crops(a.protocol, s, a.gt_crops, names) for s in ("train", "val", "test")}
    for s in ("train", "val") if a.det_crops else ():
        f, lab = det_crops(a.det_crops, s, kinds, names)
        data[s] = (data[s][0] + f, data[s][1] + lab)
    print({s: len(v[0]) for s, v in data.items()}, flush=True)
    if a.arm.endswith("probe"):
        logits, info = run_probe(a.arm, data, out, len(names))
    else:
        logits, info = run_finetune(a.arm, data, out, a.epochs, a.seed, a.workers, len(names), a.lr_body)
    t = fit_temperature(logits["val"], data["val"][1])
    res = dict(arm=a.arm, protocol=a.protocol, seed=a.seed, temperature=t, classes=names, pad=a.pad,
               background="background" in names, gt_crops=a.gt_crops, det_crops=a.det_crops, det_kinds=kinds,
               lr_body=a.lr_body, **info)
    for s in ("val", "test"):
        raw, _ = scores(logits[s], data[s][1], names=names)
        cal, p = scores(logits[s], data[s][1], t, names=names)
        res[s] = dict(raw=raw, calibrated=cal)
        pd.DataFrame(p, columns=names).assign(file=data[s][0], label=[names[k] for k in data[s][1]]).to_csv(
            out / ("probs_%s.csv" % s), index=False)
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({s: {k: round(v, 4) for k, v in res[s]["calibrated"].items() if k != "recall"} for s in ("val", "test")}))


if __name__ == "__main__":
    main()
