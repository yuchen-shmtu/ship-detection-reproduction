"""Our detector modules for the RT-DETR-family models, on the DEIM code base.

Importing this module registers the classes below into DEIM's workspace (engine.core.register),
so a config can name them like DEIM's own (`DEIM: {encoder: HybridEncoderFDI}`,
`criterion: DEIMCriterionMPD`). rtdetr_official.py imports it before DEIM's train.py runs and
before a run is loaded for testing; DEIM's code (third_party/DEIM-main/) must be on sys.path.

HybridEncoderFDI   HybridEncoder plus fine-detail injection, for small ships (R18 sits at 65.4
                   small-object AP50 on the session split against YOLO26n's 72.2; SMD and
                   ABOships are mostly small ships). The stride-4 backbone feature (PResNet
                   stage 1, 64 channels for R18) is space-to-depth'd to stride 8 (pixel
                   unshuffle by 2, so no detail is pooled away), projected by a 1x1 conv to
                   hidden_dim, batch-normed and added to the stride-8 input projection before
                   the hybrid encoder's fusion. The BN weight starts at zero, so training starts
                   from the COCO-pretrained model exactly, and every pretrained tensor keeps its
                   name and shape. Needs PResNet return_idx [0, 1, 2, 3]. Cost: one 1x1 conv on
                   the stride-8 map, ~0.8 GFLOPs at 640x640 (+1.4% on R18).
DEIMCriterionMPD   DEIMCriterion with the GIoU box loss replaced by MPDIoU loss (Ma & Xu 2023;
                   used by several ship detectors): 1 - (IoU -
                   d_tl^2 / (w^2 + h^2) - d_br^2 / (w^2 + h^2)), with d_tl, d_br the distances
                   between the top-left and bottom-right corners and w, h the image size, i.e.
                   1 in DEIM's normalised boxes. It keeps the key loss_giou (and its weight), so
                   weight dicts and logs stay as they are; the matcher still costs GIoU.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.core import register
from engine.deim.box_ops import box_cxcywh_to_xyxy
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.hybrid_encoder import HybridEncoder


class _ProjWithDetail(nn.Sequential):
    """The stride-8 input projection plus the encoder's pending detail map. Same children
    (conv, norm) as the projection it replaces, so the pretrained tensors keep their names."""

    def forward(self, x):
        y = super().forward(x)
        detail = self._owner._detail
        return y if detail is None else y + detail


@register()
class HybridEncoderFDI(HybridEncoder):
    __share__ = ['eval_spatial_size', ]

    def __init__(self, c2_channels=64, in_channels=[128, 256, 512], feat_strides=[8, 16, 32], hidden_dim=256,
                 nhead=8, dim_feedforward=1024, dropout=0.0, enc_act='gelu', use_encoder_idx=[2],
                 num_encoder_layers=1, pe_temperature=10000, expansion=0.5, depth_mult=1.0, act='silu',
                 eval_spatial_size=None, version='rt_detrv2'):
        super().__init__(in_channels, feat_strides, hidden_dim, nhead, dim_feedforward, dropout, enc_act,
                         use_encoder_idx, num_encoder_layers, pe_temperature, expansion, depth_mult, act,
                         eval_spatial_size, version)
        self.fdi = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(4 * c2_channels, hidden_dim, kernel_size=1, bias=False)),
            ('norm', nn.BatchNorm2d(hidden_dim)),
        ]))
        nn.init.zeros_(self.fdi.norm.weight)
        proj = _ProjWithDetail(OrderedDict(self.input_proj[0]._modules))
        object.__setattr__(proj, '_owner', self)  # a plain attribute, not a submodule
        self.input_proj[0] = proj
        self._detail = None

    def forward(self, feats):
        c2, feats = feats[0], list(feats[1:])
        self._detail = self.fdi(F.pixel_unshuffle(c2, 2))
        try:
            return super().forward(feats)
        finally:
            self._detail = None


def mpdiou(b1, b2):
    """MPDIoU of matched xyxy boxes in a unit image, (N, 4) x (N, 4) -> (N,)."""
    lt, rb = torch.max(b1[:, :2], b2[:, :2]), torch.min(b1[:, 2:], b2[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(1)
    area1, area2 = (b1[:, 2:] - b1[:, :2]).prod(1), (b2[:, 2:] - b2[:, :2]).prod(1)
    iou = inter / (area1 + area2 - inter).clamp(min=1e-7)
    d_tl = ((b1[:, :2] - b2[:, :2]) ** 2).sum(1)
    d_br = ((b1[:, 2:] - b2[:, 2:]) ** 2).sum(1)
    return iou - (d_tl + d_br) / 2.0  # w^2 + h^2 = 2 for normalised boxes


@register()
class DEIMCriterionMPD(DEIMCriterion):
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {'loss_bbox': F.l1_loss(src_boxes, target_boxes, reduction='none').sum() / num_boxes}
        loss = 1 - mpdiou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        loss = loss if boxes_weight is None else loss * boxes_weight
        losses['loss_giou'] = loss.sum() / num_boxes
        return losses


# ---------------------------------------------------------------- small-object copy-paste (2026-10-22)
import random as _random  # noqa: E402

import torchvision.transforms.v2 as _T  # noqa: E402
from PIL import Image as _Image  # noqa: E402

from engine.data._misc import convert_to_tv_tensor as _to_tv  # noqa: E402


def _feather(w, h, r):
    """An L-mode alpha mask of size (w, h): 255 inside, falling linearly to 0 over r pixels at the border."""
    import numpy as _np
    if r <= 0 or w <= 2 * r or h <= 2 * r:
        return _Image.new("L", (w, h), 255)
    xs, ys = _np.arange(w), _np.arange(h)
    d = _np.minimum.outer(_np.minimum(ys, h - 1 - ys), _np.minimum(xs, w - 1 - xs))
    a = _np.where(d < r, 255.0 * (d + 1) / (r + 1), 255.0)
    return _Image.fromarray(a.astype(_np.uint8), mode="L")


@register()
class SmallCopyPaste(_T.Transform):
    """Box-level copy-paste of small objects, a small-object augmentation without Mosaic's down-scaling of whole
    frames (2026-10-22: with our two modules on every base, DEIM-R18 with Mosaic led our base on SMD-Plus, where
    most ships are tiny, while Mosaic costs transfer). With probability `probability`, up to `n_max` objects whose
    box covers less than `max_area` of their frame are cut from `sources` other training images of the dataset and
    pasted into this one: with its bottom edge near that of a ship already in the frame (ships of a fixed or
    level camera share the waterline band; in a frame without boxes, at the height it had in its source), a random
    x, scaled by a factor drawn from `scale` times the ratio of the two frame widths, with a
    border feathered over `feather` pixels. A paste whose box would touch an existing or already pasted box is
    skipped. Labels are kept; area and iscrowd are filled in. Runs on PIL images before the other augmentations, in
    the same epochs as they (name it in the policy's ops)."""

    def __init__(self, probability=0.5, n_max=6, max_area=0.01, scale=(0.8, 1.25), sources=2, feather=2,
                 min_side=4):
        super().__init__()
        self.probability, self.n_max, self.max_area = probability, n_max, max_area
        self.scale, self.sources, self.feather, self.min_side = tuple(scale), sources, feather, min_side

    @staticmethod
    def _free(b, taken):
        x0, y0, x1, y1 = b
        for t in taken:
            if x0 < t[2] and t[0] < x1 and y0 < t[3] and t[1] < y1:
                return False
        return True

    def forward(self, *inputs):
        if len(inputs) == 1:
            inputs = inputs[0]
        image, target, dataset = inputs
        if self.probability < 1.0 and _random.random() > self.probability:
            return image, target, dataset
        W, H = image.size
        boxes = target["boxes"]
        taken = [list(map(float, b)) for b in boxes.tolist()]
        new_boxes, new_labels = [], []
        image = image.copy()
        for _ in range(self.sources):
            if len(new_boxes) >= self.n_max:
                break
            src, st = dataset.load_item(_random.randrange(len(dataset)))
            sw, sh = src.size
            sb, sl = st["boxes"].tolist(), st["labels"].tolist()
            cand = [i for i, b in enumerate(sb)
                    if (b[2] - b[0]) >= self.min_side and (b[3] - b[1]) >= self.min_side
                    and (b[2] - b[0]) * (b[3] - b[1]) < self.max_area * sw * sh]
            _random.shuffle(cand)
            for i in cand:
                if len(new_boxes) >= self.n_max:
                    break
                x0, y0, x1, y1 = sb[i]
                s = _random.uniform(*self.scale) * W / sw
                cw, ch = max(self.min_side, round((x1 - x0) * s)), max(self.min_side, round((y1 - y0) * s))
                if cw >= W or ch >= H:
                    continue
                if taken:  # rest the paste on the waterline of a ship already in this frame
                    ref = taken[_random.randrange(len(boxes))] if len(boxes) else taken[_random.randrange(len(taken))]
                    bottom = ref[3] + _random.uniform(-0.5, 0.5) * ch
                else:  # an empty frame: the height the object had in its source
                    bottom = y1 / sh * H
                cx = _random.uniform(cw / 2, W - cw / 2)
                px, py = int(round(cx - cw / 2)), int(round(min(max(bottom - ch, 0), H - ch)))
                nb = [float(px), float(py), float(px + cw), float(py + ch)]
                if not self._free(nb, taken):
                    continue
                crop = src.crop((int(x0), int(y0), int(round(x1)), int(round(y1)))).resize((cw, ch), _Image.BILINEAR)
                image.paste(crop, (px, py), _feather(cw, ch, self.feather))
                taken.append(nb)
                new_boxes.append(nb)
                new_labels.append(int(sl[i]))
        if new_boxes:
            nb = torch.tensor(new_boxes, dtype=torch.float32)
            target = dict(target)
            target["boxes"] = _to_tv(torch.cat([boxes.as_subclass(torch.Tensor).float(), nb]), "boxes", box_format="xyxy",
                                     spatial_size=(H, W))
            target["labels"] = torch.cat([target["labels"], torch.tensor(new_labels, dtype=target["labels"].dtype)])
            if "area" in target:
                target["area"] = torch.cat([target["area"], (nb[:, 2] - nb[:, 0]) * (nb[:, 3] - nb[:, 1])])
            if "iscrowd" in target:
                target["iscrowd"] = torch.cat([target["iscrowd"], torch.zeros(len(new_boxes), dtype=target["iscrowd"].dtype)])
        return image, target, dataset
