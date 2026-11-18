"""Patch the DEIM code in third_party/DEIM-main/ for torch 2.6 / torchvision 0.21. Idempotent: re-run
it after re-extracting DEIM (Intellindust-AI-Lab/DEIM, commit 09d35d5, the main branch on 2026-10-04).

1. engine/misc/profiler_utils.py imports calflops, which is not installed (and not needed:
   src/evaluation/model_cost.py measures cost). The import becomes optional; without it the
   training log prints the parameter count only.
2. engine/data/transforms/_transforms.py targets torchvision < 0.21. 0.21 calls
   Transform.make_params / transform, not _get_params / _transform, and has no
   F.get_spatial_size; our rtdetrv2_pytorch copy already has the same three fixes.
3. configs/deim_rtdetrv2/rtdetrv2_r18vd_120e_coco.yml (included by deim_r18vd_120e_coco.yml)
   sets collate_fn.scales: ~, a key of rtdetrv2's collate function that DEIM's
   BatchImageCollateFunction does not take (TypeError). It now takes and ignores scales=None;
   DEIM builds its scales from base_size / base_size_repeat.

    python src/models/patch_deim.py
"""
from pathlib import Path

CODE = Path(__file__).resolve().parents[2] / "third_party" / "DEIM-main"
MARK = "# boat patch (src/models/patch_deim.py)"


def patch(rel, pairs, tail=""):
    p = CODE / rel
    s = p.read_text(encoding="utf-8")
    if "# boat patch (" in s:  # MARK, or the mark of an earlier copy of this script
        print(rel, "already patched")
        return
    for old, new in pairs:
        assert s.count(old) == 1, (rel, old)
        s = s.replace(old, new)
    p.write_text(s + tail, encoding="utf-8")
    print(rel, "patched")


patch("engine/misc/profiler_utils.py", [
    ("from calflops import calculate_flops\n",
     "try:  %s: calflops is optional\n    from calflops import calculate_flops\nexcept ImportError:\n"
     "    calculate_flops = None\n" % MARK),
    ("    model_for_info = copy.deepcopy(cfg.model).deploy()\n",
     "    model_for_info = copy.deepcopy(cfg.model).deploy()\n    if calculate_flops is None:\n"
     "        params = sum(p.numel() for p in model_for_info.parameters())\n"
     "        return params, {\"Model FLOPs: n/a (calflops not installed)   Params:%s\" % params}\n"),
])
patch("engine/data/transforms/_transforms.py", [
    ("        sp = F.get_spatial_size(flat_inputs[0])\n", "        sp = F.get_size(flat_inputs[0])\n"),
    ("        fill = self._fill[type(inpt)]\n", "        fill = T._utils._get_fill(self._fill, type(inpt))\n"),
], tail="""

%s: torchvision >= 0.21 calls make_params / transform
PadToSize.make_params = lambda self, flat_inputs: self._get_params(flat_inputs)
for _cls in (PadToSize, ConvertBoxes, ConvertPILImage):
    _cls.transform = lambda self, inpt, params: self._transform(inpt, params)
""" % MARK)
patch("engine/data/dataloader.py", [
    ("        ema_restart_decay=0.9999,\n",
     "        ema_restart_decay=0.9999,\n        scales=None,  %s: rtdetrv2 configs pass scales: ~\n" % MARK),
])
