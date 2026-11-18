# Ship detection and ship-type recognition under recording-level evaluation

Code, configurations and data splits to reproduce the experiments of the paper *A detector-agnostic ship-type stage
and weight interpolation for real-time maritime detection under recording-level evaluation*.

The method is a real-time DETR with two plug-in modules:

- **Base detector**: DEIM-R18 (DEIM training on the RT-DETRv2-S model, ~20 M parameters) fine-tuned from COCO without
  Mosaic and MixUp (run name `deim-r18-nomix`).
- **Module A, WiSE-FT**: the fine-tuned weights interpolated towards the COCO weights they started from,
  θ = α·θ_ft + (1 − α)·θ_COCO with α = 0.9; no further training.
- **Module B, ship-type stage**: the detector's 100 best boxes per image are cropped (12.5% context per side) and
  classified by a temperature-scaled ConvNeXt-T; its probabilities p_c are fused with the detector's class scores s_c
  as f_c = max(s_c, ε·s_max)^(1−w) · (s_max·p_c)^w with w = 0.5 and ε = 10⁻³. Boxes are not moved.
- **Sliced inference** (the full frame plus 2 × 2 tiles of 0.6 of the frame) only where a validation rule switches it
  on, which is SMD-Plus only.

Every dataset is split by recording: no capture session, video or day spans two of train, val and test. Every model is
trained with three seeds, and comparisons use a two-level bootstrap that resamples test recordings and training seeds.

## Layout

```
ship-detection-reproduction/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/                      # configuration, kept apart from the code
│   └── base.yaml                 # method and evaluation settings, read by scripts/
├── src/
│   ├── models/                   # detectors (recipes in models/configs/), Module A, Module B, sliced inference
│   ├── data/                     # dataset builders and the recording-level split logic
│   └── evaluation/               # COCO AP with ignore regions, testing, recording-level bootstrap, tables
├── scripts/
│   ├── train.sh
│   └── evaluate.sh
├── splits/
│   └── record_level_split.json   # recording-level split manifest of every dataset
└── weights/
    └── README.md                 # where every checkpoint comes from (checkpoints are not tracked)
```

| Folder | Contents |
|---|---|
| `src/data/` | `ship_data.py` (dataset registry and ground-truth helpers). SeaShips: `extract_seaships.py`, `build_seaships_pool.py`, `seaships_osd.py` (camera and time read from the on-screen display, with `osd_*_templates.npz`), `build_seaships_splits.py` (capture sessions and the session split), `audit_seaships_split.py` (leak audit of a split), `build_seaships_coco.py`. SMD, zero-shot: `fetch_smd.py`, `build_smd.py`. `build_smdplus.py`. ABOships: `fetch_aboships.py`, `build_aboships.py`, `build_aboships_splits.py`. `build_mcships.py`. `build_wsodd.py`. Crops for Module B: `build_crops.py`, `build_crops_multi.py`, `mine_det_crops.py`. `split_manifest.py` writes and checks `splits/record_level_split.json`. |
| `src/models/` | `configs/*_seaships.yml` (the detector recipes: the official DEIM / RT-DETR configs plus our changes), `train_detector.py` (trains any detector), `rtdetr_official.py` (RT-DETR-R18, RT-DETRv2-S, DEIM-R18 and the base detector on their official code), `boat_detr_modules.py` (modules registered into DEIM: small-object copy-paste, and two variants that were dropped), `tv_detector.py` (Faster R-CNN, SSD300), `patch_deim.py`, `fetch_weights.py`, `run_suite.py` (a queue of train-and-test jobs), `wise_ft.py` (Module A), `type_classifier.py` and `type_rescore.py` (Module B: classifier, fusion), `type_variants.py` (variants of Module B), `slice_eval.py` (sliced inference). |
| `src/evaluation/` | `det_eval.py` (COCO AP in NumPy with ignore regions; matches pycocotools), `eval_detector.py` (tests a run in-domain and zero-shot), `compare_runs.py` (paired bootstrap over images or recordings), `boot_cache.py` (two-level bootstrap over recordings and seeds), `tabulate_runs.py` (`results.csv`, `results_mean.csv`), `type_confusion.py`, `type_headroom.py`, `type_threshold.py` (analyses of Module B), `model_cost.py`, `e2e_speed.py` (cost and end-to-end latency), `verify_env.py` (checks that GPU kernels really run). |

Each script's docstring states what it reads and writes. Run everything from the repository root; the code finds
`data/`, `runs/`, `weights/`, `third_party/` and `configs/` relative to it.

## Setup

Python 3.10 with CUDA 12.4. The versions are those of the experiments.

```bash
pip install -r requirements.txt
python src/evaluation/verify_env.py
```

Third-party code, not included (both Apache-2.0), at the commits used:

```bash
git clone https://github.com/lyuwenyu/RT-DETR third_party/RT-DETR-main
git -C third_party/RT-DETR-main checkout 29320b6fd828f8e0987a71426cf2d961b09dfed7
git clone https://github.com/Intellindust-AI-Lab/DEIM third_party/DEIM-main
git -C third_party/DEIM-main checkout 09d35d53d39ee3145a1e61e3a989b28b9468d1dd
python src/models/patch_deim.py
```

RT-DETR's `rtdetrv2_pytorch` serves RT-DETR-R18 and RT-DETRv2-S; DEIM serves DEIM-R18 and the base detector.
`patch_deim.py` adapts DEIM to torchvision 0.21 and makes its `calflops` import optional.

Pretrained and trained checkpoints: see [weights/README.md](weights/README.md).

## Data

Images and annotations are not redistributed; get each dataset from its authors and build it with `src/data/`:

| Dataset | Get it | Build (in this order) |
|---|---|---|
| SeaShips, public 7,000-image subset | Kaggle `nam97595/seaships7k` (a Roboflow export) as `data/_raw/seaships7k.zip` | `extract_seaships.py`, `build_seaships_pool.py`, `seaships_osd.py`, `build_seaships_splits.py`, `build_seaships_coco.py` |
| Singapore Maritime Dataset (zero-shot test) | `python src/data/fetch_smd.py` (the annotated VIS videos only, from Kaggle `mmichelli/singapore-maritime-dataset`) | `build_smd.py` |
| SMD-Plus | Google Drive file `1yokFvx_cJu-Fl5ti1wgF1WPHao_2kcKE` ([kjunhwa/SMD-Plus](https://github.com/kjunhwa/SMD-Plus)); unpack its `VIS_Onshore.zip` and `VIS_Onboard.zip` into `data/smdplus_raw/` | `build_smdplus.py` |
| ABOships | `python src/data/fetch_aboships.py` (Zenodo record 4736931, the labelled frames only) | `build_aboships.py`, `build_aboships_splits.py` |
| McShips (lite) | Google Drive file `1udewXbHCS9WKM-MPpWqouUUGs6Vx5iWf` ([ZhengYitong2333/Mcships](https://github.com/ZhengYitong2333/Mcships)) as `data/_raw/Mcship_database_lite.zip` | `build_mcships.py` |
| WSODD | the Baidu share of [sunjiaen/WSODD](https://github.com/sunjiaen/WSODD) (`https://pan.baidu.com/s/1-xT6fwH3alW78uCsm9VjRA`, code 1234); unpack `WSODD USV_dataset.zip` into `data/wsodd_raw/` | `build_wsodd.py` |

Then cut the crops of Module B and check the splits:

```bash
python src/data/build_crops.py
python src/data/build_crops_multi.py
python src/data/split_manifest.py check
```

### Recording-level splits

`splits/record_level_split.json` lists, for each protocol, every recording with its subset and image ids (McShips and
the SeaShips random split by image), plus the class names in label order. `split_manifest.py check` compares a rebuilt
`data/` with it. Image counts, with recordings in brackets:

| Protocol | Dataset | Recording | Train | Val | Test |
|---|---|---|---|---|---|
| `session` | SeaShips | capture session | 4,846 (110) | 711 (12) | 1,422 (34) |
| `random` | SeaShips | none (frames at random) | 4,886 | 698 | 1,395; cross_test 284 |
| `smdplus` | SMD-Plus | video | 2,779 (32) | 385 (5) | 1,385 (14) |
| `aboships` | ABOships | video (whole days per subset) | 6,389 (9 days, 126 videos) | 1,299 (2 days, 36 videos) | 1,353 (2 days, 40 videos) |
| `mcships` | McShips (lite) | none (web photos, official split) | 4,048 | 1,637 | 2,311; test_dedup 2,195 |
| `wsodd` | WSODD | capture session | 5,227 (23) | 747 (11) | 1,493 (9) |
| `smd` | SMD (zero-shot test) | video | — | — | 4,002 (40) |

The SeaShips random split is kept only to measure its leak: `cross_test` (random test ∩ session test) holds the test
images neither protocol trained on.

## Train

```bash
bash scripts/train.sh session 0
bash scripts/train.sh wsodd 2 deim-r18
```

The first line trains our method on SeaShips with seed 0: the base detector, then Module A and the classifier of
Module B. The second trains a baseline. Protocols are `session` (SeaShips), `smdplus`, `aboships`, `mcships`, `wsodd`,
and `random` (SeaShips, leak study); seeds are 0, 1 and 2. Baselines keep their own published recipes; their names are
listed in `configs/base.yaml`.

## Evaluate

```bash
bash scripts/evaluate.sh session 0
bash scripts/evaluate.sh compare session
```

The first line tests seed 0 step by step: the detector, + Module A, + Module B (also without Module A), and with
sliced inference where `configs/base.yaml` switches it on. SeaShips runs are also tested zero-shot on SMD and ABOships.
Each run gets `runs/seaships/<run>/eval/summary.json`, and `runs/seaships/results.csv` and `results_mean.csv` are
rebuilt. The second line compares our method with every baseline over the three seeds (two-level bootstrap) and writes
`runs/seaships/compare_*_cluster_seeds.txt`.

All runs go to `runs/seaships/`, whatever the dataset. Run names:

| Run | Contents |
|---|---|
| `deim-r18-nomix_<protocol>_s<seed>` | base detector |
| `deim-r18-nomix-wise090_<protocol>_s<seed>` | + Module A (α = 0.9) |
| `deim-r18-nomix-wise090-fw05_<protocol>_s<seed>` | + Module B (w = 0.5): our method |
| `deim-r18-nomix-wise090-sl2-fw05_<protocol>_s<seed>` | our method with sliced inference (SMD-Plus) |
| `deim-r18-nomix-fw05_<protocol>_s<seed>` | Module B without Module A (ablation) |
| `runs/types/convnext-t-ft_<protocol>_s<seed>/` | the classifier of Module B |

Results of our method, mean ± s.d. over seeds 0–2, ×100 (`own_coco_*` in `results_mean.csv`):

| Protocol | Run | mAP50 | mAP50-95 |
|---|---|---|---|
| `session` | `deim-r18-nomix-wise090-fw05` | 96.2 ± 0.3 | 76.1 ± 0.7 |
| `smdplus` | `deim-r18-nomix-wise090-sl2-fw05` | 69.9 ± 2.8 | 41.3 ± 1.4 |
| `aboships` | `deim-r18-nomix-wise090-fw05` | 59.2 ± 0.2 | 32.0 ± 0.2 |
| `mcships` | `deim-r18-nomix-wise090-fw05` | 95.8 ± 0.2 | 72.1 ± 0.2 |
| `wsodd` | `deim-r18-nomix-wise090-fw05` | 51.6 ± 1.1 | 26.2 ± 1.1 |

Trained on SeaShips (`session`) and tested zero-shot, class-agnostic AP50 at 640: SMD 50.7 ± 0.5, ABOships 13.2 ± 2.1.

DETR training with the official code is seeded but not bit-reproducible, so a rerun lands within about the seed
spread shown here rather than on the same numbers. Most runs were made on one RTX 4070 Ti SUPER (16 GB, Windows 11),
some WSODD runs on RTX 4090 D cards (Linux).

### Other experiments

| Experiment | Command |
|---|---|
| Leak of the random split | train a model on `random` and `session`, then `python src/evaluation/compare_runs.py --a <model>_random_s{0,1,2} --b <model>_session_s{0,1,2} --split cross_test --preds val` |
| Module B on another detector | `python src/models/type_rescore.py --run <run> --arm convnext-t-ft_<protocol>_s<seed> --top 100 --fuse --w 0.5 --save-as <model>-fw05` (add `--no-val` for ultralytics runs) |
| Module A on another DETR | `python src/models/wise_ft.py --run <run> --alphas 0.9 --val` |
| Sliced inference | `python src/models/slice_eval.py --run <run>` |
| Recording-level bootstrap of any two models | `python src/evaluation/boot_cache.py cache --split <split> --cluster --runs <runs>`, then `boot_cache.py compare --split <split> --cluster --seed-boot --a <runs> --b <runs>` |
| Parameters, GFLOPs, latency | `python src/evaluation/model_cost.py --models deim-r18-nomix` |
| End-to-end latency with Module B | `python src/evaluation/e2e_speed.py --run deim-r18-nomix-wise090_<protocol>_s0 --arm convnext-t-ft_<protocol>_s0` |

## Notes

- Training our base detector takes 3–6 h per run on one RTX 4070 Ti SUPER (120 epochs); its classifier takes up to
  15 min, and Module A a few minutes.
- On Windows every data-loader worker commits 0.7–2.4 GB of host memory, so train one model at a time with 4 workers.
- Do not start two ultralytics trainings on the same dataset at once: they share `data/<dataset>/labels.cache`.
- A training run whose directory already exists fails; move an interrupted run directory aside before retraining.
- The scripts set `YOLO_AUTOINSTALL=False`, so ultralytics does not install packages at run time.
- Latency was measured on Windows 11 with `e2e_speed.py`, which raises the process priority and switches off power
  throttling; batch-1 GPU inference in a throttled background process runs about twice as slow.
