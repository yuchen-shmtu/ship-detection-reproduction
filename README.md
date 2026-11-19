# Ship detection and ship-type recognition under recording-level evaluation

Code, configurations and data splits to reproduce the experiments of the
paper *A detector-agnostic ship-type stage and weight interpolation for
real-time maritime detection under recording-level evaluation*.

The method is a real-time DETR with two plug-in modules:

-   **Base detector**: DEIM-R18 (DEIM training on the RT-DETRv2-S model,
    \~20 M parameters) fine-tuned from COCO without Mosaic and MixUp
    (run name `deim-r18-nomix`).
-   **Module A, WiSE-FT**: the fine-tuned weights interpolated towards
    the COCO weights they started from, θ = α·θ_ft + (1 − α)·θ_COCO with
    α = 0.9; no further training.
-   **Module B, ship-type stage**: the detector's 100 best boxes per
    image are cropped (12.5% context per side) and classified by a
    temperature-scaled ConvNeXt-T; its probabilities p_c are fused with
    the detector's class scores s_c as f_c = max(s_c, ε·s_max)\^(1−w) ·
    (s_max·p_c)\^w with w = 0.5 and ε = 10⁻³. Boxes are not moved.
-   **Sliced inference** (the full frame plus 2 × 2 tiles of 0.6 of the
    frame) only where a validation rule switches it on, which is
    SMD-Plus only.

Every dataset is split by recording: no capture session, video or day
spans two of train, val and test. Every model is trained with three
seeds, and comparisons use a two-level bootstrap that resamples test
recordings and training seeds.

## Layout

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

  -----------------------------------------------------------------------
  ## Folder Contents

### `src/data/`

- `ship_data.py`: Dataset registry and ground-truth helpers.
- **SeaShips:**
  - `extract_seaships.py`
  - `build_seaships_pool.py`
  - `build_seaships_coco.py`
  - `seaships_osd.py`: Camera and time extraction from the on-screen display, using `osd_*_templates.npz`.
  - `build_seaships_splits.py`: Capture sessions and session-level splitting.
  - `audit_seaships_split.py`: Leakage audit of a split.
- **SMD and zero-shot:**
  - `fetch_smd.py`
  - `build_smd.py`
  - `build_smdplus.py`
- **ABOships:**
  - `fetch_aboships.py`
  - `build_aboships.py`
  - `build_aboships_splits.py`
- `build_mcships.py`
- `build_wsodd.py`
- **Crops for Module B:**
  - `build_crops.py`
  - `build_crops_multi.py`
  - `mine_det_crops.py`
- `split_manifest.py`: Writes and checks `splits/record_level_split.json`.

### `src/models/`

- `configs/*_seaships.yml`: Detector recipes based on official DEIM / RT-DETR configurations, including our changes.
- `train_detector.py`: Trains any detector.
- `rtdetr_official.py`: RT-DETR-R18, RT-DETRv2-S, DEIM-R18, and the base detector using their official implementations.
- `boat_detr_modules.py`: Modules registered into DEIM, including small-object copy-paste and two variants that were dropped.
- `tv_detector.py`: Faster R-CNN and SSD300.
- `patch_deim.py`
- `fetch_weights.py`
- `run_suite.py`: Queue of training and testing jobs.
- `wise_ft.py`: Module A.
- `type_classifier.py`
- `type_rescore.py`: Module B classifier and fusion.
- `type_variants.py`: Variants of Module B.
- `slice_eval.py`: Sliced inference.

### `src/evaluation/`

- `det_eval.py`: COCO AP in NumPy with ignore regions; matches pycocotools.
- `eval_detector.py`: Tests a run in-domain and zero-shot.
- `compare_runs.py`: Paired bootstrap over images or recordings.
- `boot_cache.py`: Two-level bootstrap over recordings and seeds.
- `tabulate_runs.py`: Generates `results.csv` and `results_mean.csv`.
- `type_confusion.py`
- `type_headroom.py`
- `type_threshold.py`: Analyses of Module B.
- `model_cost.py`
- `e2e_speed.py`: Model cost and end-to-end latency.
- `verify_env.py`: Checks that GPU kernels really run.
  
  -----------------------------------------------------------------------

Each script's docstring states what it reads and writes. Run everything
from the repository root; the code finds `data/`, `runs/`, `weights/`,
`third_party/` and `configs/` relative to it.

## Setup

Python 3.10 with CUDA 12.4. The versions are those of the experiments.

``` bash
pip install -r requirements.txt
python src/evaluation/verify_env.py
```

Third-party code, not included (both Apache-2.0), at the commits used:

``` bash
git clone https://github.com/lyuwenyu/RT-DETR third_party/RT-DETR-main
git -C third_party/RT-DETR-main checkout 29320b6fd828f8e0987a71426cf2d961b09dfed7
git clone https://github.com/Intellindust-AI-Lab/DEIM third_party/DEIM-main
git -C third_party/DEIM-main checkout 09d35d53d39ee3145a1e61e3a989b28b9468d1dd
python src/models/patch_deim.py
```

RT-DETR's `rtdetrv2_pytorch` serves RT-DETR-R18 and RT-DETRv2-S; DEIM
serves DEIM-R18 and the base detector. `patch_deim.py` adapts DEIM to
torchvision 0.21 and makes its `calflops` import optional.

Pretrained and trained checkpoints: see
[weights/README.md](weights/README.md).

## Pretrained Weights (GitHub Release)

The pretrained and trained model artifacts are published separately from
the source code to keep the Git repository manageable.

**Latest release: [v1.0.0 --- Initial Release - Ship Detection Model
Weights](https://github.com/yuchen-shmtu/ship-detection-reproduction/releases/tag/v1.0.0)**

Download all three parts of the split 7-Zip archive from the **Assets**
section of the release:

-   `ship-detection-weights_20261118_2.7z.001`
-   `ship-detection-weights_20261118_2.7z.002`
-   `ship-detection-weights_20261118_2.7z.003`

### Download and extract on Windows

1.  Download **all three** archive parts. Keep their filenames
    unchanged.
2.  Put the three files in the **same folder**. Do not rename them or
    try to extract each part separately.
3.  Install [7-Zip](https://www.7-zip.org/) if it is not already
    installed.
4.  In File Explorer, right-click
    `ship-detection-weights_20261118_2.7z.001` and choose **7-Zip →
    Extract Here** (or **Extract to...**).
5.  If you choose **Extract to...**, extract into the repository root so
    that the archive's directory structure is preserved. Keep all three
    parts together until extraction has finished.

The archive is split into volumes: `.001` is the volume to open for
extraction, while `.002` and `.003` must remain beside it. A `.001` file
is not a standalone archive.

After extraction, check the resulting directory tree and consult
[`weights/README.md`](weights/README.md) for checkpoint provenance and
the expected checkpoint locations. The release archive shown in the
published package contains a `runs/` directory; preserve that directory
structure when extracting.

### Verify downloaded files (optional but recommended)

GitHub displays a SHA-256 digest for each release asset. On Windows
PowerShell, calculate the digest of each downloaded part with:

``` powershell
Get-FileHash .\ship-detection-weights_20261118_2.7z.001 -Algorithm SHA256
Get-FileHash .\ship-detection-weights_20261118_2.7z.002 -Algorithm SHA256
Get-FileHash .\ship-detection-weights_20261118_2.7z.003 -Algorithm SHA256
```

Compare each full `Hash` value with the corresponding SHA-256 digest
shown by GitHub for that asset. Do not compare one part's hash with
another part's digest.

### Quick start

From the repository root:

``` bash
pip install -r requirements.txt
python src/evaluation/verify_env.py
```

Clone the two pinned third-party repositories and apply the project
compatibility patch as described in **Setup** above. Prepare the
required datasets as described in **Data**. Then follow **Evaluate** to
run the evaluation workflow for an available run, or **Train** first if
you are reproducing a run from scratch.

The release provides model artifacts; it does not include the original
image datasets. Dataset access and preparation are still required for
evaluation.

## Data

Images and annotations are not redistributed; get each dataset from its
authors and build it with `src/data/`:

  -------------------------------------------------------------------------------------------------------------------------------
  Dataset                 Get it                                                                    Build (in this order)
  ----------------------- ------------------------------------------------------------------------- -----------------------------
  SeaShips, public        Kaggle `nam97595/seaships7k` (a Roboflow export) as                       `extract_seaships.py`,
  7,000-image subset      `data/_raw/seaships7k.zip`                                                `build_seaships_pool.py`,
                                                                                                    `seaships_osd.py`,
                                                                                                    `build_seaships_splits.py`,
                                                                                                    `build_seaships_coco.py`

  Singapore Maritime      `python src/data/fetch_smd.py` (the annotated VIS videos only, from       `build_smd.py`
  Dataset (zero-shot      Kaggle `mmichelli/singapore-maritime-dataset`)                            
  test)                                                                                             

  SMD-Plus                Google Drive file `1yokFvx_cJu-Fl5ti1wgF1WPHao_2kcKE`                     `build_smdplus.py`
                          ([kjunhwa/SMD-Plus](https://github.com/kjunhwa/SMD-Plus)); unpack its     
                          `VIS_Onshore.zip` and `VIS_Onboard.zip` into `data/smdplus_raw/`          

  ABOships                `python src/data/fetch_aboships.py` (Zenodo record 4736931, the labelled  `build_aboships.py`,
                          frames only)                                                              `build_aboships_splits.py`

  McShips (lite)          Google Drive file `1udewXbHCS9WKM-MPpWqouUUGs6Vx5iWf`                     `build_mcships.py`
                          ([ZhengYitong2333/Mcships](https://github.com/ZhengYitong2333/Mcships))   
                          as `data/_raw/Mcship_database_lite.zip`                                   

  WSODD                   the Baidu share of [sunjiaen/WSODD](https://github.com/sunjiaen/WSODD)    `build_wsodd.py`
                          (`https://pan.baidu.com/s/1-xT6fwH3alW78uCsm9VjRA`, code 1234); unpack    
                          `WSODD USV_dataset.zip` into `data/wsodd_raw/`                            
  -------------------------------------------------------------------------------------------------------------------------------

Then cut the crops of Module B and check the splits:

``` bash
python src/data/build_crops.py
python src/data/build_crops_multi.py
python src/data/split_manifest.py check
```

### Recording-level splits

`splits/record_level_split.json` lists, for each protocol, every
recording with its subset and image ids (McShips and the SeaShips random
split by image), plus the class names in label order.
`split_manifest.py check` compares a rebuilt `data/` with it. Image
counts, with recordings in brackets:

  --------------------------------------------------------------------------
  Protocol     Dataset      Recording   Train       Val         Test
  ------------ ------------ ----------- ----------- ----------- ------------
  `session`    SeaShips     capture     4,846 (110) 711 (12)    1,422 (34)
                            session                             

  `random`     SeaShips     none        4,886       698         1,395;
                            (frames at                          cross_test
                            random)                             284

  `smdplus`    SMD-Plus     video       2,779 (32)  385 (5)     1,385 (14)

  `aboships`   ABOships     video       6,389 (9    1,299 (2    1,353 (2
                            (whole days days, 126   days, 36    days, 40
                            per subset) videos)     videos)     videos)

  `mcships`    McShips      none (web   4,048       1,637       2,311;
               (lite)       photos,                             test_dedup
                            official                            2,195
                            split)                              

  `wsodd`      WSODD        capture     5,227 (23)  747 (11)    1,493 (9)
                            session                             

  `smd`        SMD          video       ---         ---         4,002 (40)
               (zero-shot                                       
               test)                                            
  --------------------------------------------------------------------------

The SeaShips random split is kept only to measure its leak: `cross_test`
(random test ∩ session test) holds the test images neither protocol
trained on.

## Train

``` bash
bash scripts/train.sh session 0
bash scripts/train.sh wsodd 2 deim-r18
```

The first line trains our method on SeaShips with seed 0: the base
detector, then Module A and the classifier of Module B. The second
trains a baseline. Protocols are `session` (SeaShips), `smdplus`,
`aboships`, `mcships`, `wsodd`, and `random` (SeaShips, leak study);
seeds are 0, 1 and 2. Baselines keep their own published recipes; their
names are listed in `configs/base.yaml`.

## Evaluate

``` bash
bash scripts/evaluate.sh session 0
bash scripts/evaluate.sh compare session
```

The first line tests seed 0 step by step: the detector, + Module A, +
Module B (also without Module A), and with sliced inference where
`configs/base.yaml` switches it on. SeaShips runs are also tested
zero-shot on SMD and ABOships. Each run gets
`runs/seaships/<run>/eval/summary.json`, and `runs/seaships/results.csv`
and `results_mean.csv` are rebuilt. The second line compares our method
with every baseline over the three seeds (two-level bootstrap) and
writes `runs/seaships/compare_*_cluster_seeds.txt`.

All runs go to `runs/seaships/`, whatever the dataset. Run names:

  ------------------------------------------------------------------------------------------
  Run                                                    Contents
  ------------------------------------------------------ -----------------------------------
  `deim-r18-nomix_<protocol>_s<seed>`                    base detector

  `deim-r18-nomix-wise090_<protocol>_s<seed>`            \+ Module A (α = 0.9)

  `deim-r18-nomix-wise090-fw05_<protocol>_s<seed>`       \+ Module B (w = 0.5): our method

  `deim-r18-nomix-wise090-sl2-fw05_<protocol>_s<seed>`   our method with sliced inference
                                                         (SMD-Plus)

  `deim-r18-nomix-fw05_<protocol>_s<seed>`               Module B without Module A
                                                         (ablation)

  `runs/types/convnext-t-ft_<protocol>_s<seed>/`         the classifier of Module B
  ------------------------------------------------------------------------------------------

Results of our method, mean ± s.d. over seeds 0--2, ×100 (`own_coco_*`
in `results_mean.csv`):

  -----------------------------------------------------------------------------------------
  Protocol          Run                                 mAP50             mAP50-95
  ----------------- ----------------------------------- ----------------- -----------------
  `session`         `deim-r18-nomix-wise090-fw05`       96.2 ± 0.3        76.1 ± 0.7

  `smdplus`         `deim-r18-nomix-wise090-sl2-fw05`   69.9 ± 2.8        41.3 ± 1.4

  `aboships`        `deim-r18-nomix-wise090-fw05`       59.2 ± 0.2        32.0 ± 0.2

  `mcships`         `deim-r18-nomix-wise090-fw05`       95.8 ± 0.2        72.1 ± 0.2

  `wsodd`           `deim-r18-nomix-wise090-fw05`       51.6 ± 1.1        26.2 ± 1.1
  -----------------------------------------------------------------------------------------

Trained on SeaShips (`session`) and tested zero-shot, class-agnostic
AP50 at 640: SMD 50.7 ± 0.5, ABOships 13.2 ± 2.1.

DETR training with the official code is seeded but not bit-reproducible,
so a rerun lands within about the seed spread shown here rather than on
the same numbers. Most runs were made on one RTX 4070 Ti SUPER (16 GB,
Windows 11), some WSODD runs on RTX 4090 D cards (Linux).

### Other experiments

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Experiment                          Command
  ----------------------------------- ----------------------------------------------------------------------------------------------------------------------------------------
  Leak of the random split            train a model on `random` and `session`, then
                                      `python src/evaluation/compare_runs.py --a <model>_random_s{0,1,2} --b <model>_session_s{0,1,2} --split cross_test --preds val`

  Module B on another detector        `python src/models/type_rescore.py --run <run> --arm convnext-t-ft_<protocol>_s<seed> --top 100 --fuse --w 0.5 --save-as <model>-fw05`
                                      (add `--no-val` for ultralytics runs)

  Module A on another DETR            `python src/models/wise_ft.py --run <run> --alphas 0.9 --val`

  Sliced inference                    `python src/models/slice_eval.py --run <run>`

  Recording-level bootstrap of any    `python src/evaluation/boot_cache.py cache --split <split> --cluster --runs <runs>`, then
  two models                          `boot_cache.py compare --split <split> --cluster --seed-boot --a <runs> --b <runs>`

  Parameters, GFLOPs, latency         `python src/evaluation/model_cost.py --models deim-r18-nomix`

  End-to-end latency with Module B    `python src/evaluation/e2e_speed.py --run deim-r18-nomix-wise090_<protocol>_s0 --arm convnext-t-ft_<protocol>_s0`
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Notes

-   Training our base detector takes 3--6 h per run on one RTX 4070 Ti
    SUPER (120 epochs); its classifier takes up to 15 min, and Module A
    a few minutes.
-   On Windows every data-loader worker commits 0.7--2.4 GB of host
    memory, so train one model at a time with 4 workers.
-   Do not start two ultralytics trainings on the same dataset at once:
    they share `data/<dataset>/labels.cache`.
-   A training run whose directory already exists fails; move an
    interrupted run directory aside before retraining.
-   The scripts set `YOLO_AUTOINSTALL=False`, so ultralytics does not
    install packages at run time.
-   Latency was measured on Windows 11 with `e2e_speed.py`, which raises
    the process priority and switches off power throttling; batch-1 GPU
    inference in a throttled background process runs about twice as
    slow.
