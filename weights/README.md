# Weights

No checkpoint is tracked in the repository. The code reads each one from a fixed place:

| Kind | Read from |
|---|---|
| COCO / ImageNet pretrained starting points | `weights/<file>`, `weights/torch_hub/checkpoints/<file>` |
| Trained detectors | `runs/seaships/<run>/weights/best.pt` |
| Trained ship-type classifiers (Module B) | `runs/types/convnext-t-ft_<protocol>_s<seed>/model.pt` with its `metrics.json` (temperature, classes, crop padding) |

## 1. Pretrained starting points

Public files from their authors. The sha256 column is that of the copies the experiments used.

| File | Folder | Used by | Source | sha256 |
|---|---|---|---|---|
| `deim_rtdetrv2_r18vd_120e_coco.pth` | `weights/` | `deim-r18-nomix` (ours), `deim-r18` | [DEIM model zoo](https://drive.google.com/file/d/153_JKff6EpFgiLKaqkJsoDcLal_0ux_F/view) (Google Drive) | `cec854d43c88f11402c19a477127e7601195e011bf1d93d90a81fb7ed16b9d93` |
| `rtdetrv2_r18vd_120e_coco_rerun_48.1.pth` | `weights/` | `rtdetrv2-s` | https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth | `2ace52184b620204004509b72752ac7bfe64aadaf7fc1d076b18df8ab5a5c77e` |
| `rtdetr_r18vd_dec3_6x_coco_from_paddle.pth` | `weights/` | `rtdetr-r18` | https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth | `3ba8b5c909c9a1c4f21e96d0a7251ab1a485093955ca327d0061fef8d33c66f0` |
| `rtdetr_r18vd_5x_coco_objects365_from_paddle.pth` | `weights/` | `rtdetr-r18-o365` | https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth | `8f2278c42012a41971f900448b3d4dcb398356e8175e2b3ee4600e37eab0a045` |
| `yolo26n.pt` | `weights/` | `yolo26n` (baseline; also ultralytics' AMP check) | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| `yolo11n.pt` | `weights/` | `yolo11n` (baseline) | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| `yolov8n.pt` | `weights/` | `yolov8n` (baseline) | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| `yolov10n.pt` | `weights/` | `yolov10n` (baseline) | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov10n.pt | `11287ed0735678e7ba1ac2a9b3098c049155b3fde123992e724c1264bcc16b6f` |
| `yolo12n.pt` | `weights/` | `yolo12n` (baseline) | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12n.pt | `419ff3dca37d69bacc93a50fa0c186a1c6f9fe62fae0f108b0872829689e9ca6` |
| `yolov5nu.pt` | `weights/` | `yolov5nu` (baseline) | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov5nu.pt | `9e9c1be448b0e1b8598975a9abcab9a0fd0e21182eb422cda7f044a0442d0937` |
| `rtdetr-l.pt` | `weights/` | `rtdetr-l` (baseline) | https://github.com/ultralytics/assets/releases/download/v8.4.0/rtdetr-l.pt | `6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f` |
| `convnext_tiny-983f1562.pth` | `weights/torch_hub/checkpoints/` | Module B classifier (`convnext-t-ft`) | https://download.pytorch.org/models/convnext_tiny-983f1562.pth | `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d` |
| `fasterrcnn_resnet50_fpn_coco-258fb6c6.pth` | `weights/torch_hub/checkpoints/` | `faster-rcnn-r50` | https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth | `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384` |
| `ssd300_vgg16_coco-b556d3b4.pth` | `weights/torch_hub/checkpoints/` | `ssd300-vgg16` | https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth | `b556d3b43ab6c3f63d81bfb8835fe8756ac22da664357da100dccf96b6a6b42d` |
| `dinov2_vits14_pretrain.pth` | `weights/torch_hub/checkpoints/` | DINOv2 ablation arms only | https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth | `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` |
| `dinov2_vitb14_pretrain.pth` | `weights/torch_hub/checkpoints/` | DINOv2 ablation arms only | https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth | `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73` |

- ultralytics files: `python src/models/fetch_weights.py yolo26n.pt yolo11n.pt yolov8n.pt yolov10n.pt yolo12n.pt yolov5nu.pt rtdetr-l.pt`
  (checks the sha256 the release lists).
- RT-DETR files: `python src/models/fetch_weights.py --release lyuwenyu/storage@v0.1 <file>` (v0.2 for
  RT-DETRv2-S), or download the URL directly.
- The DEIM file is on Google Drive (link from DEIM's README, model zoo, DEIM-RT-DETRv2 S row); save it as
  `weights/deim_rtdetrv2_r18vd_120e_coco.pth`.
- torchvision files are downloaded into `weights/torch_hub/checkpoints/` on first use.
- DINOv2 (only for the DINOv2 classifier variants of the ablation): clone
  [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) into `weights/torch_hub/facebookresearch_dinov2_main/`
  and put the two checkpoints into `weights/torch_hub/checkpoints/`.

## 2. Trained checkpoints of the paper

Our method on each dataset and seed needs two files: the detector after Module A and the classifier of Module B.
The base detectors (before Module A) are only needed to redo Module A or the ablation rows; like the
Module A files, they hold only the EMA weights the code reads. All of them (4.1 GB) are in a separate
archive, `ship-detection-weights_20261118.zip`, not in git: unzip it at the repository root, which puts every file
at the path listed below, and check them with `sha256sum -c runs/SHA256SUMS`.

Download: *to be added when the archive is published.*

**Detector after Module A (our method's detector)**

| File | MB | sha256 |
|---|---|---|
| `runs/seaships/deim-r18-nomix-wise090_session_s0/weights/best.pt` | 80.7 | `1aeebc70aa75388c78c70bbf5ef29be8bcd7b57032bb70fe6ef75a3f1cec9fd1` |
| `runs/seaships/deim-r18-nomix-wise090_session_s1/weights/best.pt` | 80.7 | `4e0bf315bd4507c9404b0f6c0a87f0e97703f5cb13c61d54c1c7f6cbf95d7807` |
| `runs/seaships/deim-r18-nomix-wise090_session_s2/weights/best.pt` | 80.7 | `2b181cec45330d141cebdc30ed79b0be557fa7724fdee290f9127a66b3ca4fad` |
| `runs/seaships/deim-r18-nomix-wise090_smdplus_s0/weights/best.pt` | 80.7 | `4f2af49af9ebfdc27f2a696efd5544929ff378b57503cc3e2ca0fdabfcb0c44a` |
| `runs/seaships/deim-r18-nomix-wise090_smdplus_s1/weights/best.pt` | 80.7 | `ccd70b3fd89ccf2f9889b44dab002c5843ed398b0100992b2e215378f612dd2a` |
| `runs/seaships/deim-r18-nomix-wise090_smdplus_s2/weights/best.pt` | 80.7 | `960052f61bdb5c135f1c44c2b726de1b8106be5a08244250765624c0d234fb19` |
| `runs/seaships/deim-r18-nomix-wise090_aboships_s0/weights/best.pt` | 80.8 | `910318f908ebe7d17065889f03eb034bf1c735dfb5d56c65c3708230ff662551` |
| `runs/seaships/deim-r18-nomix-wise090_aboships_s1/weights/best.pt` | 80.8 | `f053d6b04ac596f4547e788c416ebfb7fac987c17be5b56e4672e004fc9e3447` |
| `runs/seaships/deim-r18-nomix-wise090_aboships_s2/weights/best.pt` | 80.8 | `9f1ce009d1837cb977fde53202807fe114f9e7f4b8c646eba5d2fc1810a30fc7` |
| `runs/seaships/deim-r18-nomix-wise090_mcships_s0/weights/best.pt` | 80.7 | `c94609185b3eb9cc42aea535635a6e1d689ab75ab0677af8b8426c173a0dae0e` |
| `runs/seaships/deim-r18-nomix-wise090_mcships_s1/weights/best.pt` | 80.7 | `e015ffc07e7327862752fb96d67e969d2bef43e8e524eb4aa49630436a04aab7` |
| `runs/seaships/deim-r18-nomix-wise090_mcships_s2/weights/best.pt` | 80.7 | `421f8ad9d0e4b2be45df20e91ef4468b78cf8472343d4444ba85f1273094a477` |
| `runs/seaships/deim-r18-nomix-wise090_wsodd_s0/weights/best.pt` | 80.8 | `552cf9c80cfe1e47078e56e01d426509c032c1e5654e045e61a95f98db138a46` |
| `runs/seaships/deim-r18-nomix-wise090_wsodd_s1/weights/best.pt` | 80.8 | `a00bd9d196a897c613303479fc49ec3d3fde9bd0975dacf11762c454d997b6ac` |
| `runs/seaships/deim-r18-nomix-wise090_wsodd_s2/weights/best.pt` | 80.8 | `90ff314c699429fb597821b6ece9422514e61a33972de59df22b180b9c222171` |

**Classifier of Module B (with `metrics.json` from the same folder)**

| File | MB | sha256 |
|---|---|---|
| `runs/types/convnext-t-ft_session_s0/model.pt` | 111.4 | `3f2b8028efa22f0b7b4fdc30b5db35a9312f7046b7684711b3d5f47832eca47b` |
| `runs/types/convnext-t-ft_session_s1/model.pt` | 111.4 | `a2b00d70d5d95cfa4e675382ad1608da9528107d0d1138c89b9bb307d6ee7334` |
| `runs/types/convnext-t-ft_session_s2/model.pt` | 111.4 | `c916891f36f1da2427da4bb968f2e4d5a6e77d653b7822cf420799096d8e603e` |
| `runs/types/convnext-t-ft_smdplus_s0/model.pt` | 111.4 | `bca3658c66fd31aaece55f1ebbea6d378ea5027b70ba10093847e6f50e0c2c27` |
| `runs/types/convnext-t-ft_smdplus_s1/model.pt` | 111.4 | `ac3a3d722649c3a1623877756254de2dafe8ef73a173093bc6d5d430d048ff80` |
| `runs/types/convnext-t-ft_smdplus_s2/model.pt` | 111.4 | `710ba3dba6f5e82b784ee095d13a523f1680d2bd7db79463458b0b50918a2774` |
| `runs/types/convnext-t-ft_aboships_s0/model.pt` | 111.4 | `37b232a16d3972d55309462f8e60b23a0a6406f41deefe511cabc8ce32286433` |
| `runs/types/convnext-t-ft_aboships_s1/model.pt` | 111.4 | `2b82ae200cc63a87e4b5b3f5c303f97148c9567b1642f6fc55385bdb1ddc7529` |
| `runs/types/convnext-t-ft_aboships_s2/model.pt` | 111.4 | `641efa5077955fdd1ab9df23f4d5d1d703df0e40a2e25f57753be16d253569ac` |
| `runs/types/convnext-t-ft_mcships_s0/model.pt` | 111.3 | `324e2dbc8c1dda8e69fd35f41fc0a61fe7e33b2351bae6b87f16e75411c6e83c` |
| `runs/types/convnext-t-ft_mcships_s1/model.pt` | 111.3 | `e53e822e21a1c453352d5ab58550cc6d4b0246f282de7a247eb20ab025ca29c8` |
| `runs/types/convnext-t-ft_mcships_s2/model.pt` | 111.3 | `2a1db409e36f674a8d5afa3424ca941e8b01edac5062e8f83e1aa283dca72914` |
| `runs/types/convnext-t-ft_wsodd_s0/model.pt` | 111.4 | `3db9bba81041dcc7bad7488517a169726c8fe8717bc5f0a495fc6e1bb64f061e` |
| `runs/types/convnext-t-ft_wsodd_s1/model.pt` | 111.4 | `7e8b3d714442b5516cdbd55b4d02c89265d9d833d589165fc3c8f38666dceaaa` |
| `runs/types/convnext-t-ft_wsodd_s2/model.pt` | 111.4 | `c2788206a4c2ec7cfbaa7469f260280cda3e50fefb7be1853d28cb59c17f6fff` |

**Base detector before Module A (optional)**

| File | MB | sha256 |
|---|---|---|
| `runs/seaships/deim-r18-nomix_session_s0/weights/best.pt` | 80.8 | `34a9e6c72c570bdce81fea0b8a35d6c910be73e77bd5e13a26245a5907e69f12` |
| `runs/seaships/deim-r18-nomix_session_s1/weights/best.pt` | 80.8 | `17585c0d0ccb2200c11231ef145ff37baf61c09a85e13e4744015516989f4353` |
| `runs/seaships/deim-r18-nomix_session_s2/weights/best.pt` | 80.8 | `4b74246370d6fc1cf3e922a432f66c6f136354f82c4d03a479bd8c8cfe3a69c4` |
| `runs/seaships/deim-r18-nomix_smdplus_s0/weights/best.pt` | 80.8 | `ad2f7a69c6d7b85aa2444d1f09795de0f303e56ffa344df16c2c15a6df30993a` |
| `runs/seaships/deim-r18-nomix_smdplus_s1/weights/best.pt` | 80.8 | `f2f6fceb7601a293326f1c53c818d7d566622a76c8c7818ec9533577fecf9013` |
| `runs/seaships/deim-r18-nomix_smdplus_s2/weights/best.pt` | 80.8 | `dea0f261647fe1f6cefdb7e5d0a5a74c13cc386050d701fc4ec903aa7dab392f` |
| `runs/seaships/deim-r18-nomix_aboships_s0/weights/best.pt` | 80.8 | `a0a37c56a74753f08e4d8d15209071cb7bee87340875d44935b1dad526ef8572` |
| `runs/seaships/deim-r18-nomix_aboships_s1/weights/best.pt` | 80.8 | `18e89c6691a543313bc1efc383779a6faaac872e76cca51fa837363312e9454a` |
| `runs/seaships/deim-r18-nomix_aboships_s2/weights/best.pt` | 80.8 | `38ac1b3bb0a367e33361de689cde95ec3f7c63bbdd0366e5414c226a8c6d58e3` |
| `runs/seaships/deim-r18-nomix_mcships_s0/weights/best.pt` | 80.7 | `cd1ed6711f07f2d578f2fa7d55348b831e1c25c5882cd2421929c95340a9f0d5` |
| `runs/seaships/deim-r18-nomix_mcships_s1/weights/best.pt` | 80.7 | `0f011785c6e7f3c66109c9671b850ba5006e99536362b43df7b6ccff424370df` |
| `runs/seaships/deim-r18-nomix_mcships_s2/weights/best.pt` | 80.7 | `c90a4c6e93ffbe81bbc0fa498190e01ccae1fd9eac066c764a02f7ba29e86c61` |
| `runs/seaships/deim-r18-nomix_wsodd_s0/weights/best.pt` | 80.8 | `e9169b8e0e518a5168cae9b21c6a231ce820e8e11569179fa6a5d2f70d2626e5` |
| `runs/seaships/deim-r18-nomix_wsodd_s1/weights/best.pt` | 80.8 | `ed61717b84a934d7a76ef3932203e36d82d504625f17614ac0a442705c24ff49` |
| `runs/seaships/deim-r18-nomix_wsodd_s2/weights/best.pt` | 80.8 | `91a06f8032412502e82cd6b83d379a4407cc88ec726e01240d2dd17e8a8e4e31` |

With the checkpoints in place, `bash scripts/evaluate.sh <protocol> <seed>` tests the method without training;
runs whose weights are missing are skipped.
