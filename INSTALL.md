# Installation

This file covers the tested GPU setup for this repository, plus dataset notes for both ImageNet-style experiments and DISFA AU fine-tuning.

## Tested Setup

The commands below were tested on Linux with:

- Python `3.8`
- PyTorch `1.8.0`
- torchvision `0.9.0`
- CUDA toolkit `11.1`
- `timm==0.3.2`
- custom `MinkowskiEngine` submodule in this repo

## Dependency Setup

Create a conda environment:

```bash
conda create -n convnextv2-gpu python=3.8 -y
conda activate convnextv2-gpu
```

Install PyTorch and core packages:

```bash
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y
conda install openblas-devel -c anaconda -y
pip install timm==0.3.2 tensorboardX six submitit ninja
```

Clone the repo and initialize submodules:

```bash
git clone https://github.com/facebookresearch/ConvNeXt-V2.git
cd ConvNeXt-V2
git submodule update --init --recursive
```

Install `MinkowskiEngine`.

Note:
- This repo depends on a customized CUDA kernel for depth-wise convolutions.
- Replace `/usr/local/cuda-11.1` if your CUDA install lives elsewhere.
- Set `TORCH_CUDA_ARCH_LIST` to match your GPU if you want faster extension builds.

```bash
cd MinkowskiEngine
TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4 python setup.py install \
  --force_cuda \
  --cuda_home=/usr/local/cuda-11.1 \
  --blas_include_dirs=${CONDA_PREFIX}/include \
  --blas=openblas
cd ..
```

## Apex

`apex` is optional.

The default fine-tuning path in this repo uses `AdamW` and works without `apex` when running with `--use_amp false`.

Only install `apex` if you specifically want Apex fused optimizers or an Apex-based AMP workflow.

Optional install:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
```

## Dataset Preparation

### ImageNet-1K

For ImageNet classification experiments, structure the data as:

```text
/path/to/imagenet-1k/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

### ImageNet-22K

For FCMAE pre-training on ImageNet-22K, structure the data as:

```text
/path/to/imagenet-22k/
  class1/
    img1.jpeg
  class2/
    img2.jpeg
  class3/
    img3.jpeg
  class4/
    img4.jpeg
```

### DISFA

The repository also includes a DISFA facial action unit fine-tuning path.

Expected DISFA layout under the repo root:

```text
dataset/DISFA/
  ActionUnit_Labels/
  Frames_Left/
  Faces_Left_RetinaFace_224/
  prepared/
    disfa_aligned_manifest.csv
    disfa_aligned_subject_splits.json
    disfa_aligned_au_summary.csv
```

If `Faces_Left_RetinaFace_224/` and `prepared/disfa_aligned_manifest.csv` already exist, you can skip the DISFA preprocessing steps below.

## DISFA Preprocessing

DISFA training uses RetinaFace-aligned `224x224` face crops plus a frame-level AU manifest.

Because the RetinaFace helper under `dataset/DISFA/` depends on TensorFlow and newer packages than the main training environment, it is recommended to run preprocessing in a separate environment.

Example preprocessing environment:

```bash
conda create -n convnextv2-disfa-prep python=3.10 -y
conda activate convnextv2-disfa-prep
pip install -r dataset/DISFA/requirements-retinaface-wsl-gpu.txt
```

Generate aligned face crops:

```bash
cd dataset/DISFA
bash run_disfa_retinaface_wsl.sh --dataset-root . --skip-existing
```

Build the manifest and deterministic subject splits:

```bash
python build_disfa_pytorch_manifest.py --dataset-root .
cd ../..
```

This writes:

- `dataset/DISFA/prepared/disfa_aligned_manifest.csv`
- `dataset/DISFA/prepared/disfa_aligned_subject_splits.json`
- `dataset/DISFA/prepared/disfa_aligned_au_summary.csv`

## DISFA Fine-Tuning

Activate the main training environment again:

```bash
conda activate convnextv2-gpu
```

DISFA fine-tuning uses:

- `--data_set DISFA`
- multi-label `BCEWithLogitsLoss`
- AU-F1 as the main model-selection metric
- optional micro-F1 logging per epoch via `--disfa_log_micro_f1 true`

Default DISFA settings:

- all 12 AUs: `1,2,4,5,6,9,12,15,17,20,25,26`
- binary labels: AU active if intensity `> 0`
- validation split from `prepared/disfa_aligned_subject_splits.json`

Example command:

```bash
python main_finetune.py \
  --model convnextv2_atto \
  --input_size 224 \
  --batch_size 16 \
  --epochs 20 \
  --warmup_epochs 2 \
  --blr 2e-4 \
  --drop_path 0.1 \
  --num_workers 4 \
  --data_set DISFA \
  --data_path /path/to/ConvNeXt-V2/dataset/DISFA \
  --finetune /path/to/fcmae_checkpoint.pth \
  --output_dir /path/to/save_results \
  --log_dir /path/to/save_results/logs \
  --auto_resume false \
  --use_amp false \
  --disfa_log_micro_f1 true
```

Useful DISFA options:

- evaluate on the test split instead of validation:

```bash
--disfa_eval_split test
```

- use strong labels instead of binary labels:

```bash
--disfa_target_mode strong
```

- train on a subset of AUs:

```bash
--disfa_selected_aus 1,2,4,12
```

Run evaluation only on a saved checkpoint:

```bash
python main_finetune.py \
  --model convnextv2_atto \
  --input_size 224 \
  --batch_size 16 \
  --eval true \
  --num_workers 4 \
  --data_set DISFA \
  --data_path /path/to/ConvNeXt-V2/dataset/DISFA \
  --disfa_eval_split test \
  --resume /path/to/checkpoint-best.pth \
  --output_dir /path/to/eval_outputs \
  --auto_resume false \
  --use_amp false
```

## Notes

- DISFA preprocessing and DISFA training do not need to share the same environment.
- For the default DISFA training path, `apex` is not required.
- The pre-training code path is shape-sensitive; use the standard `224` image size unless you have validated a different setup.
