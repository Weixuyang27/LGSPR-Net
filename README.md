# LGSPR-Net

Official PyTorch implementation of **LGSPR-Net: Local--Global Learning With
Spectral Perturbation Supervision and Prototype-Conditioned Relation Alignment
for Cross-Domain Few-Shot Hyperspectral Image Classification**.

LGSPR-Net consists of a local--global spectral--spatial encoder (LGSSE),
spectral perturbation supervision (SPS), and prototype-conditioned relation
alignment (PCRA). SPS and PCRA are used only during training.

## Requirements

The code was tested with Python 3.10.20, PyTorch 2.5.1, CUDA 12.1, and
`mamba-ssm` 2.2.4.

```bash
pip install -r requirements.txt
```

## Datasets

Download Pavia University (PU), Pavia Centre (PC), Indian Pines (IP), Salinas
(SA), and Chikusei (CH), and arrange them as follows:

```text
Datasets/
|-- paviaU/
|   |-- PaviaU.mat
|   `-- PaviaU_gt.mat
|-- paviaC/
|   |-- Pavia.mat
|   `-- Pavia_gt.mat
|-- indianPines/
|   |-- Indian_pines_corrected.mat
|   `-- Indian_pines_gt.mat
|-- salinas/
|   |-- Salinas_corrected.mat
|   `-- Salinas_gt.mat
`-- Chikusei_MATLAB/
    |-- HyperspecVNIR_Chikusei_20140729.mat
    `-- HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat
```

The datasets are not redistributed in this repository. Alternative Chikusei
paths can be supplied with `--chikusei_mat` and `--chikusei_gt`.

## Training

The four transfer settings are:

| Case | Transfer |
|---|---|
| `case1` | PU to PC |
| `case2` | IP to SA |
| `case3` | CH to PC |
| `case4` | CH to SA |

Run the complete model on PU-to-PC with:

```bash
python train.py --case case1 \
  --paired_deterministic --gpu 0 --episode 5000 \
  --seed 1337 --split_seed 1337 \
  --target_split splits/case1_5shot_split1337.npz \
  --target_query 4 --labeled_per_class 5 \
  --proto_temperature 0.5 --patch_size 9 \
  --perturb_gain 0.05 --perturb_spectral_tilt 0.05 --perturb_offset 0.02 \
  --lambda_s 0.05 --lambda_p 0.05 --sps_warmup 1000 \
  --relation_queries_per_class 2 --lambda_pair 0.1 --lambda_cd 0.01 \
  --pcra_start 1000 --pcra_warmup 1000 \
  --save_dir checkpoints/pu_to_pc \
  --metrics_json results/pu_to_pc.json
```

For another transfer, change `--case` and use the corresponding split file.
The released split seeds are `7`, `42`, `1337`, `2024`, and `3407`.

The eight component combinations are selected with `--no_lgsse`, `--no_sps`,
and `--no_pcra`; all three components are enabled by default.

## Evaluation

Generate a target-domain prediction map from a trained checkpoint:

```bash
python eval.py --case case1 \
  --checkpoint checkpoints/pu_to_pc/lgspr_net_case1_best.pth \
  --target_split splits/case1_5shot_split1337.npz
```

## Comparison Methods

`main.py` provides a common entry for SSRN, RN-FSC, DFSL, Gia-CFSL, DCFSL,
and CMFSL. For example:

```bash
python main.py --model ssrn --transfer pu_to_pc \
  --target_split splits/case1_5shot_split1337.npz \
  --seed 1337 --device cuda:0
```

Supported model names are `ssrn`, `rnfsc`, `dfsl`, `giacfsl`, `dcfsl`, and
`cmfsl`. Results are saved under
`results/comparisons/<transfer>/<model>/seed<seed>/`.

## Figures

```bash
# Dataset figures
python -m utils.draw_dataset dataset --dataset all

# Classification-map comparisons
python -m utils.draw_dataset comparison --transfer all --seed 1337

# Example patch-size sensitivity figure
python -m utils.plot_sensitivity --type patch --transfer pu_to_pc \
  --input_dir results/patch_size_pu_to_pc
```

## Repository Structure

```text
models/lgspr_net.py       LGSPR-Net, LGSSE, SPS, and PCRA
train.py                  training entry for the four transfers
eval.py                   checkpoint evaluation and map prediction
main.py                   comparison-method entry
baselines/                comparison training protocols
models/                   proposed and comparison models
utils/                    data, evaluation, plotting, and split utilities
splits/                   fixed five-shot target splits
```

## Citation

If this code is useful for your research, please cite the corresponding paper.
Citation information will be added after publication.
