# 📌 DSFedMed: Federated Image Segmentation between Foundation and Lightweight Models

## 🧠 Overview
This repository implements **DSFedMed**, a synthetic data-driven federated learning framework for medical image segmentation. The core idea is to leverage **synthetic data and bidirectional knowledge distillation** to improve global model performance under **data heterogeneity (Non-IID)** and **privacy constraints**.

## 🚀 Key Features

- 🔒 Privacy-preserving federated learning (no raw data sharing)
- 🧪 Synthetic data generation for cross-client knowledge transfer
- 🔁 Bidirectional knowledge distillation between server and clients
- 🎯 Sample learnability-based selection for efficient training
- 🧩 Support for multi-modal medical datasets (MRI, fundus, ultrasound, etc.)

## 📂 Dataset Preparation

### Data format
All datasets should be converted into:

```bash
dataset/
├── Client_1/
│   ├── data_npy/
│   ├── label_npy/
│   ├── val_data_npy/
│   └── val_label_npy/
├── Client_2/
│   ├── data_npy/
│   ├── label_npy/
│   ├── val_data_npy/
│   └── val_label_npy/
└── ...
```

## 📂 Perform Mutual KD

### Nuclei

To run experiments on the nuclei dataset, please refer to `run_nuclei.sh`.

## 🧬 Completing the DSFedMed four-stage pipeline

This codebase now exposes the missing data-generation stages and the optional private TinySAM FedAvg stage used by the paper-style DSFedMed flow.

### 1) Client-side personalized ControlNet training

Each client trains only a small ControlNet-style residual branch on its private image-mask pairs. The frozen base generator is never updated, and clients upload only the branch checkpoint:

```bash
python train_controlnet_clients.py \
  --data_path /path/to/dataset \
  --client_names MoNuSAC2020,TNBC,MoNuSAC2018 \
  --data Nuclei_od \
  --output_dir /path/to/uploaded_controlnets \
  --epochs 10 \
  --batch_size 4 \
  --controlnet_tuning_mode lora
```

Use `--controlnet_tuning_mode lora` when GPU memory is limited. This freezes the ControlNet branch's original convolution/norm parameters and trains only low-rank LoRA adapters. Use `--controlnet_tuning_mode full` to reproduce full ControlNet branch fine-tuning. The LoRA adapter size can be adjusted with `--controlnet_lora_rank` and `--controlnet_lora_alpha`.

Outputs:

```bash
/path/to/uploaded_controlnets/
├── manifest.json
├── controlnet_client_0_MoNuSAC2020.pth
├── controlnet_client_1_TNBC.pth
└── controlnet_client_2_MoNuSAC2018.pth
```

### 2) Server-side synthetic data production

The server loads uploaded ControlNet branches, samples structural masks, and writes synthetic `data_npy` / `label_npy` pairs in the same format expected by the existing dataloaders:

```bash
python generate_synthetic_controlnet.py \
  --controlnet_dir /path/to/uploaded_controlnets \
  --output_dir /path/to/synthetic_dataset \
  --client_names MoNuSAC2020,TNBC,MoNuSAC2018 \
  --mask_root /path/to/dataset \
  --samples_per_client 100 \
  --image_size 1024
```

Add `--aggregate` to average uploaded ControlNet branches before generation, which gives a global style prior while still avoiding raw-data transfer.

### 3) Collaborative SAM / TinySAM training

`train_fed_sam_kd_sam_sample_2.0.py` can now launch synthetic generation before mutual KD when supplied with uploaded ControlNet branches:

```bash
python train_fed_sam_kd_sam_sample_2.0.py \
  --data Nuclei_od \
  --controlnet_dir /path/to/uploaded_controlnets \
  --generate_syn_data \
  --samples_per_client 100
```

To include the paper's client-side TinySAM training on private real data, add:

```bash
  --enable_private_fedavg --private_fedavg_steps 10
```

This performs local TinySAM updates on each source client and FedAvg aggregation between synthetic mutual-distillation epochs.

### 4) Learnability-guided bidirectional distillation

The mutual KD script now selects synthetic samples using a learnability score that combines:

- SAM prediction confidence / reliability.
- The prediction gap between SAM and TinySAM.

The selected subset is used for both directions of KL distillation: SAM → TinySAM for generalization and TinySAM → SAM for domain adaptation. Tune the confidence term with `--learnability_conf_weight`.
