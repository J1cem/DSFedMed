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
