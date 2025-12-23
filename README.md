# Vision Deep Learning Pipeline: Training, Transfer Learning, and Attribution (CIFAR-10)

## Overview

This repository presents a complete **computer vision deep learning workflow** built on the CIFAR-10 dataset.
The project emphasizes **training engineering**, **data augmentation**, **transfer learning**, **model diagnostics**, and **interpretability** using Grad-CAM.

The goal is not just accuracy, but **reproducible experimentation**, **robust evaluation**, and **clear model understanding**—from scratch training to calibrated inference.

---

## Objectives

* Build and train CNNs **from scratch** with stable optimization
* Apply **modern augmentations** and study their effects
* Fine-tune **pretrained vision backbones**
* Diagnose failures via **misclassification analysis**
* Interpret predictions using **Grad-CAM**
* Evaluate **model calibration** and reliability
* Produce clean, reusable artifacts for inference and reporting

---

## Project Structure

```
vision-dl/
│
├── data/
│   └── cifar-10-batches-py/        # CIFAR-10 batches (non-archived)
│
├── notebooks/
│   ├── data/                       # Local dataset access for notebooks
│   ├── artifacts/
│   │   ├── gradcam/                # Grad-CAM visualizations
│   │   └── misclassified/          # Failure case examples
│   ├── wandb/                      # Experiment logs (local sync)
│   └── *.ipynb                     # Training & analysis notebooks
│
├── artifacts/
│   ├── checkpoints/                # Best model weights
│   ├── metrics/                    # Evaluation outputs
│   └── figures/                    # Plots & diagrams
│
├── samples/                        # Example images for inference
├── temp/                           # Temporary experiments
│
├── inference.py                    # Reproducible inference script
├── model_card.md                   # Model card & documentation
├── .gitignore
└── README.md
```

---

## Dataset

* **CIFAR-10** (60,000 images, 10 classes, 32×32)
* Source: University of Toronto
  [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

> Note: Large compressed archives are intentionally excluded from version control.
> Data is either downloaded programmatically or stored in unpacked batch form.

---

## Experiments & Methodology

### 1. Data & Augmentation

* Train / validation / test splits
* Baseline augmentations:

  * Random crop
  * Horizontal flip
  * Normalization
* Advanced augmentations (ablated):

  * RandAugment
  * MixUp
  * CutMix

### 2. From-Scratch CNN

* Custom lightweight CNN architecture
* Optimizers and schedules:

  * Cosine Annealing
  * OneCycleLR
* Mixed precision training (AMP)
* Stability and convergence monitoring

### 3. Transfer Learning

* Pretrained models:

  * ResNet
  * MobileNet
* Strategies compared:

  * Frozen backbone
  * Partial unfreezing
* Convergence speed and wall-clock analysis

### 4. Diagnostics & Robustness

* Misclassification review
* Grad-CAM visual explanations (20+ examples)
* Adversarial noise sanity checks
* Calibration analysis:

  * Expected Calibration Error (ECE)
  * Reliability diagrams

### 5. Reproducible Artifacts

* Saved best checkpoints
* Deterministic inference script
* Model card with:

  * Intended use
  * Limitations
  * Failure modes
  * Robustness notes

---

## Deliverables

* Training & transfer learning notebooks
* Loss/accuracy curves
* Confusion matrices
* Grad-CAM visualization gallery
* Calibrated inference pipeline
* Model card documentation

---

## How to Run

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Training & Analysis

Open and run notebooks in:

```
notebooks/
```

### 3. Inference

```bash
python inference.py --image samples/example.png
```

---

## Reproducibility Notes

* All experiments are logged locally (wandb compatible)
* Random seeds are fixed where applicable
* Checkpoints and metrics are versioned under `artifacts/`

---

## Future Improvements

* Temperature scaling for post-hoc calibration
* Larger robustness tests (blur, corruption benchmarks)
* Lightweight deployment (ONNX / TorchScript)
* Interactive visualization dashboard

---

## License

This project is intended for **educational and research use**.

