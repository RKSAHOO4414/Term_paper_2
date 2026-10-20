# Zero-Shot Affordance Detection Under Occlusion

**Occlusion-Aware Training on Raw Point Clouds for Robotic Grasping**

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](https://github.com/RKSAHOO4414/Term_paper_2)
[![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-green)](https://drive.google.com/drive/folders/1s5W0Nfz9NEN8gP14tge8GuouUDXs2Ssq)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red)](https://pytorch.org/)

---

## 📖 Overview

Robotic manipulation of novel objects in cluttered environments requires predicting functional interaction regions—**affordances**—from partial, single-view 3D observations without category-specific training. This repository contains the official implementation of our paper:

> **Zero-Shot Affordance Detection Under Occlusion via Occlusion-Aware Training on Raw Point Clouds**

We propose a training framework that enables zero-shot affordance detection under occlusion using a standard **DGCNN** operating on **raw XYZ point cloud coordinates**. The framework combines three key techniques:

1. **Viewpoint-based occlusion augmentation** that simulates realistic partial observations during training without protecting labeled regions.
2. **Focal BCE + Dice loss** with per-class positive weighting to address extreme class imbalance (1–6% positive points).
3. **Balanced batch sampling** that guarantees representation of all affordance classes in every mini-batch.

---

## ✨ Key Contributions

- **Established a baseline** for geometry-only zero-shot affordance detection on the 3D AffordanceNet benchmark.
- **Demonstrated that occlusion-aware training is essential**—removing occlusion augmentation causes a **59% drop** in unseen-occluded mIoU (0.1091 → 0.0459).
- **Proved that explicit geometric features (normals, curvature) are redundant**—the DGCNN learns geometry implicitly from raw XYZ coordinates.
- **Showed that clean performance does not predict occlusion robustness**—a critical evaluation insight.
- **Achieved 44% grasp success** on unseen Mug objects in PyBullet simulation, a 4–8× improvement over random baseline.

---

## 📊 Results Summary

| Condition | Clean | Occluded |
|-----------|-------|----------|
| **Seen categories** (20 categories) | 0.2043 | 0.1890 |
| **Unseen categories** (Mug, Vase, Scissors) | 0.0940 | **0.1091** |

### Per-Class IoU on Unseen Categories (Occluded)

| Affordance | IoU | Assessment |
|------------|-----|------------|
| grasp | 0.017 | Fails (geometrically diverse) |
| contain | 0.174 | Moderate transfer |
| wrap_grasp | 0.072 | Poor transfer |
| pourable | 0.184 | Moderate transfer |

### Ablation Study (Unseen / Occluded mIoU)

| Configuration | mIoU | Δ from Full |
|---------------|------|-------------|
| **Full method** | **0.1091** | — |
| – Occlusion augmentation | 0.0459 | **−59%** |
| – Focal loss | 0.0651 | −40% |
| – Balanced sampling | 0.0991 | −9% |
| + Explicit geometric features (7D) | 0.1041 | −5% |
| + Extended features (13D) | 0.0968 | −11% |

---

## 📂 Repository Structure

```
Term_paper_2/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── data/
│   └── preprocess.py          # Download and prepare 3D AffordanceNet
├── models/
│   ├── dgcnn.py               # DGCNN architecture
│   └── loss.py                # Focal BCE + Dice loss
├── utils/
│   ├── occlusion.py           # Viewpoint-based occlusion augmentation
│   ├── sampler.py             # Balanced batch sampler
│   └── metrics.py             # mIoU computation
├── train.py                   # Training script
├── evaluate.py                # Zero-shot evaluation script
├── grasp_simulation/
│   ├── pybullet_grasp.py      # Grasp execution in PyBullet
│   └── run_trials.py          # 50-trial grasp evaluation
└── notebooks/
    └── demo.ipynb             # Quick-start demo
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- CUDA 11.3+ (for GPU training)
- PyTorch 1.10+

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/RKSAHOO4414/Term_paper_2.git
cd Term_paper_2

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install PyBullet for grasp simulation
pip install pybullet
```

---

## 📥 Dataset

We use the **3D AffordanceNet** benchmark. The dataset and the pre-processed splits used in this work are available at:

> **[Download Dataset (Google Drive)](https://drive.google.com/drive/folders/1s5W0Nfz9NEN8gP14tge8GuouUDXs2Ssq)**

### Dataset Structure

After downloading, place the files in the `data/` directory:

```
data/
├── full_shape_train_data.pkl
├── full_shape_val_data.pkl
├── partial_train_data.pkl
└── partial_val_data.pkl
```

### Zero-Shot Split

| Split | Categories | Samples |
|-------|------------|---------|
| **Training** | 20 categories | 3,550 |
| **Validation** | 20 categories | 498 |
| **Unseen (Test)** | Mug, Vase, Scissors | 917 |

The unseen categories are **never seen during training**.

---

## 🚀 Quick Start

### 1. Train the Model

```bash
python train.py \
  --data_dir ./data \
  --epochs 60 \
  --batch_size 8 \
  --lr 5e-4 \
  --occlusion_prob 0.5 \
  --use_focal_loss \
  --use_balanced_sampler
```

### 2. Evaluate (Four Conditions)

```bash
python evaluate.py \
  --checkpoint ./checkpoints/best_model.pth \
  --data_dir ./data \
  --condition all
```

### 3. Run Grasp Simulation (PyBullet)

```bash
python grasp_simulation/run_trials.py \
  --checkpoint ./checkpoints/best_model.pth \
  --num_trials 50
```

---

## 🔬 Reproducing the Ablation Studies

| Experiment | Command |
|------------|---------|
| **Full method** | `python train.py --occlusion_prob 0.5 --use_focal_loss --use_balanced_sampler` |
| **No occlusion** | `python train.py --occlusion_prob 0.0 --use_focal_loss --use_balanced_sampler` |
| **No focal loss** | `python train.py --occlusion_prob 0.5 --no_focal_loss --use_balanced_sampler` |
| **No balanced sampler** | `python train.py --occlusion_prob 0.5 --use_focal_loss --no_balanced_sampler` |
| **+ Geometric features (7D)** | `python train.py --input_dim 7` |
| **+ Extended features (13D)** | `python train.py --input_dim 13` |

---

## 📈 Results and Logs

All experimental results reported in the paper are available in the `results/` directory:

```
results/
├── main_results.csv
├── ablation_study.csv
├── per_class_iou.csv
└── grasp_simulation.csv
```

Training logs (loss curves, mIoU per epoch) are saved to `logs/` during training.

---

## 🖼️ Figures

| Figure | Description |
|--------|-------------|
| `figures/method_overview.pdf` | Overview of the proposed occlusion-aware training pipeline. |
| `figures/ablation_chart.pdf` | Ablation study bar chart (unseen-occluded mIoU). |
| `figures/qualitative_results.pdf` | Qualitative predictions on unseen objects (Mug, Vase). |
| `figures/failure_cases.pdf` | Failure cases for `grasp` affordance. |
| `figures/grasp_simulation.pdf` | Grasp execution in PyBullet. |

---

## 🙏 Acknowledgements

- **3D AffordanceNet** benchmark: [Deng et al., CVPR 2021](https://arxiv.org/abs/2103.04900)
- **DGCNN** architecture: [Wang et al., ACM TOG 2019](https://arxiv.org/abs/1801.07829)
- **PyBullet** simulator: [Coumans & Bai](https://pybullet.org/)

---

## 📄 License

This project is licensed under the **MIT License**—see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact:

**Ranjan Kumar Sahoo**  
PG Scholar, ASAC, Alliance University, Bengaluru, India  
📧 ranjansahoo4414@gmail.com  
🔗 [GitHub](https://github.com/RKSAHOO4414)
