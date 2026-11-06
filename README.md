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
├── figures/                                      # Figures generated from experiments
│   ├── fig_ablation_bar.pdf
│   ├── fig_ablation_bar1.pdf
│   ├── fig_method_overview.pdf
│   ├── fig_method_overview1.pdf
│   └── fig_method_overview2.pdf
│
├── final_model_success/                          # Final successful model
│   ├── 1.txt
│   ├── 3rd_research_experiment11.ipynb           # ← Notebook
│   └── best_geoaff.pth
│
├── model_success_1/                              # Earlier successful model
│   ├── 1.txt
│   ├── 3rd_research_experiment10.ipynb           # ← Notebook
│   ├── best_geoaff (1).pth
│   └── best_geoaff.pth
│
├── Experiment_17_Model_improved_Failed/          # Experiment 17 (failed improvement)
│   ├── 3rd_research_experiment17.ipynb           # ← Notebook
│   ├── best_improved.pth
│   ├── history_improved.json
│   └── training_curves_improved.png
│
├── ablation_experiment_model/                    # Ablation: XYZ 3D baseline
│   ├── 3rd_research_ablation_experiment12.ipynb  # ← Notebook
│   ├── best_xyz3d.pth
│   └── history_xyz3d.json
│
├── ablation_experimement_model_2/                # Ablation: Geometry 7D features
│   ├── 3rd_research_ablation_experiment13.ipynb  # ← Notebook
│   ├── best_geo7d (1).pth
│   ├── best_geo7d.pth
│   └── history_geo7d.json
│
├── ablation_1_no_occlusion_augmentation_experiment/  # Ablation: No occlusion augmentation
│   ├── 3rd_research_ablation_experiment14.ipynb  # ← Notebook
│   ├── best_abl_no_occ.pth
│   └── history_abl_no_occ.json
│
├── ablation_2_no_balanced_sampleing_experiment/  # Ablation: No balanced sampling
│   ├── 3rd_research_ablation_experiment15.ipynb  # ← Notebook
│   ├── best_abl_occ_no_bal.pth
│   └── history_abl_occ_no_bal.json
│
├── ablation_3_no_focal_loss_experiment/          # Ablation: No focal loss
│   ├── 3rd_research_ablation_experiment16 (1).ipynb  # ← Notebook
│   ├── best_abl_occ_no_focal.pth
│   └── history_abl_occ_no_focal.json
│
└── faild_model_Without_occlusion/                # Failed model without occlusion
    ├── 1.txt
    ├── 3rd_research_experiment6.ipynb            # ← Notebook
    ├── 3rd_research_experiment7.ipynb            # ← Notebook
    ├── 3rd_research_experiment8.ipynb            # ← Notebook
    ├── best_model_weighted (1).pth
    ├── best_model_weighted.pth
    ├── best_multitask.pth
    ├── best_part_detector.pth
    └── ... (other checkpoint files)
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
