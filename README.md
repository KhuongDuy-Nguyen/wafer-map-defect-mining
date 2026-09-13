# Wafer Map Defect Pattern Mining (WM-811K)
## Closed-Loop Semi-Supervised Learning & Novel Defect Discovery

[![Python](https://img.shields.io/badge/Python-3.10%2B-2b5b84?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x%20(GPU)-ff6f00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x%20%2F%2013.x-76b900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Dataset](https://img.shields.io/badge/Dataset-WM--811K-007acc?style=flat-square)](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)

---

## 1. BENCHMARK DATASET (WM-811K)

- **Dataset Scale**: 811,457 real-world semiconductor wafer maps from manufacturing fabs.
  - **Labeled Subset**: 172,950 wafers across 9 original classes (`Normal` ~85.2%, `Edge-Ring`, `Edge-Loc`, `Center`, `Loc`, `Scratch`, `Random`, `Donut`, `Near-full`).
  - **Unlabeled Subset**: 638,507 wafers (~78.7% of total dataset). 100,000 wafers randomly sampled for large-scale semi-supervised mining.
- **Matrix Encoding**: 2D discrete matrices with pixel values:
  - `0`: Background / Non-wafer area.
  - `1`: Normal / Functional die.
  - `2`: Defective die.
- **Dataset Acquisition**:
  - The raw dataset file **`LSWMD.pkl`** (~2.09 GB) is located in `data/LSWMD.pkl` (or automatically downloaded via Cell 5 in `data-mining.ipynb`).

---

## 2. SYSTEM PIPELINE ARCHITECTURE

```
[ Labeled Wafers (172.9k) ]                     [ Unlabeled Wafers (100k) ]
            │                                                │
 Stratified Split                                            │
 ├── Unbiased Test (10.0k, preserves 85.2% Normal)           │
 └── Balanced Train (40.8k, 1:1 Normal/Defects)              │
            │                                                │
 Resize 64x64 (Nearest)                          Resize 64x64 (Nearest)
            │                                                │
            ├──► PCA (128 dims, 51.01% Var) + DBI K-Means   │
            │                                                │
            └──► Multi-Scale Dilated CNN v1 ─────────────────┤
                 (Focal Loss gamma=2.0 + Rot90 Aug)          ▼
                                         Adaptive Pseudo-Labeling (Conf 0.80 - 0.95: ~2.7k)
                                         Uncertain Pool (89.3k) -> Isolation Forest (~8.9k)
                                         Two-Stage CURE Clustering -> Novel Morphological Class
                                                             │
                                                             ▼
                                         Retrain Multi-Scale CNN v2.0 (10 Classes)
                                         Spatial Heatmaps + KDE + t-SNE 2D
```

---

## 3. CORE ALGORITHMS & METHODOLOGY

| Algorithm | Technical Specification | Purpose & Scientific Rationale |
| :--- | :--- | :--- |
| **PCA** | 128 components, cumulative explained variance = 51.01% | Compresses flattened image vectors from $4096 ightarrow 128$ dimensions, eliminating pixel-level die noise while retaining principal variance. |
| **MiniBatchKMeans** | $K = 18$, mathematically validated via Davies-Bouldin Index | Evaluates $DBI = \frac{1}{K}\sum \max R_{ij}$ over $K \in [10, 22]$ to objectively validate $K=18$ rather than relying on subjective Elbow inspection. |
| **Multi-Scale Dilated CNN** | Parallel Dilated ($d=2$) + Asymmetric ($1\times 5, 5\times 1$) Convolutions | Resolves fine-grained detection bottlenecks for slender scratches (`Scratch`) and micro-clusters (`Loc`) via expanded multi-scale receptive fields. |
| **Categorical Focal Loss** | $\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t), \gamma=2.0$ | Down-weights abundant easy samples (`Normal`, `Edge-Ring`), focusing backpropagation gradients on hard-to-classify defect patterns. |
| **Class-Adaptive Pseudo-Labeling** | Dynamic Thresholds ($0.80 - 0.95$) | Assigns high-confidence pseudo-labels (~2,747 wafers) adaptively across classes to prevent label noise while mining rare defects. |
| **Isolation Forest** | $\text{contamination} = 0.10$, anomaly score $< 0$ | Isolates 8,927 anomalous wafers exhibiting irregular spatial patterns from the low-confidence pool (<60%). |
| **Two-Stage CURE Clustering** | Macro CURE ($K=5$) $\rightarrow$ Fine-grained Sub-clustering ($\alpha=0.2, p=12$) | Overcomes the single mega-cluster bottleneck, discovering distinct non-spherical failure patterns with mathematical validation ($N \ge 50$). |
| **Cluster Metrics Validation** | $\text{Silhouette} = 0.5610, \text{DBI} = 0.2692$ | Validates that discovered novel clusters are statistically compact, well-separated, and distinct from surrounding manufacturing noise. |
| **PCA $\rightarrow$ t-SNE** | PCA 50 dims $\rightarrow$ `TSNE(perplexity=35)` | Preserves global manifold topology to project high-dimensional defect representations into an interpretable 2D feature space. |

---

## 4. EXPERIMENTAL BENCHMARK RESULTS

Performance evaluated on the **Unbiased Factory Test Set (10,000 wafers)** reflecting real semiconductor fab distribution (85.2% Normal):

| Defect Class | Support | CNN v1.0 (Baseline) | CNN v2.0 (Closed-Loop Retrained) | Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Normal** | 8,524 | 0.02 (Rec: 0.01) | **0.97** (Prec: 0.98, Rec: 0.96) | **+0.95 F1 (Resolved distribution shift)** |
| **Center** | 248 | 0.57 | **0.88** (Prec: 0.86, Rec: 0.90) | **+0.31 F1** |
| **Edge-Ring** | 560 | 0.95 | **0.92** (Prec: 0.86, Rec: 0.98) | High sensitivity maintained |
| **Edge-Loc** | 300 | 0.24 | **0.65** (Prec: 0.68, Rec: 0.62) | **+0.41 F1** |
| **Loc** | 208 | 0.04 | **0.57** (Prec: 0.72, Rec: 0.48) | **+0.53 F1** |
| **Random** | 50 | 0.66 | **0.68** (Prec: 0.52, Rec: 0.98) | **+0.02 F1** |
| **Donut** | 32 | 0.66 | **0.73** (Prec: 0.67, Rec: 0.81) | **+0.07 F1** |
| **Near-full** | 9 | 0.82 | **0.50** (Prec: 0.57, Rec: 0.44) | Rare defect sample ($N=9$) |
| **Scratch** | 69 | 0.00 | **0.19** (Prec: 0.47, Rec: 0.12) | **+0.19 F1 (Multi-Scale Breakthrough)** |
| **Overall Accuracy** | **10,000** | **13.0%** | **94.0%** | **+81.0% Leap** |
| **Weighted F1** | **10,000** | **0.10** | **0.94** | **+0.84 Leap** |
| **Macro F1** | **10,000** | **0.44** | **0.61** | **+0.17 Leap** |

---

## 5. SETUP & EXECUTION

### 5.1. Environment Installation

```bash
# Install TensorFlow with full CUDA/cuDNN support on Linux
pip install --upgrade "tensorflow[and-cuda]"

# Install data science, computer vision, and clustering dependencies
pip install pandas numpy scipy scikit-learn matplotlib seaborn opencv-python gdown pyclustering
```

### 5.2. Execution Workflow
1. Ensure `LSWMD.pkl` is placed in `data/` directory (or download automatically via Cell 5 in `data-mining.ipynb`).
2. Open `data-mining.ipynb`, select your Python kernel, and execute cells sequentially from 1 to 54.

---

## 6. REPOSITORY STRUCTURE

```text
.
├── data-mining.ipynb          # End-to-end experimental notebook (54 cells, 9 phases)
├── README.md                  # Project technical documentation & benchmark report
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Excludes raw datasets & binary caches (>100MB)
└── data/                      # Benchmark dataset directory (LSWMD.pkl)
```

---

## 7. REFERENCES

1. **Wu, M. J., Jang, J. S. R., & Chen, J. L. (2014)**. *Wafer map failure pattern recognition and similarity ranking for large-scale semiconductor manufacturing*. IEEE Transactions on Semiconductor Manufacturing, 28(1), 1-12.
2. **Guha, S., Rastogi, R., & Shim, K. (1998)**. *CURE: an efficient clustering algorithm for large databases*. ACM SIGMOD Record, 27(2), 73-84.
3. **Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008)**. *Isolation forest*. In 2008 Eighth IEEE International Conference on Data Mining (pp. 413-422). IEEE.
4. **Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017)**. *Focal loss for dense object detection*. IEEE ICCV.
5. **Davies, D. L., & Bouldin, D. W. (1979)**. *A cluster separation measure*. IEEE TPAMI.
6. **van der Maaten, L., & Hinton, G. (2008)**. *Visualizing data using t-SNE*. JMLR.
