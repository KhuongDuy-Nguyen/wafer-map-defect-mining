# Wafer Map Defect Pattern Mining (WM-811K)
## Closed-Loop Semi-Supervised Learning & Novel Defect Discovery

[![Python](https://img.shields.io/badge/Python-3.10%2B-2b5b84?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x%20(GPU)-ff6f00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x%20%2F%2013.x-76b900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Dataset](https://img.shields.io/badge/Dataset-WM--811K-007acc?style=flat-square)](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)

---

## 1. BENCHMARK DATASET (WM-811K)

- **Dataset Scale**: 811,457 real-world semiconductor wafer maps.
  - **Labeled Subset**: 172,950 wafers across 9 original classes (`Normal` ~85.2%, `Edge-Ring`, `Edge-Loc`, `Center`, `Loc`, `Scratch`, `Random`, `Donut`, `Near-full`).
  - **Unlabeled Subset**: 638,507 wafers (~78.7% of total dataset). 100,000 wafers randomly sampled for large-scale semi-supervised mining.
- **Matrix Encoding**: 2D discrete matrices with pixel values:
  - `0`: Background / Non-wafer area.
  - `1`: Normal / Functional die.
  - `2`: Defective die.
- **Dataset Acquisition**:
  - The raw dataset file **`LSWMD.pkl`** (~2.09 GB) is **not bundled in the repository** due to file size limits. It must be acquired prior to execution:
    - **Option 1 (Automated)**: Cell 5 in `data-mining.ipynb` automatically downloads the benchmark file from Google Drive via `gdown`.
    - **Option 2 (Manual)**: Download from Kaggle [WM-811K Dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) and place `LSWMD.pkl` into the repository root:
      ```bash
      kaggle datasets download -d qingyi/wm811k-wafer-map --unzip
      ```

---

## 2. SYSTEM PIPELINE ARCHITECTURE

```
[ Labeled Wafers (172.9k) ]                    [ Unlabeled Wafers (100k) ]
            │                                               │
 Stratified Split (80/20)                                   │
 ├── Test (34.5k, preserves 85.2% Normal)                   │
 └── Train (Balanced 1:1 Normal/Defects)                    │
            │                                               │
 Resize 64x64 (Nearest)                         Resize 64x64 (Nearest)
            │                                               │
            ├──► PCA (128 dims) + K-Means (K=18, DBI min)   │
            │                                               │
            └──► Train CNN v1 (3-Block + Data Aug) ─────────┤
                                                            ▼
                                           Pseudo-Labeling (Conf >= 0.98: ~11.4k)
                                           Uncertain Pool (88.6k) -> Isolation Forest (~4.4k)
                                           CURE Clustering -> Novel Class: Horizontal_Stripes (N >= 50)
                                                            │
                                                            ▼
                                           Retrain CNN v2.0 (10 Classes)
                                           Spatial Heatmaps + KDE + t-SNE 2D
```

---

## 3. CORE ALGORITHMS

| Algorithm | Technical Specification | Purpose & Scientific Rationale |
| :--- | :--- | :--- |
| **PCA** | 128 components, retains >75% variance | Compresses flattened image vectors from $4096 \rightarrow 128$ dimensions, eliminating pixel-level die noise and accelerating clustering. |
| **MiniBatchKMeans** | $K = 18$, optimized via Davies-Bouldin Index (DBI) | Minimizes $DBI = \frac{1}{K}\sum \max R_{ij}$ to quantitatively validate $K=18$ rather than relying on subjective Elbow visual inspection. |
| **Upgraded CNN v1** | 3-block `Conv2D(32/64/128) + BatchNorm` + Circular Augmentation | Resolves fine-grained detection bottlenecks for `Scratch` (slender scratches) and `Loc` (micro-clusters) via deeper receptive fields and rotation symmetry. |
| **Pseudo-Labeling** | Softmax Confidence threshold $\ge 0.98$ | Automatically assigns high-confidence pseudo-labels to ~11,400 unlabeled wafers to augment the training pool. |
| **Isolation Forest** | $\text{contamination} = 0.05$, score $< 0$ | Isolates ~4,400 anomalous wafers with irregular spatial patterns from the uncertain subset. |
| **CURE Clustering** | Shrinkage factor $\alpha = 0.3$, size threshold $N \ge 50$ | Identifies non-spherical, elongated cluster geometries; discovers the 10th failure pattern: `Horizontal_Stripes` while filtering noise. |
| **PCA $\rightarrow$ t-SNE** | PCA 50 dims $\rightarrow$ `TSNE(perplexity=35)` | Overcomes the Curse of Dimensionality, preserving global manifold geometry to visualize clear separation of 10 defect classes in 2D. |

---

## 4. SETUP & EXECUTION

### 4.1. Environment Installation

```bash
# Install TensorFlow with full CUDA/cuDNN support on Linux
pip install --upgrade "tensorflow[and-cuda]"

# Install data science, computer vision, and clustering dependencies
pip install pandas numpy scipy scikit-learn matplotlib seaborn opencv-python gdown pyclustering
```

### 4.2. Local GPU Activation (`data-mining.ipynb`)

Cell 1 dynamically binds CUDA shared libraries via `ctypes` to prevent dynamic linker (`dlopen`) errors on Linux environments:

```python
import os, sys, glob, ctypes

for lib_dir in glob.glob(os.path.join(sys.prefix, "lib", "python*", "site-packages", "nvidia", "*", "lib")):
    for so in glob.glob(os.path.join(lib_dir, "*.so*")):
        try:
            ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass

import tensorflow as tf
print("Available GPUs:", tf.config.list_physical_devices("GPU"))
```

### 4.3. Execution Workflow
1. Ensure `LSWMD.pkl` is located in `data/` (or repository root, or run Cell 5 to automatically download via `gdown`).
2. Open `data-mining.ipynb`, click **Restart Kernel**, and execute cells sequentially from 1 to 54.

---

## 5. REPOSITORY STRUCTURE

```text
.
├── data-mining.ipynb          # End-to-end experimental notebook (54 cells, 9 phases)
├── README.md                  # Project technical documentation & execution guide
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Excludes raw datasets & binary caches (>100MB)
└── data/                      # Benchmark dataset directory (LSWMD.pkl excluded from git)
```

---

## 6. REFERENCES

1. **Wu, M. J., Jang, J. S. R., & Chen, J. L. (2014)**. *Wafer map failure pattern recognition and similarity ranking for large-scale semiconductor manufacturing*. IEEE Transactions on Semiconductor Manufacturing, 28(1), 1-12.
2. **Guha, S., Rastogi, R., & Shim, K. (1998)**. *CURE: an efficient clustering algorithm for large databases*. ACM SIGMOD Record, 27(2), 73-84.
3. **Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008)**. *Isolation forest*. In 2008 Eighth IEEE International Conference on Data Mining (pp. 413-422). IEEE.
4. **Davies, D. L., & Bouldin, D. W. (1979)**. *A cluster separation measure*. IEEE Transactions on Pattern Analysis and Machine Intelligence, (2), 224-227.
5. **van der Maaten, L., & Hinton, G. (2008)**. *Visualizing data using t-SNE*. Journal of Machine Learning Research, 9(11), 2579-2605.
6. **Han, J., Kamber, M., & Pei, J. (2011)**. *Data mining: concepts and techniques*. Morgan Kaufmann.
