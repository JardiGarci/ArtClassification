# Interpretable Art Image Classification via Multifractal and Topological Analysis

**Master's thesis project · IPN-UPIITA · Advanced Technologies**

This repository implements an interpretable classification system for artwork images that combines **multifractal analysis** and **topological data analysis (TDA)** — two tools from complexity science — with XGBoost and SHAP explanations.

The key idea: instead of treating a CNN as a black box, every feature has a direct physical meaning (fractal dimension, Hurst exponent, persistence entropy…), so the model's decisions can be traced back to measurable structural properties of the painting.

---

## Results snapshot

| Task | Classes | Balanced Accuracy | AUC |
|---|---|---|---|
| Movement (multiclass) | 12 | 0.49 | — |
| Genre (multiclass) | 10 | 0.49 | — |
| Artist (multiclass) | 60 | 0.38 | — |
| Impressionism vs. rest | 2 | 0.76 | — |
| Van Gogh vs. rest | 2 | 0.77 | — |
| **Savrasov vs. Levitan** | 2 | **0.90** | **0.95** |
| **Landscape vs. Portrait** | 2 | **0.90** | **0.97** |
| **Cubism vs. Romanticism** | 2 | **0.92** | **0.98** |
| Claude Monet vs. Aivazovsky | 2 | — | — |

Random baseline: ≈ 0.083 (movement) · 0.100 (genre) · 0.017 (artist).

---

## Methods

### Feature extraction

Each painting is normalised to 1380 × 1380 px and analysed at three spatial scales: global (1×1), quadrant (2×2) and nonet (3×3). For each segment, three descriptors are computed:

| Method | Output | Key descriptors |
|---|---|---|
| **MF-DFA 2D** | 14 features / segment | Hurst exponent, Δα, α*, asymmetry index, τ(q) curvature |
| **MF-Rényi** | 14 features × 3 measures / segment | D₀, D₁, D₂, Δα, α* for intensity sum, variance and entropy |
| **TDA (H0/H1)** | 10 features / segment | Persistence entropy, mean/std/max lifetime, component and cycle counts |

### Classification pipeline

1. **Variance threshold** — remove near-zero variance features  
2. **StandardScaler + Isolation Forest** — outlier removal (5% contamination)  
3. **SMOTE** — per-fold oversampling to handle class imbalance  
4. **XGBoost** — 5-fold stratified cross-validation  
5. **SHAP TreeExplainer** — out-of-fold importance aggregation  

---

## Repository structure

```
ArtClassification/
├── utils.py                  # Image normalisation, segmentation, scale helpers
├── mfdfa.py                  # 2-D MF-DFA (low-level, image-by-image)
├── mfrenyi.py                # MF-Rényi box-counting (intensity/variance/entropy)
├── multifractal_functions.py # Batch pipelines: mf_dfa_features, mf_renyi_features,
│                             #   mf_combined_features (single scale loop)
├── tda.py                    # Persistent homology H0/H1 (Union-Find + numba)
├── tda_functions.py          # TDA feature extraction wrappers
├── validationtda.py          # Synthetic tests for H1 cycle computation
│
├── compute_features.ipynb    # Feature extraction over the full dataset
├── classification.ipynb      # Classification experiments + SHAP analysis
├── Validacion.ipynb          # Validation and ablation studies
├── pruebas.ipynb             # Exploratory experiments
└── DATA/                     # Feature files — not tracked in git (see .gitignore)
```

---

## Classification experiments (`classification.ipynb`)

The notebook runs the full pipeline for the following cases, each corresponding to a result in the thesis presentation:

| Case | `validos` | Slide |
|---|---|---|
| Impressionism vs. rest | `['Impressionism']` | p. 24 |
| Van Gogh vs. rest | `['vincent-van-gogh']` | p. 25 |
| Savrasov vs. Levitan | `['aleksey-savrasov', 'isaac-levitan']` | p. 26 |
| Landscape vs. Portrait | `['landscape', 'portrait']` | p. 27 |
| Cubism vs. Romanticism | `['Cubism', 'Romanticism']` | p. 28 |
| Monet vs. Aivazovsky | `['claude-monet', 'ivan-aivazovsky']` | p. 28 |

Switch between cases by uncommenting the corresponding `validos = [...]` line in Cell 14 of the notebook.

---

## Tech stack

- **Python 3.10+** · numpy · scipy · numba (JIT compilation)
- **Computer vision**: opencv-python
- **ML**: scikit-learn · xgboost · imbalanced-learn · shap
- **Dataset**: WikiArt (portraits, landscapes — via [SNIC](https://github.com/achanta/SNIC))

```bash
pip install numpy scipy numba opencv-python scikit-learn xgboost imbalanced-learn shap
```

---

## Background

Deep learning achieves ~79% accuracy on art style classification (Ma et al., 2025) but offers little interpretability. This project trades some accuracy for full transparency: every prediction can be attributed to specific spatial regions and physical descriptors of the painting's texture.

**Key findings from SHAP:**
- **TDA** captures brushstroke topology differences between artists with similar techniques.
- **MF-Rényi** detects intensity contrast patterns characteristic of portrait backgrounds and faces.
- **MF-DFA** reveals that Romanticism paintings have smoother textures in upper quadrants — consistent with their frequent depiction of skies and horizons.

---

*Jardi Yulistian García Bustamante · Director: Dr. Lev Guzmán Vargas · Co-director: Dr. Daniel Aguilar Velázquez*
