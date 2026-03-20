# MAGIC Gamma Telescope Classification: Grid Search vs. Optuna

## Project Overview

This repository contains an analysis of the **MAGIC Gamma Telescope Data Set**. The goal is to simulate the registration of high-energy gamma particles in a ground-based atmospheric Cherenkov gamma telescope.

The project solves a binary classification problem:
- **Class g (Gamma):** Signal (0)
- **Class h (Hadron):** Background (1)

The unique aspect of this repository is the comparison of two different hyperparameter tuning strategies:
1. **Multi-Model Randomized Search:** Comparing SVC, XGBoost, LightGBM, and Naive Bayes.
2. **Optuna Optimization:** Deep-dive tuning specifically for XGBoost using Bayesian optimization.

---

## Dataset

- **Source:** [MAGIC Gamma Telescope Data Set — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope)
- **File:** `magic04.csv` (19,019 observations, 11 columns)
- **Features:** 10 numerical features (fLength, fWidth, fSize, fConc, fConc1, fAsym, fM3Long, fM3Trans, fAlpha, fDist)
- **Target:** `class` (g = Gamma / h = Hadron)
- **Preprocessing:**
  - Label Encoding (g=0, h=1)
  - Train/Test Split (80/20)
  - Standard Scaling (applied within pipelines)

---

## Exploratory Data Analysis

EDA is powered by **[ydata-profiling](https://github.com/ydataai/ydata-profiling)**, generating an interactive report covering feature distributions, correlations, class imbalance, and missing value analysis.

```python
from ydata_profiling import ProfileReport

cols = ['fLength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans','fAlpha','fDist','class']
df = pd.read_csv('magic04.csv', names=cols)

profile = ProfileReport(df, title='MAGIC Gamma Telescope EDA Report', explorative=True)
profile.to_notebook_iframe()
```

---

## Methodologies

### Approach 1: Multi-Model Selection (`magic04.data-gridsearch.ipynb`)

This notebook focuses on breadth — evaluating multiple algorithms to find the best baseline model.

- **Algorithms:** Support Vector Classifier (SVC), XGBoost, LightGBM, Gaussian Naive Bayes
- **Technique:** `RandomizedSearchCV` (5 iterations per model)
- **Metric:** ROC AUC
- **Outcome:** Generated ROC Curves for all models, identified **LightGBM** and **XGBoost** as top performers, produced Confusion Matrix and Classification Report for the winner

### Approach 2: Advanced Tuning with Optuna (`magic04.data-optuna.ipynb`)

This notebook focuses on depth — taking the top model and pushing its performance further with Bayesian optimization.

- **Algorithm:** XGBoost Classifier
- **Technique:** `Optuna` (Tree-structured Parzen Estimator)
- **Search Space:** `learning_rate`, `max_depth`, `n_estimators`, `gamma`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`
- **Visualization:** Optimization History Plot, Hyperparameter Slice Plots, Parameter Importance

---

## Results Comparison

| Feature | Randomized Search | Optuna |
| :--- | :--- | :--- |
| **Strategy** | Breadth (Many Models) | Depth (One Model) |
| **Best Model** | LightGBM / XGBoost | XGBoost |
| **Tuning Logic** | Random sampling from grid | Bayesian optimization (learns from past trials) |
| **Accuracy** | ~88% | ~89% |
| **Macro Avg F1** | 0.87 | 0.88 |
| **Visualization** | ROC Curves | Hyperparameter Importance & Slice Plots |

---

## Key Findings

- **Class Imbalance:** The dataset is imbalanced (~2:1 Gamma to Hadron). Both notebooks handle this; Optuna explicitly tuned `scale_pos_weight`.
- **Model Performance:** While LightGBM showed slightly higher ROC AUC in the random search, the Optuna-tuned XGBoost achieved a robust 89% accuracy with strong precision/recall balance.
- **Optuna Utility:** Optuna revealed that `learning_rate` was the dominant hyperparameter, offering explainability that random search lacks.

---

## Project Structure

```
MAGIC-Gamma-Telescope/
├── LICENSE
├── README.md                          # This document
├── magic04.csv                        # Dataset
├── magic04.data-gridsearch.ipynb      # Approach 1: Multi-model RandomizedSearchCV
└── magic04.data-optuna.ipynb          # Approach 2: XGBoost + Optuna tuning
```

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Ishaq-ML/MAGIC-Gamma-Telescope-Classification-Grid-Search-vs.-Optuna.git
   cd MAGIC-Gamma-Telescope-Classification-Grid-Search-vs.-Optuna
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm optuna ydata-profiling
   ```

3. Launch Jupyter and run either notebook:
   ```bash
   jupyter notebook
   ```

---

## Libraries Used

- **Core:** `pandas`, `numpy`
- **EDA:** `ydata-profiling`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`
- **Optimization:** `optuna`

---

## License

This project is licensed under the **MIT License**.
