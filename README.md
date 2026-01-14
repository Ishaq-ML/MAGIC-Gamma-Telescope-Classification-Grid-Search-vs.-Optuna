# MAGIC Gamma Telescope Classification: Grid Search vs. Optuna

## Project Overview
This repository contains an analysis of the **MAGIC Gamma Telescope Data Set**. The goal is to simulate the registration of high-energy gamma particles in a ground-based atmospheric Cherenkov gamma telescope.

The project solves a binary classification problem:
*   **Class g (Gamma):** Signal (0)
*   **Class h (Hadron):** Background (1)

The unique aspect of this repository is the comparison of two different hyperparameter tuning strategies:
1.  **Multi-Model Randomized Search:** Comparing SVC, XGBoost, LightGBM, and Naive Bayes.
2.  **Optuna Optimization:** Deep-dive tuning specifically for XGBoost using Bayesian optimization.

## Dataset
*   **Source:** MAGIC Gamma Telescope Data Set (UCI Machine Learning Repository)
*   **Features:** 10 numerical features (fLength, fWidth, fSize, fConc, etc.)
*   **Target:** Class (g/h)
*   **Preprocessing:**
    *   Label Encoding (g=0, h=1)
    *   Train/Test Split (80/20)
    *   Standard Scaling (applied within pipelines)

## Methodologies

### Approach 1: Multi-Model Selection (`gridsearch.ipynb`)
This notebook focuses on breadth. It evaluates four distinct algorithms to find the best baseline model.
*   **Algorithms:** Support Vector Classifier (SVC), XGBoost, LightGBM, Gaussian Naive Bayes.
*   **Technique:** `RandomizedSearchCV` (5 iterations per model).
*   **Metric:** ROC AUC.
*   **Outcome:**
    *   Generated ROC Curves for all models.
    *   identified **LightGBM** and **XGBoost** as top performers.
    *   Produced Confusion Matrix and Classification Report for the winner.

### Approach 2: Advanced Tuning with Optuna (`optuna.ipynb`)
This notebook focuses on depth. It takes one of the top models (XGBoost) and uses advanced Bayesian optimization to maximize performance.
*   **Algorithm:** XGBoost Classifier.
*   **Technique:** `Optuna` (Tree-structured Parzen Estimator).
*   **Search Space:** Tuned `learning_rate`, `max_depth`, `n_estimators`, `gamma`, `min_child_weight`, `subsample`, `colsample_bytree`, and Regularization terms (`reg_alpha`, `reg_lambda`).
*   **Visualization:**
    *   Optimization History Plot.
    *   Hyperparameter Slice Plots.
    *   Parameter Importance (identified `learning_rate` and `max_depth` as most critical).

## Results Comparison

| Feature | Randomized Search (Notebook 1) | Optuna (Notebook 2) |
| :--- | :--- | :--- |
| **Strategy** | Breadth (Many Models) | Depth (One Model) |
| **Best Model** | LightGBM / XGBoost | XGBoost |
| **Tuning Logic** | Random sampling from grid | Bayesian optimization (learns from past trials) |
| **Accuracy** | ~88% | ~89% |
| **Macro Avg F1** | 0.87 | 0.88 |
| **Visualization** | ROC Curves | Hyperparameter Importance & Slice Plots |

## Libraries Used
*   **Core:** `pandas`, `numpy`
*   **Visualization:** `matplotlib`, `seaborn`
*   **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`
*   **Optimization:** `optuna`

## How to Run
1.  Install dependencies:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm optuna
    ```
2.  Ensure `magic04.data` is in the project directory.
3.  Run the notebooks:
    *   Start with **Approach 1** to see model selection.
    *   Run **Approach 2** to see how Optuna optimizes the specific XGBoost parameters.

## Key Findings
*   **Class Imbalance:** The dataset is imbalanced (approx 2:1 Gamma to Hadron). Both notebooks handle this; Optuna explicitly tuned `scale_pos_weight`.
*   **Model Performance:** While LightGBM showed slightly higher ROC AUC in the random search, the Optuna-tuned XGBoost achieved a robust 89% accuracy with high precision/recall balance.
*   **Optuna Utility:** Optuna provided significant insight into *which* hyperparameters mattered most (`learning_rate` was the dominant factor), offering explainability that the random search lacked.
