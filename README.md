# Customer Churn Prediction using k-Nearest Neighbors (Similarity-Based Learning)

## Problem Statement
Customer churn is a critical business problem in the telecom industry, where retaining existing customers is often more cost-effective than acquiring new ones.  
The goal of this project is to predict whether a customer is likely to churn based on demographic, service usage, and billing information.

---

## Dataset
- **Source:** Telco Customer Churn dataset (Kaggle - Dataset access: https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers × 21 features
- **Target:** `Churn` (1 = churn, 0 = no churn)
- **Class distribution:**
  - No churn: ~73%
  - Churn: ~27%

The dataset contains a mix of numeric and categorical features, reflecting real-world customer data.

---

## Approach

### Model Choice
This project uses **k-Nearest Neighbors (k-NN)**, an instance-based learning algorithm that predicts churn by comparing a customer to similar customers in feature space.

k-NN was chosen to:
- Demonstrate similarity-based learning
- Highlight the importance of preprocessing and scaling
- Serve as a strong, interpretable baseline model

---

### Data Preprocessing
- Dropped identifier column (`customerID`)
- Converted `TotalCharges` to numeric and imputed missing values
- Encoded the target variable as binary
- Used:
  - **StandardScaler** for numeric features
  - **OneHotEncoder** for categorical features
- All preprocessing and modeling steps were combined using a **scikit-learn Pipeline** to prevent data leakage

---

### Model Training
- Stratified train/test split (80/20)
- Cross-validation used for reliable performance estimation
- Evaluation metric selection aligned with business goals

---

## Model Evaluation

### Baseline k-NN
- **ROC-AUC (CV):** ~0.77
- **ROC-AUC (Test):** ~0.77

---

### Tuned k-NN (Recall-Focused)
Hyperparameters were optimized using cross-validation with **recall** as the primary metric.

- **Best parameters:**
  - `n_neighbors = 25`
  - `weights = uniform`
- **ROC-AUC (Test):** **0.83**

---

### Threshold Tuning
Instead of using the default 0.5 probability threshold, the decision threshold was adjusted to balance recall and precision.

This allows the model to:
- Identify more at-risk customers
- Limit unnecessary retention actions

---

## Final model configuration
- Algorithm: k-Nearest Neighbors
- n_neighbors: 25
- weights: uniform
- Decision threshold: 0.45

## Final Performance (Test Set)

- **Recall (Churn = 1):** 0.64
- **Precision (Churn = 1):** 0.58
- **ROC-AUC:** 0.83
- **F1-score (Churn = 1):** 0.61
- **Overall accuracy:** 0.78

**Confusion Matrix:**
- True Positives (correctly identified churners): 239
- False Negatives (missed churners): 135
- False Positives (non-churners flagged as churn): 170
- True Negatives: 865

---

## Interpretation
By tuning both hyperparameters and the decision threshold, the model increased churn recall from the baseline while maintaining reasonable precision. The final operating point identifies approximately 64% of customers who eventually churn, enabling more effective retention targeting without excessive false positives.

## Business Insights
- Customers with shorter tenure are significantly more likely to churn
- Monthly charges and contract type strongly influence churn behavior
- Similarity-based models can effectively support targeted retention strategies

---

## Limitations
- k-NN does not scale well to very large datasets
- Inference time increases with dataset size
- More advanced models (e.g. gradient boosting) may achieve higher recall

---

## Next Steps
- Compare against tree-based and boosting models
- Incorporate cost-sensitive learning
- Deploy the model as an API for real-time scoring

---

## How to Run

```bash
pip install -r requirements.txt
