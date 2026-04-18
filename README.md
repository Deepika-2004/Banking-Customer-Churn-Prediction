
# Banking Customer Churn Prediction

## Overview

Developed a machine learning model to predict customer churn in a retail bank, enabling prioritized retention campaigns and minimizing revenue loss.

* **Features:** 10 attributes (e.g., credit score, age, balance, etc.)
* **Target:** Churn (1 = churn, 0 = retained)
* **Challenge:** Highly imbalanced dataset (churners = minority)

---

## Data Preprocessing

* Dropped non-informative `customer_id`
* Encoded categorical variables (country, gender, credit card, active member) via One-Hot Encoding
* Scaled numerical features (credit score, age, balance, etc.)
* Addressed class imbalance using **SMOTE**
* Optimized classification threshold using precision-recall curves

---

## Models & Performance

### Random Forest

* **Precision:** 0.57 | **Recall:** 0.70 | **F1-score:** 0.63 | **PR-AUC:** 0.688
* High recall ensures churners are identified; a strong baseline model

### XGBoost

* **Precision:** 0.57 | **Recall:** 0.70 | **F1-score:** 0.63 | **PR-AUC:** 0.719
* Best PR-AUC → ranks high-risk customers most effectively
* Strong balance of recall and probability ranking

### LightGBM

* **Precision:** 0.60 | **Recall:** 0.65 | **F1-score:** 0.62 | **PR-AUC:** 0.703
* Fast training and efficient for large datasets
* Slightly better precision, balanced recall

### Ensemble (RF + XGBoost + LightGBM)

* **Precision:** 0.65 | **Recall:** 0.62 | **F1-score:** 0.63 | **PR-AUC:** 0.712
* Combines strengths: higher precision and balanced recall
* Useful for prioritizing high-confidence churn predictions

---

## Workflow

1. Built baseline model (Random Forest)
2. Applied **SMOTE** for class balance
3. Tuned hyperparameters with **GridSearchCV**
4. Optimized decision threshold (F1-score)
5. Evaluated with precision, recall, F1-score, PR-AUC

---

## Key Insights

* **XGBoost** and **LightGBM** provide the best balance between probability ranking and recall
* **LightGBM**'s speed advantage makes it ideal for large customer bases
* Ensemble model balances precision and recall for high-confidence targeting

---

## Conclusion

* Deploy **XGBoost** for best recall and probability ranking
* Use **LightGBM** for fast training on large datasets
* Random Forest as a simpler baseline, and ensemble for confident churn predictions

