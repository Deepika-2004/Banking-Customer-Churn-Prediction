# Banking-Customer-Churn-Prediction

## Overview
This project predicts customer churn for a bank using machine learning. The goal is to identify high-risk customers so that retention campaigns can be prioritized, improving customer retention and minimizing revenue loss.

- **Dataset Features:** credit_score, age, tenure, balance, products_number, credit_card, active_member, estimated_salary, country, gender  
- **Target:** churn (1 = churn, 0 = retained)  
- **Challenge:** Imbalanced dataset (minority class = churners)  

---

## Preprocessing
1. Dropped irrelevant columns (`customer_id`)  
2. Encoded categorical features (`country`, `gender`, `credit_card`, `active_member`) with One-Hot Encoding  
3. Scaled numerical features (`credit_score`, `age`, `tenure`, `balance`, `products_number`, `estimated_salary`)  
4. Handled class imbalance with **SMOTE**  
5. Optimized classification threshold using **precision-recall curves**  

---

## Models Used

### 1. Random Forest (RF)
- **Metrics (class 1):** Precision: 0.57, Recall: 0.70, F1: 0.63, PR-AUC: 0.688  
- **Business Perspective:**  
  - Catches most churners (high recall)  
  - Simpler to deploy and maintain  
  - Suitable if interpretability and fast retraining are priorities  

### 2. XGBoost
- **Metrics (class 1):** Precision: 0.57, Recall: 0.70, F1: 0.63, PR-AUC: 0.719  
- **Business Perspective:**  
  - High recall and highest PR-AUC → better ranking of high-risk customers  
  - Best choice for targeting retention campaigns effectively  
  - Slightly more complex than RF but improves probability-based decision-making  

### 3. Ensemble (Random Forest + XGBoost)
- **Metrics (class 1):** Precision: 0.66, Recall: 0.58, F1: 0.62, PR-AUC: 0.704  
- **Business Perspective:**  
  - Higher precision → more confident predictions, fewer false positives  
  - Slightly lower recall → misses some churners  
  - Useful if campaign budget is limited and prioritizing high-confidence customers is important  

---

## Workflow
1. Baseline model training (RF)  
2. Handle class imbalance with SMOTE  
3. Hyperparameter tuning with GridSearchCV  
4. Threshold optimization for F1-score  
5. Model evaluation using precision, recall, F1-score, PR-AUC  

---

## Key Insights
- XGBoost provides the **best trade-off between recall and probability ranking**  
- Ensemble improves precision but reduces recall slightly  

---

## Conclusion
For maximizing churn detection and targeting retention campaigns:  
- **XGBoost** is the recommended model for business deployment  
- **Random Forest** is a simpler alternative for faster retraining and interpretability  
- **Ensemble** can be used for higher confidence predictions when minimizing false positives is crucial  
