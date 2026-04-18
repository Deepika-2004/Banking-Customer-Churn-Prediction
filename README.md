# Banking Customer Churn Prediction

## Overview

This project focuses on predicting customer churn in a banking environment using machine learning techniques. The objective is to identify customers at high risk of leaving, enabling targeted retention strategies and reducing potential revenue loss.

* **Target Variable:** Churn (1 = exited, 0 = retained)
* **Key Features:** Credit score, age, tenure, balance, number of products, credit card status, activity status, estimated salary, geography, and gender
* **Core Challenge:** Class imbalance in churn labels

---

## Data Preprocessing

* Removed non-informative identifiers (e.g., customer ID)
* Applied **one-hot encoding** to categorical variables (geography, gender)
* Scaled numerical features using standardization techniques
* Addressed class imbalance using **SMOTE (Synthetic Minority Oversampling Technique)**
* Optimized classification threshold using **precision-recall trade-off analysis**

---

## Modeling Approach

### Random Forest

* Achieved strong performance with balanced precision and recall
* Effective baseline model with robust handling of feature interactions
* Easier to interpret and faster to deploy in production settings

### XGBoost

* Delivered the **best overall performance**, particularly in ranking high-risk customers
* Achieved the highest **PR-AUC (~0.72)**, indicating superior precision-recall balance
* Suitable for business scenarios requiring accurate prioritization of churn risk

### Ensemble Model (RF + XGBoost)

* Improved **precision (~0.64)**, reducing false positives
* Slight trade-off in recall compared to individual models
* Useful when targeting high-confidence churn predictions under budget constraints

---

## Workflow

1. Exploratory Data Analysis (EDA)
2. Data preprocessing and feature transformation
3. Baseline model development (Random Forest)
4. Handling class imbalance with SMOTE
5. Hyperparameter tuning using **GridSearchCV**
6. Threshold optimization based on F1-score and PR curves
7. Model comparison using precision, recall, F1-score, and PR-AUC

---

## Key Insights

* **XGBoost outperforms other models** in identifying and ranking churn-prone customers
* Precision-recall optimization significantly improves business decision-making over accuracy alone
* Handling class imbalance is critical for improving recall of churners

---

## Conclusion

* **XGBoost** is the most effective model for churn prediction and customer targeting
* **Random Forest** provides a strong, interpretable baseline
* **Ensemble approach** is beneficial when prioritizing prediction confidence

---

## Tech Stack

* **Languages:** Python
* **Libraries:** pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn
* **Techniques:** SMOTE, GridSearchCV, threshold tuning, ensemble modeling

