# Customer Churn Prediction & Segmentation

## Project Overview

This project focuses on analyzing customer behavior to predict churn and segment the customer base for strategic business insights. By using machine learning models and clustering techniques, the goal is to identify high-risk customers, understand key patterns among loyal and disengaged users, and provide actionable recommendations for customer retention and upselling.

---

## Objectives

1. Predict customer churn using historical profile, transaction, and engagement data.
2. Identify at-risk customers before they churn.
3. Segment customers into behavioral and demographic clusters.
4. Analyze RFM (Recency, Frequency, Monetary) metrics to assess customer value.
5. Translate analytical insights into strategic business actions.

---

## Dataset

The dataset contains anonymized customer-level data including:

- Demographics: Age, Gender, Marital Status
- Engagement Metrics: Login frequency, Support interactions
- Transaction Behavior: Number and amount of transactions
- Churn Label: Binary indicator of historical churn
- RFM Features: Derived metrics used for segmentation

---

## Key Techniques Used

- **Data Preprocessing**: Missing value treatment, encoding, scaling
- **Clustering**: KMeans for demographic segmentation
- **RFM Analysis**: Customer value assessment
- **Risk Flagging**: Business-defined risk categories
- **Random Forest Classifier**: For churn prediction
- **GridSearchCV**: Hyperparameter tuning
- **Threshold Optimization**: Based on precision-recall curve
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Feature Importance**: Interpretation of key drivers of churn

---

## Model Performance

- **Accuracy**: 78%
- **Precision (Churn)**: 77%
- **Recall (Churn)**: 80%
- **F1-Score (Churn)**: 81%
- **AUC-ROC**: Computed and visualized to assess model separability

These metrics reflect a highly effective model capable of correctly identifying a majority of churners while minimizing false positives.

---

## Insights Summary

### Demographic Segmentation:
- Majority of customers are under 30 but average age hovers in the 40s.
- Married and widowed customers are more likely to churn.
- Younger, single customers are more loyal and digitally engaged.

### RFM & Risk Segmentation:
- Champions and Loyal customers represent high value.
- At-Risk customers show declining engagement.
- Customers with low login frequency and lower transaction amounts tend to churn more.

### Churn Modeling:
- High model precision enables confident action on predictions.
- 73% of churners can be proactively identified for retention outreach.

---

## Recommendations

1. Launch retention programs for identified at-risk customers.
2. Enhance digital support experiences to reduce churn from remote banking users.
3. Design targeted campaigns for married and widowed demographics.
4. Focus growth efforts on young, digitally-active users with upsell opportunities.
5. Implement loyalty and referral programs for Loyal and Champion segments.

---

## Files in this Repository

- `data/` — Cleaned and raw datasets
- `notebooks/` — EDA, modeling, segmentation, and evaluation steps
- `artifacts/` — Saved model artifacts 
- `src/` — Python scripts for preprocessing, training, evaluation
- `output/` — Reports, figures, and model outputs
- `README.md` — This file
- `requirements.txt` — Python dependencies
