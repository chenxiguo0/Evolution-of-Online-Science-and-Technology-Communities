# Machine Learning (ML) Findings

## Introduction

This document presents the Machine Learning (ML) component of our project, focusing on predictive modeling related to AI and technology discussions on Reddit. We leverage Apache Spark's MLlib for scalable model training and evaluation. This section addresses two core ML business questions, presenting the methodologies, model performance metrics, and key insights derived from our analysis.

---

## Research Question 9: Can we predict whether an AI-related post will become highly engaging?

**Business Question:**
Can we predict which AI-related posts will garner significant user engagement (e.g., high scores, many comments) based on their characteristics?

**Analysis Approach:**
We framed this as a binary classification problem. Features engineered from post metadata (e.g., subreddit, time of posting) and textual content (e.g., sentiment, topic categories from NLP) were used to train a Logistic Regression model. The model aims to classify posts into "highly engaging" or "not highly engaging."

**Features and training details:**
- Positive label definition: `score >= 6` (threshold used in `code/ml/ml_Q1.py`).
- Feature set: TF-IDF text features (HashingTF + IDF) plus `comment_length`, `has_url`, `hour_of_day`, and `day_of_week`.
- Train/validation/test split: ~80/20 overall with an internal train/val split (train_val -> train,val approximately 75/25).
- Class weighting: computed from training set and passed to LogisticRegression via `weightCol` to mitigate class imbalance.
- Model saved at: `code/ml/models/logistic_regression`.

**Visualizations and metrics**

### Confusion Matrix
- true_0 → pred_0: **949,839** ; pred_1: **1,112,372**
- true_1 → pred_0: **132,447** ; pred_1: **318,706**

![Confusion Matrix for Logistic Regression](data/plots/ML1_confusion_matrix_logistic_regression.png)

**confusion matrix:**
- The confusion matrix shows the counts of true/false positives and negatives. The model produces a large number of predicted positives (pred_1), resulting in many false positives (true_0→pred_1 = 1,112,372).
- Practical implication: the model favors sensitivity (finding most high-engagement posts) at the cost of precision; useful as a candidate selector but not for direct automated actions without further filtering.

### ROC Curve

![ROC Curve for Logistic Regression](data/plots/ML1_roc_logistic_regression.png)

**ROC:**
- The ROC curve displays the trade-off between true positive rate and false positive rate as the decision threshold varies. The computed AUC = **0.626** indicates modest discriminative ability — better than random guessing but limited for high-confidence decisions.
- Use the ROC to compare models or to select thresholds that balance sensitivity and specificity according to operational needs.

### Precision-Recall Curve

![Precision-Recall Curve for Logistic Regression](data/plots/ML1_pr_logistic_regression.png)

**Precision–Recall:**
- The PR curve focuses on precision vs recall, which is more informative for imbalanced problems. Reported precision = **0.223** and recall = **0.706**.
- This pattern (high recall, low precision) means the model detects most truly high-engagement posts but returns many false positives. For production, tune the classification threshold to match the desired trade-off (e.g., increase precision if human review cost is high).

**Model performance**
- Accuracy: **0.5047**
- Precision: **0.2227**
- Recall: **0.7064**
- F1-score: **0.3386**
- AUC: **0.6257**

**Interpretation:**
- High recall but low precision suggests the model is useful for identifying candidates but not for direct publication decisions. Improving precision is the primary engineering goal.

**Future Improvement:**
- Enrich text representations using contextual embeddings (BERT / Sentence-BERT) instead of or in addition to HashingTF+IDF.
- Add author- and subreddit-level features (author history, prior post scores, subreddit norms).
- Try ensemble or tree-based models (Gradient Boosting, XGBoost, LightGBM) and calibrate probabilities. Use PR curve-based threshold selection aligned to business goals.

---

## Research Question 10: Can we forecast the next month’s public sentiment toward AI?

**Business Question:**
Can we build a predictive model to forecast the overall public sentiment towards AI in the coming month within specific subreddits?

**Analysis Approach:**
This question is framed as a regression problem. We used historical monthly sentiment scores (derived from NLP analysis) as the target variable, along with aggregated features such as past activity levels, topic prevalence, and external event indicators. A regression model (e.g., Gradient Boosting) would typically be trained to predict the continuous sentiment score for the subsequent month.

**Completed work and available artifacts:**
- Clustering analysis of subreddit-level text/sentiment patterns is implemented in `code/ml/ml_Q2.py` and the model artifacts are saved under `code/ml/models/kmeans_k13`.
- Cluster analysis summary is available in `data/csv/ML2_cluster_analysis.csv` and visualized in `data/plots/ML2_cluster_visualization.png`.
- Elbow method chart (used to select K) is available: `data/plots/ML2_elbow_method.png`.

### Cluster summary (selected clusters)
- Largest cluster id: **10** (size = **43,558**) — frequent tokens include 'removed', 'im', 'ai', 'chatgpt', 'gpt', 'new', 'use', 'get'.
- Cluster **8** (size = **4,989**) — casual language cluster with tokens like 'im', 'like', 'time', 'ive', 'know'.
- Cluster **6** (size = **511**) — includes more technical tokens like 'ai', 'code', 'import', 'data'.

![Cluster Visualization of Subreddit Sentiment](data/plots/ML2_cluster_visualization.png)

**clustering:**
- Clustering reveals subreddit groups with similar language and sentiment patterns; cluster 10 dominates the corpus, likely representing high-volume or moderated/removed content.
- These clusters enable stratified modeling: forecasting models trained per-cluster or using cluster id as a feature may yield improved accuracy.

**Limitations & Future Improvement:**
- No supervised regression model for next-month sentiment is currently saved in the repository; the cluster analysis is preparatory work.
- To implement forecasting: aggregate lagged features, train regression models with cross-validation, evaluate using RMSE/MAE/R2, and interpret with SHAP to identify drivers.

---

## Summary

Our ML experiments demonstrate that classification models can assist in identifying potentially high-engagement posts, while clustering helps segment communities for refined downstream forecasting. However, further improvements in text representation, feature engineering, and model complexity are needed to raise precision and overall predictive utility before deploying automatic decision systems.

---