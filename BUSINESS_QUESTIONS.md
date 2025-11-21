# Project: Evolution of Online Science and Technology Communities on Reddit
**Team:** Chenxi Guo, Linjin He, Xiaoya Meng  
**Dataset:** Reddit comments and submissions (filtered for science, technology, and AI subreddits, ~hundreds of millions of rows)  
**High-Level Problem Statement:** How do online science and technology communities on Reddit evolve and shape public understanding and sentiment toward AI and emerging technologies?

---

## Question 1: How has community activity evolved across AI and technology subreddits over time?
**Analysis Type:** EDA
**Technical Approach:**
- Aggregate monthly post and comment counts for selected subreddits using Spark
- Visualize temporal trends to identify surges and lulls, especially in connection with technology/AI events

---

## Question 2: Which AI-related subreddits demonstrate the strongest user engagement and retention over time?
**Analysis Type:** EDA
**Technical Approach:**
- Calculate monthly ratios of returning vs. active users per subreddit to quantify stickiness
- Visualize retention and engagement as a heatmap across time and communities

---

## Question 3: How concentrated is attention within AI and tech discussions—are conversations dominated by a few topics or widely shared?
**Analysis Type:** EDA
**Technical Approach:**
- Calculate Gini index for comment distribution per post within subreddits
- Visualize inequality/concentration using violin and bar plots

---

## Question 4: How does public sentiment toward AI and emerging technologies fluctuate across time and events?
**Analysis Type:** EDA
**Technical Approach:**
- Perform sentiment analysis per subreddit/month using text mining techniques
- Visualize sentiment dynamics using box/strip plots and compare with event timelines

---

## Question 5: What are the dominant topics within Technology/Science/AI-related subreddits?

**Analysis Type:** NLP

**Technical Approach:**
- Preprocess text (tokenization, stopword removal, lemmatization)
- Apply TF-IDF + LDA topic modeling on post titles and comment bodies
- Extract top keywords and emerging themes (e.g., regulation, ethics, automation, creativity)
- Visualize topic distributions over time
- Expected outcome: Identify trending discussions and emerging themes

---

## Question 6: How does sentiment toward Technology/Science/AI-related vary across subreddits?

**Analysis Type:** NLP

**Technical Approach:**
- Apply sentiment analysis using Spark NLP or VADER on post and comment text
- Aggregate monthly sentiment scores by subreddit
- Visualize sentiment trends over time
- Expected outcome: Detect optimism, skepticism, or ethical concerns in different communities

---

## Question 7: How do sentiment and topic trends evolve across Reddit communities over the course of one year?

**Analysis Type:** NLP

**Technical Approach:**
- Align sentiment and topic time series with external event dates (model releases, policy debates)
- Detect changes in discussion volume, topic prevalence, or sentiment patterns
- Visualize before-and-after event comparisons
- Expected outcome: Understand impact of real-world events on community perception

---

## Question 8: How are ethical, technical, and societal topics represented in Technology/Science/AI-related subreddit discussions?

**Analysis Type:** NLP

**Technical Approach:**
- Classify posts/comments into discussion types using keyword/topic assignment
- Correlate type with engagement metrics (upvotes, awards, comment depth)
- Visualize engagement by discussion type using bar charts or boxplots
- Expected outcome: Identify which discussion types resonate most with the community

---

## Question 9: Can we automatically identify high-quality Reddit comments based solely on their textual content and simple metadata (e.g., length, presence of URLs, posting time), using classification methods on large-scale social media data?

**Analysis Type:** ML

**Technical Approach:**

A Spark ML pipeline performs tokenization, stopword removal, TF-IDF transformation, feature assembly, and trains a Logistic Regression classifier with class-imbalance weighting.

Model predictions are evaluated using scikit-learn, generating accuracy, F1, AUC, confusion matrix, and ROC/PR plots saved to structured output folders.

---

## Question 10: How can Reddit submissions in Tech/Science/AI domains be grouped into meaningful semantic clusters, and what linguistic patterns characterize each cluster?

**Analysis Type:** ML

**Technical Approach:**

This project applies large‐scale text processing using a Spark pipeline that performs tokenization, stopword removal, and TF-IDF feature extraction. K-Means clustering is then conducted to uncover latent thematic groupings within the comments, with the optimal number of clusters selected via the elbow method. Finally, PCA is used to project high-dimensional TF-IDF vectors into 2D space for visualization and interpretation of cluster structure.

---

## Summary

**EDA Questions:** 1, 2, 3, 4 (4 questions)  
**NLP Questions:** 5, 6, 7, 8 (4 questions)  
**ML Questions:** 9, 10 (2 questions)  

This set of questions covers statistical analysis, temporal patterns, text mining, sentiment analysis, and predictive modeling, providing a comprehensive roadmap for understanding AI and technology-related communities on Reddit.
