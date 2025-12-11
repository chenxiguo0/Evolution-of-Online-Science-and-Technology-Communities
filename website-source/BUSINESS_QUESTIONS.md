# Project: Evolution of Online Science and Technology Communities on Reddit
**Team:** Chenxi Guo, Linjin He, Xiaoya Meng  
**Dataset:** Reddit comments and submissions (filtered for science, technology, and AI subreddits, ~hundreds of millions of rows)  
**High-Level Problem Statement:** How do online science and technology communities on Reddit evolve, interact, and structure their discussions around AI and emerging technologies?

---

## Question 1: How has community activity evolved across science, technology, and AI subreddits over time?
**Analysis Type:** EDA
**Technical Approach:**
- Use Spark to load S3 parquet data (submissions + comments), filter target subreddits and date range.
- Aggregate posts and comments by subreddit and month (distinct ids) and compute total activity per month.
- Export CSV and visualize monthly activity trends (line charts) to compare engagement patterns and identify rapid growth or decline periods.

---

## Question 2: Which technology-related subreddits demonstrate the strongest user engagement and retention over time?
**Analysis Type:** EDA
**Technical Approach:**
- Use Spark to extract distinct active users per subreddit and month (union of comments and submissions authors).
- Compute monthly returning-to-active user ratio using set intersections across adjacent months (window lag) as a proxy for retention.
- Export CSV and visualize retention with a heatmap to highlight communities with consistent returning activity.

---

## Question 3: How concentrated is attention within technology-related discussions (comments and scores)?
**Analysis Type:** EDA
**Technical Approach:**
- Use Spark to compute per-post statistics (comments per post and post scores) for each subreddit and month.
- Compute Gini coefficients over collected lists (comments per post, and separately post scores) via a UDF to measure attention concentration.
- Export results and visualize distributions with violin plots and sorted bar charts comparing average concentration across subreddits.

---

## Question 4: Do science, technology, and AI subreddits share overlapping user communities?
**Analysis Type:** EDA
**Technical Approach:**
- Use Spark to collect unique authors per subreddit (from comments and submissions).
- Compute pairwise Jaccard similarity (size of intersection / size of union) between subreddit user sets.
- Export the similarity matrix and visualize it as a heatmap to reveal shared vs. distinct audiences.

---

## Question 5: What are the dominant topics and trends within fast-growing technology-related subreddits?

**Analysis Type:** NLP
**Technical Approach:**
- Combine submissions (title + selftext) and comments, preprocess text (lowercasing, URL removal, non-letter removal, tokenization, stopword removal).
- Use Spark ML: CountVectorizer (term counts) → IDF (TF-IDF-style weighting) → LDA to discover topics.
- Extract top keywords per topic and analyze topic prevalence over time and across subreddits; save topic trends and topic-by-subreddit tables to CSV/S3 and generate visualizations (topic trends, heatmaps, keyword bars, pie charts).
- Expected outcome: Identify trending discussions and emerging themes (e.g., regulation, ethics, tooling, applications).

---

## Question 6: What are the baseline emotional patterns of discussions about AI and emerging technologies?

**Analysis Type:** NLP
**Technical Approach:**
- Preprocess and combine comments and submissions; apply VADER (VaderSentiment / NLTK VADER) via a Spark UDF to compute compound sentiment scores per document.
- Aggregate monthly or subreddit-level average sentiment scores and sentiment-category distributions; export CSVs and visualize trends, subreddit comparisons, and stacked sentiment distributions/heatmaps.
- Expected outcome: Detect optimism, skepticism, or ethical concerns and how they vary by community and time.

---

## Question 7: How do external technological or policy events disrupt or reshape existing discussion patterns in online technology-related communities?

**Analysis Type:** NLP

**Technical Approach:**
- Align time series of sentiment (from Q6) and topic prevalence (from Q5) with an event timeline (e.g., model releases, major product announcements, policy milestones).
- Detect before/after changes in discussion volume, dominant topics, or sentiment averages; visualize event-aligned comparisons (vertical event markers on time series plots).
- Expected outcome: Understand the short-term and medium-term impact of external events on community perception and topic emphasis.

---

## Question 8: How do users shape topic emphasis and sentiment dynamics across science, technology, and AI subreddits?

**Analysis Type:** NLP
**Technical Approach:**
- Use rule-based keyword/topic assignment to classify documents into discussion types (example implementation: "technical", "ethical", "societal", "other").
- Aggregate counts and generate visual summaries (word clouds per class, frequency by subreddit, temporal counts). The existing implementation focuses on keyword-based classification and visual inspection (word clouds); correlation with engagement metrics is possible but not currently implemented in the code.
- Expected outcome: Identify which discussion types dominate different subreddits and their representative keywords/semantic fields.

---

## Question 9: Can the Quality of Reddit Comments in Science, Technology, and AI Subreddits Be Predicted from comment content and basic behavioral features?

**Analysis Type:** ML
**Technical Approach:**
- Label comments quality using a score threshold.
- Feature engineering includes text features produced by Tokenizer → StopWordsRemover → HashingTF → IDF (text_features), plus numeric/behavioral features (comment length, presence of URL, hour_of_day, day_of_week). Class weighting is applied to address imbalance.
- Train a Spark ML pipeline with Logistic Regression (implemented) and evaluate on test data with accuracy, precision, recall, F1, ROC/AUC, and confusion matrix; save model and output metrics.
- Expected outcome: Predict high-engagement/high-quality/low-quality comments and report model performance.

---

## Question 10: Can distinct discussion communities be identified within technology-related subreddits based on patterns of language use?

**Analysis Type:** ML
**Technical Approach (implemented):**
- Combine title and selftext to create document text, clean non-letter characters, and extract TF features using Tokenizer → StopWordsRemover → HashingTF → IDF (or CountVectorizer + IDF).
- Use K-Means clustering (implementation includes elbow method to choose K), apply PCA for 2D visualization, and save cluster assignments and a cluster analysis (cluster sizes and top terms per cluster). Optionally compute sentiment per cluster for interpretation (TextBlob used in the script for simple sentiment aggregation).
- Expected outcome: Identify distinct discussion directions, visualize cluster structure, and provide representative keywords per cluster.

---

## Summary

**EDA Questions:** 1, 2, 3, 4 (4 questions)  
**NLP Questions:** 5, 6, 7, 8 (4 questions)  
**ML Questions:** 9, 10 (2 questions)  