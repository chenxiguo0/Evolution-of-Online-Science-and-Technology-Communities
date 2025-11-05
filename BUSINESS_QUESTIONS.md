# Project: Evolution of Online Science and Technology Communities on Reddit
**Team:** Chenxi Guo Linjin He Xiaoya Meng  
**Dataset:** Reddit comments and submissions (filtered for science, technology, and AI subreddits, ~hundreds of millions of rows)  
**High-Level Problem Statement:** How do online science and technology communities on Reddit evolve and shape public understanding and sentiment toward AI and emerging technologies?

---

## Question 1: How has posting and commenting activity changed across major tech-related subreddits between June 2023 and July 2024?

**Analysis Type:** EDA

**Technical Approach:**
- Aggregate monthly post and comment counts using Spark `groupBy` on `year_month` and `subreddit`
- Visualize growth or decline patterns to identify community surges (e.g., around major AI releases like ChatGPT or Gemini)
- Expected outcome: Detect periods of rapid community growth or decline and understand posting trends

---

## Question 2: Which technology subreddits show the strongest user engagement and topic retention over time?

**Analysis Type:** EDA

**Technical Approach:**
- Track active and returning users per month
- Compute engagement ratios (e.g., comments per post, average score per user) to identify “sticky” vs. “flash” tech communities
- Visualize user retention over time using line charts
- Expected outcome: Identify communities that maintain long-term engagement

---

## Question 3: How concentrated is attention within AI and tech discussions?

**Analysis Type:** EDA

**Technical Approach:**
- Compute Gini index of post scores or comment counts per subreddit
- Measure whether engagement is dominated by a few popular threads (e.g., AI safety, LLMs) or distributed across topics
- Visualize distribution of attention using Lorenz curves
- Expected outcome: Quantify inequality of engagement across topics

---

## Question 4: Are there temporal or event-driven posting patterns?

**Analysis Type:** EDA

**Technical Approach:**
- Group posts/comments by date × hour × major AI events (NeurIPS, Google I/O, OpenAI announcements)
- Identify activity surges aligned with events
- Visualize using heatmaps or time-series plots
- Expected outcome: Understand how major events influence posting behavior

---

## Question 5: What are the dominant topics and trends within fast-growing AI-related subreddits?

**Analysis Type:** NLP

**Technical Approach:**
- Preprocess text (tokenization, stopword removal, lemmatization)
- Apply TF-IDF + LDA topic modeling on post titles and comment bodies
- Extract top keywords and emerging themes (e.g., regulation, ethics, automation, creativity)
- Visualize topic distributions over time
- Expected outcome: Identify trending discussions and emerging themes

---

## Question 6: How does sentiment toward AI systems vary across subreddits and over time?

**Analysis Type:** NLP

**Technical Approach:**
- Apply sentiment analysis using Spark NLP or VADER on post and comment text
- Aggregate monthly sentiment scores by subreddit
- Visualize sentiment trends over time
- Expected outcome: Detect optimism, skepticism, or ethical concerns in different communities

---

## Question 7: Do real-world AI milestones trigger noticeable sentiment shifts or topic changes?

**Analysis Type:** NLP

**Technical Approach:**
- Align sentiment and topic time series with external event dates (model releases, policy debates)
- Detect changes in discussion volume, topic prevalence, or sentiment patterns
- Visualize before-and-after event comparisons
- Expected outcome: Understand impact of real-world events on community perception

---

## Question 8: Which types of discussions (technical, ethical, societal) receive the most positive engagement?

**Analysis Type:** NLP + EDA

**Technical Approach:**
- Classify posts/comments into discussion types using keyword/topic assignment
- Correlate type with engagement metrics (upvotes, awards, comment depth)
- Visualize engagement by discussion type using bar charts or boxplots
- Expected outcome: Identify which discussion types resonate most with the community

---

## Question 9: Can we predict whether an AI-related post will become highly engaging?

**Analysis Type:** ML

**Technical Approach:**
- Engineer features from post metadata and text (posting time, subreddit, title sentiment, topic category, text complexity)
- Train a binary classification model (Random Forest or Gradient Boosting) using Spark MLlib
- Evaluate model with precision, recall, and F1-score
- Expected outcome: Predict high-engagement posts and guide content strategy

---

## Question 10: Can we forecast the next month’s public sentiment toward AI?

**Analysis Type:** ML

**Technical Approach:**
- Aggregate historical monthly sentiment by subreddit
- Train regression model (e.g., gradient boosting) to forecast sentiment trends
- Use feature importance to interpret drivers of sentiment changes
- Expected outcome: Predict short-term mood trends and identify influential factors

---

## Summary

**EDA Questions:** 1, 2, 3, 4 (4 questions)  
**NLP Questions:** 5, 6, 7, 8 (4 questions)  
**ML Questions:** 9, 10 (2 questions)  

This set of questions covers statistical analysis, temporal patterns, text mining, sentiment analysis, and predictive modeling, providing a comprehensive roadmap for understanding AI and technology-related communities on Reddit.
