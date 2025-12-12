# Exploratory Data Analysis (EDA) Findings

## Introduction

This document summarizes the key results and insights from the exploratory data analysis stage of our project, focused on AI and technology communities on Reddit (June 2023 – July 2024). We leverage Spark and Python to analyze posting, engagement, discussion concentration, and public sentiment at true massive scale, grounding all claims in real processed data and visualizations.

## Data Overview
- **Time range:** 2023-06-01 to 2024-07-31
- **Dataset size:** Reddit comments and submissions, filtered for science, technology, AI-related subreddits
- **Scale:** ~12.5M comments, ~0.8M submissions ([dataset_summary_final.csv](data/csv/dataset_summary_final.csv))
- **Top active subreddits:** See [subreddit_statistics_final.csv](data/csv/subreddit_statistics_final.csv)

---

## Business Question 1. How has community activity evolved across science, technology, and AI subreddits over time?
**Method:** We aggregate monthly posts and comments for key AI/tech-related subreddits, plotted longitudinally ([rq1_activity.csv](data/plots/rq1_activity.csv)).

**Visualization:**
![Activity Trends](data/plots/rq1_activity_trends.png)

**Findings:**
- Activity in subreddits like ChatGPT, technology, ArtificialInteligence, and Futurology exhibits clear temporal fluctuations.
- Notable spikes in early/mid-2024 appear correlated with major LLM launches and industry news, as captured by dramatic increases in both posts and comments.

---

## Business Question 2. Which technology-related subreddits demonstrate the strongest user engagement and retention over time?
**Method:** For selected subreddits, we compute the ratio of returning (previously active) users to all monthly active users, visualized via a heatmap ([rq2_engagement.csv](data/plots/rq2_engagement.csv)).

**Visualization:**
![User Engagement Heatmap](data/plots/rq2_engagement_heatmap.png)

**Findings:**
- Subreddits like MachineLearning and ChatGPT maintain higher and more stable retention ratios, suggesting strong community “stickiness.” Smaller subreddits display lower or more volatile engagement.
- Some AI/tech communities have a high proportion of one-time contributors, whereas others foster ongoing conversations.

---

## Business Question 3. How concentrated is attention within technology-related discussions (comments and scores)?
**Method:** We calculate the Gini coefficient (measuring inequality) for comment distribution per post, both per subreddit/month and on average ([rq3_gini.csv](data/plots/rq3_gini.csv)).

**Visualizations:**
- Violin plot of Gini by subreddit: ![Gini Violin](data/plots/rq3_gini_violin.png)
- Sorted bar chart of average Gini: ![Average Gini Bar](data/plots/rq3_gini_bar_sorted.png)

**Findings:**
- Subreddits such as technology and ChatGPT show the highest attention concentration (Gini~0.9+), meaning most engagement is absorbed by a few highly popular threads.
- Communities like datascience and Futurology exhibit more egalitarian discussion patterns, with attention spread more broadly among posts.

---

## Business Question 4. Do science, technology, and AI subreddits share overlapping user communities?
**Method:** We compute pairwise Jaccard similarity between subreddit user sets and visualize the resulting matrix as a heatmap to reveal shared vs. distinct audiences ([rq4_user_overlap_matrix.csv](data/csv/rq4_user_overlap_matrix.csv)).

**Visualization:**
![User Overlap Heatmap](data/plots/rq4_user_overlap_heatmap.png)

**Findings:**
- Strong user overlap exists between core AI subreddits like `ChatGPT`, `OpenAI`, and `ArtificialInteligence`, indicating a shared, highly engaged user base.
- General technology subreddits (`technology`, `Futurology`) share audiences with AI communities but also maintain distinct user groups, acting as bridges in the ecosystem.
- Specialized communities like `robotics` show lower overlap, suggesting more niche audiences.

---

## Key Insights and Implications
- Community activity and engagement are highly event-responsive, with surges aligning with major technology milestones/language model launches.
- User retention and engagement patterns reveal both “sticky” AI forums and more transient, event-driven communities.
- Attention is very unequally distributed: dominant topics and posts monopolize discussion in some spaces, while others sustain more diverse conversation.
- Public sentiment is dynamic and event-sensitive, indicating the importance of tracking mood when analyzing tech/AI discussion spaces.

---

