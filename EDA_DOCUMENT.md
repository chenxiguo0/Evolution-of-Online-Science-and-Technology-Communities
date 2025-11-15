# Exploratory Data Analysis (EDA) Findings

## Introduction

This document summarizes the key results and insights from the exploratory data analysis stage of our project, focused on AI and technology communities on Reddit (June 2023 – July 2024). We leverage Spark and Python to analyze posting, engagement, discussion concentration, and public sentiment at true massive scale, grounding all claims in real processed data and visualizations.

## Data Overview
- **Time range:** 2023-06-01 to 2024-07-31
- **Dataset size:** Reddit comments and submissions, filtered for science, technology, AI-related subreddits
- **Scale:** ~12.5M comments, ~0.8M submissions ([dataset_summary_final.csv](data/csv/dataset_summary_final.csv))
- **Top active subreddits:** See [subreddit_statistics_final.csv](data/csv/subreddit_statistics_final.csv)

---

## RQ1. How has community activity evolved across AI and technology subreddits over time?
**Method:** We aggregate monthly posts and comments for key AI/tech-related subreddits, plotted longitudinally ([rq1_activity.csv](data/csv/rq1_activity.csv)).

**Visualization:**
![Activity Trends](data/plots/rq1_activity_trends.png)

**Findings:**
- Activity in subreddits like ChatGPT, technology, ArtificialInteligence, and Futurology exhibits clear temporal fluctuations.
- Notable spikes in early/mid-2024 appear correlated with major LLM launches and industry news, as captured by dramatic increases in both posts and comments.
- The technology subreddit maintains the highest and most stable activity levels (200k-300k range), while specialized AI communities show more variable engagement patterns.

---

## RQ2. Which AI-related subreddits demonstrate the strongest user engagement and retention over time?
**Method:** For selected subreddits, we compute the ratio of returning (previously active) users to all monthly active users, visualized via a heatmap ([rq2_engagement.csv](data/csv/rq2_engagement.csv)).

**Visualization:**
![User Engagement Heatmap](data/plots/rq2_engagement_heatmap.png)

**Findings:**
- Specialized communities like Alethics demonstrate occasional high-retention periods (reaching ~1.0 in specific months), indicating strong user return behavior during peak interest phases.
- Larger mainstream subreddits such as ChatGPT, MachineLearning, and OpenAI display moderate but consistent engagement ratios (0.2-0.4), suggesting steady community participation with a mix of new and returning contributors.
- The heatmap reveals that user retention varies considerably across time and community type, with niche subreddits showing more volatile patterns compared to established general AI forums.

---

## RQ3. How concentrated is attention within AI and tech discussions—are conversations dominated by a few topics or widely shared?
**Method:** We calculate the Gini coefficient (measuring inequality) for comment distribution per post, both per subreddit/month and on average ([rq3_gini.csv](data/csv/rq3_gini.csv)).

**Visualizations:**
- Violin plot of Gini by subreddit: ![Gini Violin](data/plots/rq3_gini_violin.png)
- Sorted bar chart of average Gini: ![Average Gini Bar](data/plots/rq3_gini_bar_sorted.png)

**Findings:**
- Subreddits such as technology and ChatGPT show the highest attention concentration (Gini~0.88-0.90), meaning most engagement is absorbed by a few highly popular threads.
- Communities like MachineLearning and datascience exhibit more egalitarian discussion patterns (Gini~0.70-0.78), with attention spread more broadly among posts.
- This suggests that general-interest tech communities tend toward "viral hit" dynamics, while technical/professional communities foster more distributed engagement across multiple conversations.

---

## RQ4. How does public sentiment toward AI and emerging technologies fluctuate across time and events?
**Method:** Average sentiment scores are computed per month and subreddit using text sentiment models (see [rq4_sentiment.csv](data/csv/rq4_sentiment.csv)), visualized as box+strip plots.

**Visualization:**
![Sentiment Boxplot](data/plots/rq4_sentiment_boxplot.png)

**Findings:**
- Sentiment across all AI and tech communities remains predominantly neutral (median values clustering around 0), with limited variance in monthly averages.
- While the distribution is tightly centered, occasional outliers in specific months suggest isolated events or discussions that generate stronger emotional responses (both positive and negative).
- Overall, public sentiment toward AI appears relatively stable across the 2023-2024 period, though more granular temporal analysis would be needed to identify correlations with specific industry announcements or controversies.

---

## Key Insights and Implications
- Community activity and engagement are highly event-responsive, with surges aligning with major technology milestones/language model launches.
- User retention and engagement patterns reveal both “sticky” AI forums and more transient, event-driven communities.
- Attention is very unequally distributed: dominant topics and posts monopolize discussion in some spaces, while others sustain more diverse conversation.
- Public sentiment is dynamic and event-sensitive, indicating the importance of tracking mood when analyzing tech/AI discussion spaces.

---

