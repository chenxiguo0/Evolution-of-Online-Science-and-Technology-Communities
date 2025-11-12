#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate Sentiment Trend across 4 Subreddit Categories (with VADER)
"""
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, functions as F, types as T
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# ==============================
# CONFIGURATION
# ==============================
NET_ID = "lh1085"
COMMENTS_PATH = f"s3a://{NET_ID}-dsan6000-datasets-01/project/reddit/parquet/comments/"
OUTDIR = "/home/ubuntu/dsan6000/eda_outputs"
DATE_START, DATE_END = "2023-06-01", "2024-07-31"
os.makedirs(OUTDIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="Set2")

# ==============================
# INIT SPARK
# ==============================
spark = (
    SparkSession.builder
    .appName("Reddit-Category-Sentiment-Trend-VADER")
    .config("spark.hadoop.fs.s3a.request.payer", "requester")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .getOrCreate()
)

to_ts = F.to_timestamp

# ==============================
# LOAD NLTK VADER
# ==============================
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

# register UDF
def vader_sentiment(text):
    if text is None:
        return 0.0
    try:
        return float(analyzer.polarity_scores(text)['compound'])
    except Exception:
        return 0.0

vader_udf = F.udf(vader_sentiment, T.FloatType())

# ==============================
# FUNCTION: Load + Preprocess
# ==============================
def preprocess(df, subs):
    return (
        df.withColumn("created_ts", to_ts("created_utc"))
        .withColumn("month", F.date_trunc("month", "created_ts").cast("date"))
        .filter((F.col("created_ts") >= F.lit(DATE_START)) & (F.col("created_ts") <= F.lit(DATE_END)))
        .filter(F.col("subreddit").isin(subs))
        .filter(F.col("body").isNotNull())
    )

comments = spark.read.option("request-payer", "requester").parquet(COMMENTS_PATH)
print("Reddit data loaded.")

# ==============================
# DEFINE CATEGORY GROUPS
# ==============================
CATEGORIES = {
    "AI/ML": [
        "ChatGPT", "OpenAI", "ArtificialInteligence", "MachineLearning", "GenerativeAI", "AIethics"
    ],
    "Programming/Data": [
        "datascience", "bigdata", "programming", "Python", "learnprogramming", "CloudComputing"
    ],
    "Science/STEM": [
        "science", "Physics", "Engineering", "Astronomy", "Neuroscience", "MaterialsScience"
    ],
    "Tech/Future Trends": [
        "technology", "Futurology", "TechCulture", "Innovation", "FutureTechnology"
    ]
}

# ==============================
# LOOP OVER CATEGORIES
# ==============================
all_dfs = []

for cat_name, subs_list in CATEGORIES.items():
    print(f"Processing category: {cat_name}")
    df_filtered = preprocess(comments, subs_list)
    df_sample = df_filtered.sample(withReplacement=False, fraction=0.05, seed=42)

    # calculate sentiment
    df_with_sent = df_sample.withColumn("sentiment_score", vader_udf(F.col("body")))

    df_monthly = (
        df_with_sent.groupBy("month")
        .agg(F.avg("sentiment_score").alias("avg_sentiment"))
        .withColumn("category", F.lit(cat_name))
        .orderBy("month")
    )
    all_dfs.append(df_monthly)

# merge all categories
combined_df = all_dfs[0]
for df in all_dfs[1:]:
    combined_df = combined_df.unionByName(df)

sent_trend = combined_df.toPandas()
sent_trend.to_csv(f"{OUTDIR}/NLP_Q3_sentiment_trend_all_categories_VADER.csv", index=False)

# ==============================
# EVENT TIMELINE + VISUALIZATION
# ==============================
EVENTS = {
    "2023-07-12": "Claude 2 Launch",
    "2023-11-06": "OpenAI Developer Day",
    "2024-02-15": "Gemini Launch",
    "2024-04-17": "EU AI Act",
    "2024-06-20": "GPT-5 Rumors"
}

plt.figure(figsize=(12, 6))
sent_trend["month"] = pd.to_datetime(sent_trend["month"])

ax = sns.lineplot(
    data=sent_trend,
    x="month",
    y="avg_sentiment",
    hue="category",
    marker="o",
    linewidth=2.2
)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

# dynamic event labels
y_max, y_min = sent_trend["avg_sentiment"].max(), sent_trend["avg_sentiment"].min()
y_range = y_max - y_min
base_y = y_min + 0.9 * y_range
alt_offset = 0.15 * y_range

for i, (date, label) in enumerate(EVENTS.items()):
    x_pos = pd.to_datetime(date)
    plt.axvline(x=x_pos, color="gray", linestyle="--", alpha=0.5, linewidth=1.2)
    y_text = base_y - alt_offset
    plt.text(
        x_pos, y_text, label,
        rotation=90, fontsize=9, color="dimgray",
        va="center", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.6)
    )

plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.title("Aggregate Sentiment Dynamics across Reddit Tech Communities (VADER)",
          fontsize=15, weight="bold")
plt.xlabel("Month")
plt.ylabel("Average Sentiment (VADER compound score)")
plt.xticks(rotation=45)
plt.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(f"{OUTDIR}/NLP_Q3_sentiment_trend_all_categories_VADER.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: sentiment_trend_all_categories_VADER.png and CSV.")


