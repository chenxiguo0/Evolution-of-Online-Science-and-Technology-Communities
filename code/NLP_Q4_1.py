#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reddit NLP Analysis – Clean Word Cloud Visualization by Topic
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from pyspark.sql import SparkSession, functions as F
from nltk.corpus import stopwords as nltk_stop
from nltk.stem import WordNetLemmatizer
import nltk

# ==============================
# CONFIGURATION
# ==============================
NET_ID = "lh1085"
COMMENTS_PATH = f"s3a://{NET_ID}-dsan6000-datasets-01/project/reddit/parquet/comments/"
SUBMISSIONS_PATH = f"s3a://{NET_ID}-dsan6000-datasets-01/project/reddit/parquet/submissions/"
OUTDIR = "/home/ubuntu/dsan6000/eda_outputs"
DATE_START, DATE_END = "2023-06-01", "2024-07-31"

os.makedirs(OUTDIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="Set2")

# ==============================
# INIT SPARK
# ==============================
spark = (
    SparkSession.builder
    .appName("Reddit-NLP-RQ2-WordCloud-Clean")
    .config("spark.hadoop.fs.s3a.request.payer", "requester")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .getOrCreate()
)

to_ts = F.to_timestamp

# ==============================
# NLTK INITIALIZATION
# ==============================
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
lemmatizer = WordNetLemmatizer()

# ==============================
# LOAD DATA
# ==============================
def preprocess(df, subs):
    return (
        df.withColumn("created_ts", to_ts("created_utc"))
        .withColumn("month", F.date_trunc("month", "created_ts").cast("date"))
        .filter((F.col("created_ts") >= F.lit(DATE_START)) & (F.col("created_ts") <= F.lit(DATE_END)))
        .filter(F.col("subreddit").isin(subs))
    )

comments = spark.read.option("request-payer", "requester").parquet(COMMENTS_PATH)
subs = spark.read.option("request-payer", "requester").parquet(SUBMISSIONS_PATH)
print("Reddit Data Loaded")

RQ2_SUBS = [
    "MachineLearning", "ArtificialInteligence", "AIethics", "Futurology",
    "technology", "AI_Art", "AIforGood", "TechCulture"
]

subs_rq2 = preprocess(subs, RQ2_SUBS)
comments_rq2 = preprocess(comments, RQ2_SUBS)

# Merge content
comments_rq2 = comments_rq2.withColumnRenamed("body", "content")
subs_rq2 = subs_rq2.withColumn(
    "content",
    F.concat_ws(" ", F.coalesce(F.col("title"), F.lit("")), F.coalesce(F.col("selftext"), F.lit("")))
)
merged = comments_rq2.select("subreddit", "content").unionByName(subs_rq2.select("subreddit", "content"))

merged_pd = merged.sample(withReplacement=False, fraction=0.05, seed=42).toPandas()
merged_pd["content"] = merged_pd["content"].fillna("").astype(str)

# ==============================
# TEXT CLEANING FUNCTION
# ==============================
custom_stop = set(STOPWORDS) | set(nltk_stop.words("english")) | {
    "ai", "artificial", "intelligence", "machine", "learning", "use", "people",
    "think", "make", "good", "get", "really", "would", "one", "also", "even",
    "go", "could", "want", "said", "like", "thing", "see", "much", "way", "well"
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)       # remove punctuation/numbers
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in custom_stop and len(w) > 2]
    return " ".join(tokens)

merged_pd["clean_text"] = merged_pd["content"].apply(clean_text)

# ==============================
# TOPIC CLASSIFICATION
# ==============================
def classify_topic(text):
    text = text.lower()
    if any(k in text for k in ["model", "neural", "training", "parameter", "architecture", "optimizer"]):
        return "technical"
    elif any(k in text for k in ["ethic", "bias", "regulation", "moral", "responsible", "fairness"]):
        return "ethical"
    elif any(k in text for k in ["society", "impact", "education", "future", "job", "policy"]):
        return "societal"
    else:
        return "other"

merged_pd["topic_type"] = merged_pd["clean_text"].apply(classify_topic)

# ==============================
# WORD CLOUD GENERATION
# ==============================
topic_groups = merged_pd.groupby("topic_type")["clean_text"].apply(lambda x: " ".join(x)).to_dict()

plt.figure(figsize=(14, 10))
cols = 2
rows = (len(topic_groups) + 1) // cols

for i, (topic, text) in enumerate(topic_groups.items(), 1):
    plt.subplot(rows, cols, i)
    if not text.strip():
        plt.text(0.5, 0.5, f"No data for {topic}", ha="center", va="center")
        plt.axis("off")
        continue

    wc = WordCloud(
        width=900,
        height=600,
        background_color="white",
        colormap="tab10" if topic == "technical" else "tab20c",
        stopwords=custom_stop,
        max_words=100,
        min_font_size=8
    ).generate(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"{topic.capitalize()} Discussions", fontsize=15, weight="bold")

plt.suptitle("Semantic Fields of Reddit AI & Tech Discussions", fontsize=18, weight="bold", y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"{OUTDIR}/NLP_Q4_wordclouds_by_topic_clean.png", dpi=300)
plt.close()
print("Saved clean wordcloud → NLP_Q4_wordclouds_by_topic_clean.png")

spark.stop()

