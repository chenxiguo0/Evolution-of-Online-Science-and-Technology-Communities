#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession, functions as F

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
    .appName("Reddit-RQ5-User-Overlap")
    .config("spark.hadoop.fs.s3a.request.payer", "requester")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .getOrCreate()
)

print("Spark initialized. Loading data...")

# ==============================
# LOAD DATA
# ==============================
comments = spark.read.option("request-payer", "requester").parquet(COMMENTS_PATH)
subs = spark.read.option("request-payer", "requester").parquet(SUBMISSIONS_PATH)

to_ts = F.to_timestamp
print("Reddit data loaded successfully.")

# ==============================
# COMMON PREPROCESS FUNCTION
# ==============================
def preprocess(df, target_subs):
    return (
        df.withColumn("created_ts", to_ts("created_utc"))
        .withColumn("month", F.date_trunc("month", "created_ts").cast("date"))
        .filter(
            (F.col("created_ts") >= DATE_START) &
            (F.col("created_ts") <= DATE_END)
        )
        .filter(F.col("subreddit").isin(target_subs))
    )

# ==============================
# Cross-Subreddit User Overlap
# ==============================
print("\n=== Starting RQ5: Cross-Subreddit User Overlap ===")

RQ5_SUBS = [
    "technology", "technews", "science",
    "Futurology", "ArtificialInteligence",
    "MachineLearning", "OpenAI", "ChatGPT",
    "robotics"
]

comments_rq5 = preprocess(comments, RQ5_SUBS)
subs_rq5 = preprocess(subs, RQ5_SUBS)

# Get unique authors per subreddit
user_sets = (
    comments_rq5.select("subreddit", "author")
    .union(subs_rq5.select("subreddit", "author"))
    .distinct()
    .groupBy("subreddit")
    .agg(F.collect_set("author").alias("users"))
)

user_sets_pdf = user_sets.toPandas()

sub_list = list(user_sets_pdf["subreddit"])
overlap_matrix = pd.DataFrame(index=sub_list, columns=sub_list, dtype=float)

# Calculate Jaccard Similarity
for i, row_i in user_sets_pdf.iterrows():
    sub_i = row_i["subreddit"]
    users_i = set(row_i["users"])
    
    for j, row_j in user_sets_pdf.iterrows():
        sub_j = row_j["subreddit"]
        users_j = set(row_j["users"])
        
        inter = len(users_i & users_j)
        uni = len(users_i | users_j)
        jaccard = inter / uni if uni > 0 else 0
        
        overlap_matrix.loc[sub_i, sub_j] = jaccard

# Save CSV
overlap_matrix.to_csv(f"{OUTDIR}/rq5_user_overlap_matrix.csv")
print("Saved matrix → rq5_user_overlap_matrix.csv")

# ==============================
# VISUALIZATION — Heatmap
# ==============================
plt.figure(figsize=(10, 8))

sns.heatmap(
    overlap_matrix.astype(float),
    cmap="Blues",
    linewidths=0.5,
    square=True,
    cbar_kws={"label": "Jaccard Similarity"}
)

plt.title("Cross-Subreddit User Overlap (Jaccard Similarity)",
          fontsize=14, weight="bold")

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/rq5_user_overlap_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved RQ5 Heatmap → rq5_user_overlap_heatmap.png")
print("=== RQ5 Completed ===")
