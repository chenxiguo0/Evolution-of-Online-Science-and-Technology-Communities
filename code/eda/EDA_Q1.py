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
    .appName("Reddit-RQ1-Activity-Evolution")
    .config("spark.hadoop.fs.s3a.request.payer", "requester")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .getOrCreate()
)

# ==============================
# LOAD DATA
# ==============================
comments = spark.read.option("request-payer", "requester").parquet(COMMENTS_PATH)
subs = spark.read.option("request-payer", "requester").parquet(SUBMISSIONS_PATH)
to_ts = F.to_timestamp

print("Reddit data loaded from S3")

# ==============================
# COMMON FILTER FUNCTION
# ==============================
def preprocess(df, target_subs):
    return (
        df.withColumn("created_ts", to_ts("created_utc"))
        .withColumn("month", F.date_trunc("month", "created_ts").cast("date"))
        .filter((F.col("created_ts") >= F.lit(DATE_START)) & (F.col("created_ts") <= F.lit(DATE_END)))
        .filter(F.col("subreddit").isin(target_subs))
    )

# ==============================
# RQ1: Activity Evolution
# ==============================
RQ1_SUBS = [
    "MachineLearning", "ArtificialInteligence", "OpenAI", "ChatGPT",
    "technology", "Futurology", "technews", "TechNewsToday", "AIforGood"
]

comments_rq1 = preprocess(comments, RQ1_SUBS)
subs_rq1 = preprocess(subs, RQ1_SUBS)

# Aggregate monthly posts and comments
posts_monthly = subs_rq1.groupBy("subreddit", "month").agg(F.countDistinct("id").alias("posts"))
comments_monthly = comments_rq1.groupBy("subreddit", "month").agg(F.countDistinct("id").alias("comments"))

activity_monthly = (
    posts_monthly.join(comments_monthly, ["subreddit", "month"], "outer")
    .na.fill({"posts": 0, "comments": 0})
    .withColumn("total_activity", F.col("posts") + F.col("comments"))
)

# Convert to Pandas and save
activity_df = activity_monthly.toPandas()
activity_df.to_csv(f"{OUTDIR}/rq1_activity.csv", index=False)
print("Activity data saved to rq1_activity.csv")

# ==============================
# DATA CLEANING (Pandas)
# ==============================
activity_df = activity_df.dropna(subset=["month", "total_activity", "subreddit"])
activity_df["month"] = pd.to_datetime(activity_df["month"], errors="coerce")
activity_df["total_activity"] = pd.to_numeric(activity_df["total_activity"], errors="coerce").fillna(0)
activity_df["subreddit"] = activity_df["subreddit"].astype(str)

print("Cleaned DataFrame summary:")
print(activity_df.dtypes)
print(activity_df.head(5))

# ==============================
# VISUALIZATION
# ==============================
# activity_df = pd.read_csv(f"{OUTDIR}/rq1_activity.csv")
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=activity_df,
    x="month",
    y="total_activity",
    hue="subreddit",
    marker="o",
    linewidth=2
)
plt.title("Monthly Activity Trends across AI & Tech Subreddits",
          fontsize=14, weight="bold")
plt.xlabel("Month")
plt.ylabel("Total Posts + Comments")
plt.xticks(rotation=45)

# Move legend outside
plt.legend(
    title="Subreddit",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0,
    frameon=False
)

# Adjust layout
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(f"{OUTDIR}/rq1_activity_trends.png", dpi=300, bbox_inches="tight")
plt.close()

print("Activity Trends Plot Saved → rq1_activity_trends.png")
print(f"All outputs saved under: {OUTDIR}")

