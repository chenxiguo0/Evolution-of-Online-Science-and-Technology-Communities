import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession, functions as F, types as T, Window

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
    .appName("Reddit-MultiRQ-EDA")
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

# Common filtering utility
def preprocess(df, target_subs):
    return (
        df.withColumn("created_ts", to_ts("created_utc"))
        .withColumn("month", F.date_trunc("month", "created_ts").cast("date"))
        .filter((F.col("created_ts") >= F.lit(DATE_START)) & (F.col("created_ts") <= F.lit(DATE_END)))
        .filter(F.col("subreddit").isin(target_subs))
    )

# ==============================
# RQ2: Engagement & Retention
# ==============================
RQ2_SUBS = [
    "MachineLearning", "AIethics", "GenerativeAI", "ReinforcementLearning", "AI_Research",
    "AI_Programming", "AI_Agents", "deeplearning", "ChatGPT", "OpenAI"
]

comments_rq2 = preprocess(comments, RQ2_SUBS)
subs_rq2 = preprocess(subs, RQ2_SUBS)

authors = (
    comments_rq2.select("subreddit", "month", "author").distinct()
    .unionByName(subs_rq2.select("subreddit", "month", "author").distinct())
)

user_sets = authors.groupBy("subreddit", "month").agg(F.collect_set("author").alias("authors_set"))
w = Window.partitionBy("subreddit").orderBy("month")
with_prev = user_sets.withColumn("prev_authors_set", F.lag("authors_set").over(w))

engagement = (
    with_prev
    .withColumn("active_users", F.size("authors_set"))
    .withColumn("returning_users",
                F.when(F.col("prev_authors_set").isNull(), F.lit(0))
                .otherwise(F.size(F.array_intersect("authors_set", "prev_authors_set"))))
    .withColumn("engagement_ratio",
                F.when(F.col("active_users") > 0, F.col("returning_users") / F.col("active_users"))
                .otherwise(F.lit(0.0)))
)
eng_df = engagement.toPandas()
eng_df.to_csv(f"{OUTDIR}/rq2_engagement.csv", index=False)

# --- Visualization ---
pivot = eng_df.pivot(index="month", columns="subreddit", values="engagement_ratio")
plt.figure(figsize=(12, 6))
sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.3)
plt.title("User Engagement Ratio", fontsize=14, weight="bold")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/rq2_engagement_heatmap.png", dpi=300)
plt.close()
print("Engagement Heatmap Saved")