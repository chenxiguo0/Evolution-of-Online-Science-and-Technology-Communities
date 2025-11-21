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
# RQ3: Attention Concentration (Gini)
# ==============================
@F.udf("double")
def gini_from_list(values):
    xs = sorted([float(v) for v in values if v is not None and float(v) >= 0])
    n = len(xs)
    if n == 0: return 0.0
    s = sum(xs)
    if s == 0: return 0.0
    weighted = sum((i + 1) * x for i, x in enumerate(xs))
    return (2.0 * weighted / (n * s)) - (n + 1.0) / n

RQ3_SUBS = [
    "ChatGPT", "OpenAI", "ArtificialInteligence", "technology",
    "datascience", "AI_Art", "MachineLearning", "GenerativeAI", "TechCulture", "AI_Research"
]

comments_rq3 = preprocess(comments, RQ3_SUBS)
subs_rq3 = preprocess(subs, RQ3_SUBS)

cmts_per_post = (
    comments_rq3.withColumn("post_id", F.regexp_replace("link_id", "^t3_", ""))
    .groupBy("subreddit", "month", "post_id")
    .agg(F.countDistinct("id").alias("comments_on_post"))
)
scores_per_post = (
    subs_rq3.groupBy("subreddit", "month", "id")
    .agg(F.max("score").alias("post_score"))
)

gini_comments = cmts_per_post.groupBy("subreddit", "month").agg(
    gini_from_list(F.collect_list("comments_on_post")).alias("gini_comments"))
gini_scores = scores_per_post.groupBy("subreddit", "month").agg(
    gini_from_list(F.collect_list("post_score")).alias("gini_scores"))

gini_df = (
    gini_comments.join(gini_scores, ["subreddit", "month"], "outer")
    .orderBy("subreddit", "month")
    .toPandas()
)
gini_df.to_csv(f"{OUTDIR}/rq3_gini.csv", index=False)

# --- Visualization ---
# gini_df=pd.read_csv(f"{OUTDIR}/rq3_gini.csv")
plt.figure(figsize=(12, 6))
sns.violinplot(
    data=gini_df,
    x="subreddit",
    y="gini_comments",
    palette="coolwarm",
    inner="box"
)
plt.title("Distribution of Attention Concentration (Violin Plot)", fontsize=14, weight="bold")
plt.ylabel("Gini Index (Comments per Post)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/rq3_gini_violin.png", dpi=300)
plt.close()
print("RQ3 Done: Violin Plot Saved")

# --- Visualization: Sorted Bar Chart of Average Gini ---
avg_gini = (
    gini_df.groupby("subreddit")[["gini_comments", "gini_scores"]]
    .mean()
    .reset_index()
    .sort_values("gini_comments", ascending=False)
)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=avg_gini,
    x="gini_comments",
    y="subreddit",
    palette="viridis"
)
plt.title("Average Discussion Concentration by Subreddit", fontsize=14, weight="bold")
plt.xlabel("Average Gini Index (Comments per Post)")
plt.ylabel("")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/rq3_gini_bar_sorted.png", dpi=300)
plt.close()
print("RQ3 Done: Sorted Bar Chart Saved")

