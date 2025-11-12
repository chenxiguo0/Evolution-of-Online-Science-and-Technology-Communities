#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reddit User–Subreddit Interaction Network (VADER Sentiment, Top 50 Users)
Performs sentiment analysis using VADER (no Spark NLP)
and visualizes top 50 active users’ engagement networks.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import pandas_udf, PandasUDFType

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
    .appName("Reddit-VADER-Network-Top50")
    .config("spark.hadoop.fs.s3a.request.payer", "requester")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .getOrCreate()
)

print("Spark session ready.")
to_ts = F.to_timestamp

# ==============================
# LOAD REDDIT DATA
# ==============================
comments = spark.read.option("request-payer", "requester").parquet(COMMENTS_PATH)

RQ_SUBS = [
    "MachineLearning", "ArtificialInteligence", "ChatGPT", "OpenAI",
    "technology", "Futurology", "AIethics", "AIforGood"
]

comments_f = (
    comments
    .withColumn("created_ts", to_ts("created_utc"))
    .filter((F.col("created_ts") >= F.lit(DATE_START)) & (F.col("created_ts") <= F.lit(DATE_END)))
    .filter(F.col("subreddit").isin(RQ_SUBS))
    .select("author", "subreddit", "body")
    .na.drop(subset=["author", "body"])
)

total_count = comments_f.count()
print(f"Filtered comments count: {total_count:,}")

sample_fraction = 0.01 if total_count > 1_000_000 else 0.05
comments_sample = comments_f.sample(False, sample_fraction, seed=42)
print(f"Using sample fraction = {sample_fraction:.2%} → {comments_sample.count():,} rows")

# ==============================
# SENTIMENT ANALYSIS (VADER)
# ==============================
print("Performing VADER sentiment analysis...")

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon', quiet=True)

@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def vader_sentiment(texts):
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    results = []
    for t in texts:
        if not t or not isinstance(t, str):
            results.append(0.0)
        else:
            compound = sia.polarity_scores(t)["compound"]
            if compound >= 0.05:
                results.append(1.0)
            elif compound <= -0.05:
                results.append(-1.0)
            else:
                results.append(0.0)
    return pd.Series(results)

sent_df = comments_sample.withColumn("sentiment_score", vader_sentiment(F.col("body")))
print("Sentiment analysis completed.")

# ==============================
# EXPORT SENTIMENT CSV
# ==============================
sent_pd = sent_df.select("author", "subreddit", "sentiment_score").toPandas()
sent_csv_path = f"{OUTDIR}/reddit_vader_user_subreddit.csv"
sent_pd.to_csv(sent_csv_path, index=False)
print(f"Sentiment results saved → {sent_csv_path}")


# ==============================
# NETWORK GRAPH (Top 500 users)
# ==============================
# Load user–subreddit sentiment data
# sent_pd = pd.read_csv(f"{OUTDIR}/reddit_vader_user_subreddit.csv")

# Filter top active users
top_users = sent_pd["author"].value_counts().nlargest(500).index
filtered = sent_pd[sent_pd["author"].isin(top_users)]

# ==============================
# BUILD NETWORK
# ==============================
G = nx.Graph()

for _, row in filtered.iterrows():
    user, sub, sent = row["author"], row["subreddit"], row["sentiment_score"]
    G.add_node(user, type="user")
    G.add_node(sub, type="subreddit")
    G.add_edge(user, sub, weight=1, sentiment=sent)

# Add degree attribute
degree_dict = dict(G.degree())
nx.set_node_attributes(G, degree_dict, "degree")

# ==============================
# ADD SENTIMENT ATTRIBUTES
# ==============================
# Average sentiment for subreddits
sub_sent = filtered.groupby("subreddit")["sentiment_score"].mean().to_dict()
nx.set_node_attributes(G, sub_sent, "sentiment")

# Average sentiment for users
user_sent = filtered.groupby("author")["sentiment_score"].mean().to_dict()
nx.set_node_attributes(G, user_sent, "sentiment")

# ==============================
# DRAW NETWORK GRAPH
# ==============================
plt.figure(figsize=(13, 11))
pos = nx.spring_layout(G, k=0.2, seed=42)

# Separate users and subreddits
sub_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "subreddit"]
user_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "user"]

# Node colors based on sentiment
user_colors = [G.nodes[n].get("sentiment", 0) for n in user_nodes]
sub_colors = [G.nodes[n].get("sentiment", 0) for n in sub_nodes]

# Node sizes based on degree
user_sizes = [40 + 3 * G.nodes[n]["degree"] for n in user_nodes]
sub_sizes = [120 + 5 * G.nodes[n]["degree"] for n in sub_nodes]

# Draw edges
nx.draw_networkx_edges(G, pos, width=0.4, alpha=0.35, edge_color="gray")

# Draw user nodes (lighter)
user_nodes_plot = nx.draw_networkx_nodes(
    G, pos,
    nodelist=user_nodes,
    node_color=user_colors,
    node_size=user_sizes,
    cmap=plt.cm.RdYlGn,
    alpha=0.6,
    label="Users"
)

# Draw subreddit nodes (larger, outlined)
sub_nodes_plot = nx.draw_networkx_nodes(
    G, pos,
    nodelist=sub_nodes,
    node_color=sub_colors,
    node_size=sub_sizes,
    cmap=plt.cm.RdYlGn,
    alpha=0.95,
    edgecolors="black",
    linewidths=0.8,
    label="Subreddits"
)

# Label only subreddit nodes
nx.draw_networkx_labels(
    G, pos,
    labels={n: n for n in sub_nodes},
    font_size=9,
    font_color="black"
)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn)
sm.set_array([])
plt.colorbar(sm, ax=plt.gca(), label="Sentiment (–1 = Negative, +1 = Positive)")

# Final touches
plt.title(
    "Reddit User–Subreddit Interaction Network",
    fontsize=20, weight="bold", pad=10
)
plt.legend(scatterpoints=1, frameon=False, loc="upper right", fontsize=10, labelspacing=1.5)
plt.axis("off")
plt.tight_layout()

# Save figure
out_path = f"{OUTDIR}/reddit_user_subreddit_network_vader.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Network graph saved → {out_path}")
spark.stop()