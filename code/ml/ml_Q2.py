#!/usr/bin/env python3
"""
Clustering Reddit comments by text content (unsupervised).
Outputs: cluster labels (single CSV file), elbow/silhouette plot (PNG), saved pipeline model.
"""

import sys
import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("comment_clustering")

RANDOM_SEED = 42

def save_single_csv(df, output_path):
    """Save Spark DataFrame to a single CSV file"""
    tmp_path = output_path + "_tmp"
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(tmp_path)
    # Find the generated part file
    part_file = next(f for f in os.listdir(tmp_path) if f.startswith("part-") and f.endswith(".csv"))
    # Move and rename
    shutil.move(os.path.join(tmp_path, part_file), output_path)
    shutil.rmtree(tmp_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("net_id")
    parser.add_argument("--master_url", default=None)
    parser.add_argument("--sample", type=float, default=1.0, help="Fraction of data to use for testing")
    args = parser.parse_args()

    net_id = args.net_id
    master_url = args.master_url
    sample_fraction = args.sample

    # Init Spark
    spark = (
        SparkSession.builder
        .appName("RedditCommentClustering")
        .master(master_url)
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.hadoop.fs.s3a.request.payer", "requester")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
    )

    s3_input = f"s3a://{net_id}-dsan6000-datasets/project/reddit/parquet/comments/"
    df = spark.read.parquet(s3_input)

    if sample_fraction < 1.0:
        df = df.sample(fraction=sample_fraction, seed=RANDOM_SEED)

    df = df.filter(col("body").isNotNull())
    df = df.withColumn("comment_length", length(col("body")))

    # Text features
    tokenizer = Tokenizer(inputCol="body", outputCol="words")
    stop = StopWordsRemover(inputCol="words", outputCol="clean_words")
    hashing = HashingTF(inputCol="clean_words", outputCol="raw_features", numFeatures=2**16)
    idf = IDF(inputCol="raw_features", outputCol="text_features")
    assembler = VectorAssembler(inputCols=["text_features", "comment_length"], outputCol="features")

    # Pipeline
    pipeline = Pipeline(stages=[tokenizer, stop, hashing, idf, assembler])
    model_pipeline = pipeline.fit(df)
    df_features = model_pipeline.transform(df)

    # 保存 pipeline 模型
    os.makedirs("./models", exist_ok=True)
    model_pipeline.write().overwrite().save("./models/comment_pipeline")

    # 聚类
    evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette")
    ks = [2, 3, 4, 5, 6]
    silhouettes = []
    os.makedirs("./output", exist_ok=True)

    for k in ks:
        kmeans = KMeans(featuresCol="features", k=k, seed=RANDOM_SEED)
        km_model = kmeans.fit(df_features)
        pred = km_model.transform(df_features)
        sil = evaluator.evaluate(pred)
        silhouettes.append(sil)

        # 保存聚类标签到单个 CSV 文件
        cluster_output_path = f"./output/cluster_labels_k{k}.csv"
        save_single_csv(pred.select("prediction"), cluster_output_path)

    # 绘制 elbow/silhouette 图
    plt.figure()
    plt.plot(ks, silhouettes, marker='o')
    plt.xlabel("k"); plt.ylabel("Silhouette"); plt.title("Elbow/Silhouette Plot")
    plt.savefig("./output/elbow_silhouette.png", dpi=200)
    plt.close()

    spark.stop()

if __name__ == "__main__":
    main()

