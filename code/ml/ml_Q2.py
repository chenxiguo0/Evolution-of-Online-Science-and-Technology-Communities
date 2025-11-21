#!/usr/bin/env python3
"""
Task: K-Means Clustering on Reddit Submissions with Elbow Method, PCA Visualization, Cluster Analysis (including sentiment), and Model Saving

Dataset: Reddit submissions parquet
S3 Location: s3a://{NET_ID}-dsan6000-datasets/project/reddit/parquet/submissions/
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob  # for sentiment analysis

from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, PCA
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, concat_ws, regexp_replace, udf
from pyspark.sql.types import FloatType

# ---------------- Constants ----------------
DEFAULT_SAMPLE_SIZE: int = 0
DEFAULT_NUM_FEATURES: int = 2000 
DEFAULT_K_VALUE: int = 5
MIN_K: int = 2
MAX_K: int = 10

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------- Private functions ----------------

def _get_master_url(cli_value: Optional[str] = None) -> str:
    if cli_value:
        return cli_value
    env_value = os.getenv("MASTER_PRIVATE_IP")
    if env_value:
        return f"spark://{env_value}:7077"
    raise ValueError("Master URL must be provided via --master-url or MASTER_PRIVATE_IP env var")

def _create_spark_session(master_url: str) -> SparkSession:
    logger.info("Creating Spark session")
    spark = (
        SparkSession.builder
        .appName("RedditSubmissions_KMeans")
        .master(master_url)
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )
    return spark

def _get_s3_path() -> str:
    net_id = os.getenv("NET_ID")
    if not net_id:
        raise ValueError("NET_ID not set")
    s3_path = f"s3a://{net_id}-dsan6000-datasets/project/reddit/parquet/submissions/"
    logger.info(f"Using S3 path: {s3_path}")
    return s3_path

def _load_data(spark: SparkSession, s3_path: str, sample_size: int) -> DataFrame:
    logger.info(f"Loading data from {s3_path}")
    df = spark.read.parquet(s3_path)
    total_count = df.count()
    logger.info(f"Loaded {total_count:,} rows")
    if sample_size and 0 < sample_size < total_count:
        df = df.sample(False, fraction=sample_size/total_count, seed=42).limit(sample_size)
        logger.info(f"Sampled {df.count():,} rows")
    return df

def _preprocess_data(df: DataFrame) -> DataFrame:
    logger.info("Preprocessing data: combine title + selftext, remove empty/null, remove non-alpha")
    df = df.withColumn("text", concat_ws(" ", col("title"), col("selftext")))
    df_clean = df.filter(col("text").isNotNull()).filter(col("text") != "")
    df_clean = df_clean.withColumn("text", regexp_replace(col("text"), "[^A-Za-z\\s]", ""))
    logger.info(f"Remaining rows after clean: {df_clean.count():,}")
    return df_clean

def _build_feature_pipeline(num_features: int) -> Pipeline:
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=num_features)
    idf = IDF(inputCol="raw_features", outputCol="features")
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])
    return pipeline

def _extract_features(df: DataFrame, num_features: int) -> DataFrame:
    logger.info("Extracting TF-IDF features")
    pipeline = _build_feature_pipeline(num_features)
    model = pipeline.fit(df)
    df_features = model.transform(df)
    return df_features

def _compute_elbow_method(df: DataFrame, min_k: int, max_k: int) -> Tuple[List[int], List[float]]:
    logger.info(f"Computing elbow method for K={min_k}..{max_k}")
    k_values = list(range(min_k, max_k + 1))
    wcss_values = []
    for k in k_values:
        logger.info(f"Training KMeans with K={k}")
        kmeans = KMeans(k=k, seed=42, featuresCol="features", predictionCol="cluster")
        model = kmeans.fit(df)
        wcss_values.append(model.summary.trainingCost)
        logger.info(f"K={k}, WCSS={model.summary.trainingCost:.2f}")
    return k_values, wcss_values

def _plot_elbow_method(k_values: List[int], wcss_values: List[float], output_path: str):
    plt.figure(figsize=(10,6))
    plt.plot(k_values, wcss_values, 'bo-', linewidth=2, markersize=8)
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.xticks(k_values)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Elbow plot saved to {output_path}")

def _train_kmeans(df: DataFrame, k: int) -> Tuple[KMeans, DataFrame]:
    logger.info(f"Training KMeans with K={k}")
    kmeans = KMeans(k=k, seed=42, featuresCol="features", predictionCol="cluster")
    model = kmeans.fit(df)
    predictions = model.transform(df)
    return model, predictions

def _apply_pca(df: DataFrame) -> DataFrame:
    logger.info("Applying PCA to reduce features to 2D")
    pca = PCA(k=2, inputCol="features", outputCol="pca_features")
    model = pca.fit(df)
    df_pca = model.transform(df)
    return df_pca

def _plot_clusters(predictions_pd, k: int, output_path: str):
    coords = np.array([x for x in predictions_pd['pca_features']])
    x = coords[:,0]
    y = coords[:,1]
    clusters = predictions_pd['cluster']
    plt.figure(figsize=(12,8))
    scatter = plt.scatter(x, y, c=clusters, cmap='viridis', alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    plt.colorbar(scatter, label='Cluster ID')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'KMeans Clustering Visualization (K={k})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Cluster plot saved to {output_path}")

def _analyze_clusters(predictions: DataFrame, k: int) -> Dict[int, Dict]:
    logger.info("Analyzing clusters with top terms")
    cluster_analysis = {}
    for cluster_id in range(k):
        cluster_df = predictions.filter(col("cluster")==cluster_id)
        cluster_size = cluster_df.count()
        
        # Sample top words
        sample_words = (cluster_df.select("filtered_words")
                        .limit(1000)
                        .rdd.flatMap(lambda r: r['filtered_words'])
                        .map(lambda w: (w,1))
                        .reduceByKey(lambda a,b:a+b)
                        .takeOrdered(10, key=lambda x:-x[1]))
        top_terms = [w for w,_ in sample_words]
        cluster_analysis[cluster_id] = {
            "size": cluster_size,
            "top_terms": top_terms
        }
        logger.info(f"Cluster {cluster_id}: size={cluster_size}, top_terms={', '.join(top_terms[:5])}")
    return cluster_analysis


def _save_cluster_analysis(cluster_analysis: Dict[int, Dict], output_path: str):
    with open(output_path, 'w') as f:
        json.dump(cluster_analysis, f, indent=2)
    logger.info(f"Cluster analysis saved to {output_path}")

# ---------------- Main ----------------
def main() -> int:
    parser = argparse.ArgumentParser(description="KMeans Clustering on Reddit Submissions")
    parser.add_argument("--master-url", type=str, help="Spark master URL")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--num-features", type=int, default=DEFAULT_NUM_FEATURES)
    parser.add_argument("--k", type=int, default=DEFAULT_K_VALUE)
    parser.add_argument("--min-k", type=int, default=MIN_K)
    parser.add_argument("--max-k", type=int, default=MAX_K)
    parser.add_argument("--elbow-only", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    os.makedirs(args.output_dir, exist_ok=True)
    spark = _create_spark_session(_get_master_url(args.master_url))
    
    df = _load_data(spark, _get_s3_path(), args.sample_size)
    df = _preprocess_data(df)
    df_features = _extract_features(df, args.num_features)

    if args.elbow_only:
        k_values, wcss_values = _compute_elbow_method(df_features, args.min_k, args.max_k)
        _plot_elbow_method(k_values, wcss_values, os.path.join(args.output_dir, "elbow_method.png"))
    else:
        # Train KMeans
        model, predictions = _train_kmeans(df_features, args.k)
        # Save model
        models_dir = os.path.join(args.output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"kmeans_k{args.k}")
        model.save(model_path)
        logger.info(f"KMeans model saved to {model_path}")

        # PCA and visualization
        predictions_pca = _apply_pca(predictions)
        sample_for_plot = min(10000, predictions_pca.count())
        predictions_pd = predictions_pca.select("pca_features","cluster").limit(sample_for_plot).toPandas()
        _plot_clusters(predictions_pd, args.k, os.path.join(args.output_dir, "cluster_visualization.png"))

        # Cluster analysis with sentiment
        analysis = _analyze_clusters(predictions, args.k)
        _save_cluster_analysis(analysis, os.path.join(args.output_dir, "cluster_analysis.json"))

    spark.stop()
    logger.info("K-Means clustering finished")
    return 0

if __name__ == "__main__":
    sys.exit(main())
