#!/usr/bin/env python3
"""
High-quality comment classification (binary) using Logistic Regression.
Outputs: metrics CSVs, confusion matrix (CSV+PNG), ROC/PR curves (PNG), prediction CSVs, saved model.
"""

import sys
import os
import time
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, when, hour, dayofweek, from_unixtime
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("high_quality_classification")

THRESHOLD = 6
RANDOM_SEED = 42

def plot_confusion_matrix(cm, filename, title="Confusion Matrix"):
    import seaborn as sns
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.savefig(filename, dpi=200)
    plt.close()

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
        .appName("HighQualityCommentClassification")
        .master(master_url)
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.hadoop.fs.s3a.request.payer", "requester")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
    )

    s3_input = f"s3a://{net_id}-dsan6000-datasets/project/reddit/parquet/comments/"
    logger.info("Reading comments from %s", s3_input)
    df = spark.read.parquet(s3_input)

    # --------------------
    # Sample for testing
    # --------------------
    if sample_fraction < 1.0:
        df = df.sample(fraction=sample_fraction, seed=RANDOM_SEED)

    # Basic feature engineering
    df = df.withColumn("label", when(col("score") >= THRESHOLD, 1).otherwise(0))
    df = df.withColumn("comment_length", length(col("body")))
    df = df.withColumn("has_url", col("body").rlike("http").cast("int"))
    df = df.withColumn("created_ts", from_unixtime(col("created_utc")))
    df = df.withColumn("hour_of_day", hour(col("created_ts")))
    df = df.withColumn("day_of_week", dayofweek(col("created_ts")))
    df = df.filter(col("body").isNotNull())

    # Train/val/test split
    train_val, test = df.randomSplit([0.8, 0.2], seed=RANDOM_SEED)
    train, val = train_val.randomSplit([0.75, 0.25], seed=RANDOM_SEED)
    logger.info("Splits: train=%d, val=%d, test=%d", train.count(), val.count(), test.count())

    # --------------------
    # Compute class weights
    # --------------------
    counts = train.groupBy("label").count().toPandas()
    count_dict = dict(zip(counts.label, counts["count"]))
    total = sum(count_dict.values())
    class_weight_dict = {0: total / (2 * count_dict.get(0,1)), 1: total / (2 * count_dict.get(1,1))}
    # add column
    from pyspark.sql.functions import udf
    from pyspark.sql.types import DoubleType
    weight_udf = udf(lambda label: float(class_weight_dict[label]), DoubleType())
    train = train.withColumn("class_weight", weight_udf(col("label")))

    # Text pipeline
    tokenizer = Tokenizer(inputCol="body", outputCol="words")
    stop = StopWordsRemover(inputCol="words", outputCol="clean_words")
    hashing = HashingTF(inputCol="clean_words", outputCol="raw_features", numFeatures=2**16)
    idf = IDF(inputCol="raw_features", outputCol="text_features")

    assembler = VectorAssembler(
        inputCols=["text_features", "comment_length", "has_url", "hour_of_day", "day_of_week"],
        outputCol="features"
    )

    # Logistic Regression with class weight
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20, weightCol="class_weight")
    pipeline_lr = Pipeline(stages=[tokenizer, stop, hashing, idf, assembler, lr])

    evaluator_auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

    # --------------------
    # Train model
    # --------------------
    t0 = time.time()
    logger.info("Training LogisticRegression pipeline")
    model_lr = pipeline_lr.fit(train)
    t1 = time.time()
    logger.info("Training completed in %.1f seconds", t1 - t0)

    # -------------------------------
    # Save model
    # -------------------------------
    os.makedirs("./models", exist_ok=True)
    model_lr.write().overwrite().save("./models/logistic_regression")

    # -------------------------------
    # Metrics, confusion matrix, ROC/PR
    # -------------------------------
    os.makedirs("./output", exist_ok=True)
    pred = model_lr.transform(test)
    from pyspark.sql.functions import udf
    extract_prob_udf = udf(lambda x: float(x[1]), DoubleType())
    pred = pred.withColumn("prob", extract_prob_udf(col("probability")))

    pred_pd = pred.select("label", "prediction", "prob").toPandas()
    y_true = pred_pd["label"].astype(int)
    y_pred = pred_pd["prediction"].astype(int)
    y_prob = pred_pd["prob"].astype(float)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_true, y_prob)

    metrics_df = pd.DataFrame([{
        "model": "LogisticRegression",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc_score
    }])
    metrics_df.to_csv("./output/model_metrics_summary.csv", index=False)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"])
    cm_df.to_csv("./output/confusion_matrix_logistic_regression.csv", index=True)
    plot_confusion_matrix(cm, "./output/confusion_matrix_logistic_regression.png", title="Confusion Matrix LogisticRegression")

    # ROC Curve
    plt.figure(figsize=(6,6))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"LogisticRegression (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1], "--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
    plt.legend()
    plt.savefig("./output/roc_logistic_regression.png", dpi=200)
    plt.close()

    # Precision-Recall Curve
    plt.figure(figsize=(6,6))
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f"LogisticRegression (AUPR={pr_auc:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig("./output/pr_logistic_regression.png", dpi=200)
    plt.close()

    spark.stop()

if __name__ == "__main__":
    main()
