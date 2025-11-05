"""
Reddit Filtered Dataset Summary Generator
-----------------------------------------
Reads filtered Reddit comments & submissions from S3
and generates summary CSVs for documentation.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, min, max, countDistinct
)

import os

def main():
    net_id = "xm149"  # <<--- your netid
    base_path = f"s3a://{net_id}-dsan6000-datasets/project/reddit/parquet"
    output_path = "data/csv" 

    spark = (
        SparkSession.builder
        .appName("Reddit Dataset Summary")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.InstanceProfileCredentialsProvider")
        .getOrCreate()
    )


    comments = spark.read.parquet(f"{base_path}/comments/")
    submissions = spark.read.parquet(f"{base_path}/submissions/")

    # --- dataset_summary.csv ---
    summary_rows = []
    for name, df in [("comments", comments), ("submissions", submissions)]:
        total_rows = df.count()
        date_min = df.selectExpr("min(date)").collect()[0][0]
        date_max = df.selectExpr("max(date)").collect()[0][0]

        size_gb = round(total_rows * 0.000001, 2)  

        summary_rows.append((name, total_rows, size_gb, str(date_min), str(date_max)))

    summary_df = spark.createDataFrame(summary_rows, ["data_type", "total_rows", "size_gb", "date_range_start", "date_range_end"])
    summary_df.coalesce(1).write.mode("overwrite").csv(f"{output_path}/dataset_summary.csv", header=True)

    # --- subreddit_statistics.csv ---
    comments_sub = comments.groupBy("subreddit").agg(
        count("*").alias("num_comments"),
        avg("score").alias("avg_comment_score")
    )

    submissions_sub = submissions.groupBy("subreddit").agg(
        count("*").alias("num_submissions"),
        avg("score").alias("avg_submission_score")
    )

    subreddit_stats = (
        comments_sub.join(submissions_sub, "subreddit", "outer")
        .fillna(0)
        .withColumn("total_rows", col("num_comments") + col("num_submissions"))
    )

    subreddit_stats.coalesce(1).write.mode("overwrite").csv(f"{output_path}/subreddit_statistics.csv", header=True)

    # --- temporal_distribution.csv ---
    comments_time = comments.groupBy("year_month").agg(count("*").alias("num_comments"))
    submissions_time = submissions.groupBy("year_month").agg(count("*").alias("num_submissions"))

    temporal = (
        comments_time.join(submissions_time, "year_month", "outer")
        .fillna(0)
        .withColumn("total_rows", col("num_comments") + col("num_submissions"))
    )

    temporal.coalesce(1).write.mode("overwrite").csv(f"{output_path}/temporal_distribution.csv", header=True)

    print("\n✅ CSV files created successfully:")
    print(f" - {output_path}/dataset_summary.csv")
    print(f" - {output_path}/subreddit_statistics.csv")
    print(f" - {output_path}/temporal_distribution.csv")

    spark.stop()


if __name__ == "__main__":
    main()

import os
import shutil
from glob import glob

def rename_part_file(folder_path, final_path):
    """
    Move the first part-*.csv in folder_path to final_path (can be full path),
    then delete the original folder.
    """
    part_file = glob(os.path.join(folder_path, "part-*.csv"))[0]

    os.makedirs(os.path.dirname(final_path), exist_ok=True)

    shutil.move(part_file, final_path)

    shutil.rmtree(folder_path)


rename_part_file("data/csv/dataset_summary.csv", "data/csv/dataset_summary_final.csv")
rename_part_file("data/csv/subreddit_statistics.csv", "data/csv/subreddit_statistics_final.csv")
rename_part_file("data/csv/temporal_distribution.csv", "data/csv/temporal_distribution_final.csv")

