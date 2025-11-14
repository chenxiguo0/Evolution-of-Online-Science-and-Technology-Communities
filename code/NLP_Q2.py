#!/usr/bin/env python3
"""
Sentiment analysis using VADER sentiment
"""

import sys
import logging
import os
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lower, regexp_replace, trim, udf, 
    avg, count, desc, year, month, concat_ws,
    when, lit
)
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s,%(msecs)03d,p%(process)d,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_spark_session(app_name, master_url=None):
    """Create and configure Spark session."""
    builder = SparkSession.builder.appName(app_name)
    
    if master_url:
        builder = builder.master(master_url)
    
    builder = builder \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", 
                "com.amazonaws.auth.InstanceProfileCredentialsProvider") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    
    return builder.getOrCreate()


class VaderSentimentWrapper:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text):
        if not text or text.strip() == "":
            return 0.0
        score = self.analyzer.polarity_scores(text)
        return float(score["compound"])


def get_sentiment_udf():
    """Create UDF for VADER sentiment analysis."""
    analyzer = VaderSentimentWrapper()

    def analyze_sentiment(text):
        if not text:
            return 0.0
        return analyzer.analyze(text)

    return udf(analyze_sentiment, DoubleType())


def categorize_sentiment(score):
    """Categorize sentiment score into positive/neutral/negative."""
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'


def preprocess_text(df, text_col):
    """Basic text preprocessing."""
    # Remove URLs
    df = df.withColumn(
        text_col,
        regexp_replace(col(text_col), r'http\S+|www\.\S+', ' ')
    )
    
    # Remove markdown formatting
    df = df.withColumn(
        text_col,
        regexp_replace(col(text_col), r'[\*_~`#]', ' ')
    )
    
    # Remove extra whitespace
    df = df.withColumn(
        text_col,
        regexp_replace(col(text_col), r'\s+', ' ')
    )
    
    return df


def main(net_id, spark_master=None):
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("NLP Q2: Sentiment Analysis")
    logger.info("=" * 80)
    
    # Setup output directory (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    OUTDIR = os.path.join(script_dir, "..", "data", "plots")
    os.makedirs(OUTDIR, exist_ok=True)
    logger.info(f"Output directory: {OUTDIR}")
    
    # Create Spark session
    spark = create_spark_session("Reddit_Sentiment_Analysis", spark_master)
    logger.info("Spark session created")
    
    try:
        # Load filtered data
        comments_path = f"s3a://{net_id}-dsan6000-datasets/project/reddit/parquet/comments/"
        submissions_path = f"s3a://{net_id}-dsan6000-datasets/project/reddit/parquet/submissions/"
        
        logger.info(f"Loading comments from: {comments_path}")
        comments_df = spark.read.parquet(comments_path)
        
        logger.info(f"Loading submissions from: {submissions_path}")
        submissions_df = spark.read.parquet(submissions_path)
        
        logger.info(f"Comments count: {comments_df.count():,}")
        logger.info(f"Submissions count: {submissions_df.count():,}")
        
        # Filter out deleted/removed content
        comments_df = comments_df.filter(
            (col('body').isNotNull()) & 
            (~col('body').isin('[deleted]', '[removed]')) &
            (col('body') != '')
        )
        
        # Prepare comments
        comments_df = comments_df.select(
            col('body').alias('text'),
            col('subreddit'),
            col('date'),
            col('created_utc'),
            col('score').alias('upvotes')
        ).withColumn('type', lit('comment'))
        
        # Prepare submissions (combine title and selftext)
        submissions_df = submissions_df.filter(
            (col('title').isNotNull()) & 
            (col('title') != '')
        )
        
        submissions_df = submissions_df.select(
            concat_ws(' ', col('title'), col('selftext')).alias('text'),
            col('subreddit'),
            col('date'),
            col('created_utc'),
            col('score').alias('upvotes')
        ).withColumn('type', lit('submission'))
        
        # Union datasets
        all_data = comments_df.union(submissions_df)
        logger.info(f"Total documents for sentiment analysis: {all_data.count():,}")
        
        # Preprocess text
        logger.info("Preprocessing text...")
        all_data = preprocess_text(all_data, 'text')
        
        # Sample for faster processing (optional - remove for full analysis)
        # all_data = all_data.sample(0.2, seed=42)
        
        # Cache for performance
        all_data = all_data.repartition(200).cache()
        
        # Apply sentiment analysis
        logger.info("Applying sentiment analysis (this may take a while)...")
        sentiment_udf = get_sentiment_udf()
        all_data = all_data.withColumn('sentiment_score', sentiment_udf(col('text')))
        
        # Categorize sentiment
        categorize_udf = udf(categorize_sentiment, StringType())
        all_data = all_data.withColumn(
            'sentiment_category', 
            categorize_udf(col('sentiment_score'))
        )
        
        # Add time dimensions
        all_data = all_data.withColumn('year', year(col('date')))
        all_data = all_data.withColumn('month', month(col('date')))
        all_data = all_data.withColumn(
            'year_month',
            concat_ws('-', col('year'), col('month'))
        )
        
        # Cache results
        all_data.cache()
        
        logger.info("\n" + "=" * 80)
        logger.info("OVERALL SENTIMENT DISTRIBUTION")
        logger.info("=" * 80)
        sentiment_dist = all_data.groupBy('sentiment_category') \
            .agg(count('*').alias('count')) \
            .orderBy(desc('count'))
        sentiment_dist.show()
        
        # Sentiment by subreddit
        logger.info("\n" + "=" * 80)
        logger.info("AVERAGE SENTIMENT BY SUBREDDIT")
        logger.info("=" * 80)
        sentiment_by_subreddit = all_data.groupBy('subreddit') \
            .agg(
                avg('sentiment_score').alias('avg_sentiment'),
                count('*').alias('total_posts')
            ) \
            .orderBy(desc('avg_sentiment'))
        
        sentiment_by_subreddit.show(30, truncate=False)
        
        # Sentiment trends over time
        logger.info("\n" + "=" * 80)
        logger.info("SENTIMENT TRENDS OVER TIME")
        logger.info("=" * 80)
        sentiment_trends = all_data.groupBy('year_month') \
            .agg(
                avg('sentiment_score').alias('avg_sentiment'),
                count('*').alias('total_posts')
            ) \
            .orderBy('year_month')
        
        sentiment_trends.show(50, truncate=False)
        
        # Sentiment by subreddit over time
        logger.info("\nCalculating sentiment by subreddit over time...")
        sentiment_subreddit_time = all_data.groupBy('subreddit', 'year_month') \
            .agg(
                avg('sentiment_score').alias('avg_sentiment'),
                count('*').alias('post_count')
            ) \
            .orderBy('subreddit', 'year_month')
        
        # Sentiment distribution by subreddit
        sentiment_dist_by_subreddit = all_data.groupBy('subreddit', 'sentiment_category') \
            .agg(count('*').alias('count')) \
            .orderBy('subreddit', 'sentiment_category')
        
        # Save results
        output_base = f"s3a://{net_id}-dsan6000-datasets/project/nlp_analysis/q2_sentiment/"
        
        logger.info(f"\nSaving results to: {output_base}")
        
        # Save overall sentiment trends
        sentiment_trends.coalesce(1).write.mode('overwrite') \
            .option('header', 'true') \
            .csv(output_base + 'sentiment_trends/')
        
        # Save sentiment by subreddit
        sentiment_by_subreddit.coalesce(1).write.mode('overwrite') \
            .option('header', 'true') \
            .csv(output_base + 'sentiment_by_subreddit/')
        
        # Save sentiment by subreddit over time
        sentiment_subreddit_time.coalesce(1).write.mode('overwrite') \
            .option('header', 'true') \
            .csv(output_base + 'sentiment_subreddit_time/')
        
        # Save sentiment distribution by subreddit
        sentiment_dist_by_subreddit.coalesce(1).write.mode('overwrite') \
            .option('header', 'true') \
            .csv(output_base + 'sentiment_distribution_by_subreddit/')
        
        # Show some interesting insights
        logger.info("\n" + "=" * 80)
        logger.info("MOST POSITIVE SUBREDDITS")
        logger.info("=" * 80)
        sentiment_by_subreddit.orderBy(desc('avg_sentiment')).show(10, truncate=False)
        
        logger.info("\n" + "=" * 80)
        logger.info("MOST NEGATIVE SUBREDDITS")
        logger.info("=" * 80)
        sentiment_by_subreddit.orderBy('avg_sentiment').show(10, truncate=False)
        
        # ==============================
        # VISUALIZATION AND CSV EXPORT
        # ==============================
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 80)
        
        # Convert to pandas for visualization
        sentiment_trends_pd = sentiment_trends.toPandas()
        sentiment_by_subreddit_pd = sentiment_by_subreddit.toPandas()
        sentiment_subreddit_time_pd = sentiment_subreddit_time.toPandas()
        sentiment_dist_pd = sentiment_dist.toPandas()
        sentiment_dist_by_subreddit_pd = sentiment_dist_by_subreddit.toPandas()
        
        # Save CSVs locally
        sentiment_trends_pd.to_csv(f"{OUTDIR}/NLP_Q2_sentiment_trends.csv", index=False)
        sentiment_by_subreddit_pd.to_csv(f"{OUTDIR}/NLP_Q2_sentiment_by_subreddit.csv", index=False)
        sentiment_subreddit_time_pd.to_csv(f"{OUTDIR}/NLP_Q2_sentiment_subreddit_time.csv", index=False)
        sentiment_dist_by_subreddit_pd.to_csv(f"{OUTDIR}/NLP_Q2_sentiment_distribution.csv", index=False)
        logger.info(f"CSVs saved to {OUTDIR}")
        
        # Set seaborn style
        sns.set_theme(style="whitegrid", palette="Set2")
        
        # 1. Overall Sentiment Trends Over Time with Events
        plt.figure(figsize=(14, 7))
        
        sentiment_trends_pd['year_month'] = pd.to_datetime(sentiment_trends_pd['year_month'])
        sentiment_trends_pd = sentiment_trends_pd.sort_values('year_month')
        
        plt.plot(
            sentiment_trends_pd['year_month'],
            sentiment_trends_pd['avg_sentiment'],
            marker='o',
            linewidth=2.5,
            markersize=6,
            color='steelblue',
            label='Average Sentiment'
        )
        
        # Add key AI events (similar to reference code)
        EVENTS = {
            "2023-07-12": "Claude 2 Launch",
            "2023-11-06": "OpenAI DevDay",
            "2024-02-15": "Gemini Launch",
            "2024-04-17": "EU AI Act",
        }
        
        y_max = sentiment_trends_pd['avg_sentiment'].max()
        y_min = sentiment_trends_pd['avg_sentiment'].min()
        y_range = y_max - y_min
        base_y = y_min + 0.85 * y_range
        
        for date_str, label in EVENTS.items():
            event_date = pd.to_datetime(date_str)
            if sentiment_trends_pd['year_month'].min() <= event_date <= sentiment_trends_pd['year_month'].max():
                plt.axvline(x=event_date, color='gray', linestyle='--', alpha=0.5, linewidth=1.2)
                plt.text(
                    event_date, base_y, label,
                    rotation=90, fontsize=9, color='dimgray',
                    va='center', ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='lightgray', alpha=0.6)
                )
        
        plt.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        plt.title("Sentiment Trends Over Time in AI/Tech Communities", fontsize=16, weight='bold', pad=20)
        plt.xlabel("Month", fontsize=12)
        plt.ylabel("Average Sentiment Score", fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/NLP_Q2_sentiment_trends_over_time.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved: NLP_Q2_sentiment_trends_over_time.png")
        
        # 2. Sentiment by Subreddit (Bar Chart)
        plt.figure(figsize=(14, 8))
        
        # Sort by sentiment and get top/bottom subreddits
        top_subreddits = sentiment_by_subreddit_pd.nlargest(15, 'avg_sentiment')
        bottom_subreddits = sentiment_by_subreddit_pd.nsmallest(15, 'avg_sentiment')
        combined = pd.concat([top_subreddits, bottom_subreddits]).drop_duplicates()
        combined = combined.sort_values('avg_sentiment')
        
        colors = ['#d73027' if x < -0.05 else '#fee08b' if x < 0.05 else '#1a9850' 
                  for x in combined['avg_sentiment']]
        
        plt.barh(range(len(combined)), combined['avg_sentiment'], color=colors, alpha=0.8)
        plt.yticks(range(len(combined)), combined['subreddit'], fontsize=10)
        plt.axvline(0, color='black', linestyle='-', linewidth=0.8)
        plt.xlabel("Average Sentiment Score", fontsize=12)
        plt.title("Sentiment Comparison Across Subreddits", fontsize=16, weight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/NLP_Q2_sentiment_by_subreddit_bar.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved: NLP_Q2_sentiment_by_subreddit_bar.png")
        
        # 3. Sentiment Trends by Category (Top Subreddits)
        plt.figure(figsize=(15, 8))
        
        # Select top 6 most active subreddits
        top_6_subreddits = sentiment_by_subreddit_pd.nlargest(6, 'total_posts')['subreddit'].tolist()
        
        for subreddit in top_6_subreddits:
            sub_data = sentiment_subreddit_time_pd[
                sentiment_subreddit_time_pd['subreddit'] == subreddit
            ].copy()
            sub_data['year_month'] = pd.to_datetime(sub_data['year_month'])
            sub_data = sub_data.sort_values('year_month')
            
            plt.plot(
                sub_data['year_month'],
                sub_data['avg_sentiment'],
                marker='o',
                linewidth=2,
                markersize=4,
                label=subreddit,
                alpha=0.8
            )
        
        plt.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        plt.title("Sentiment Trends by Subreddit Over Time", fontsize=16, weight='bold', pad=20)
        plt.xlabel("Month", fontsize=12)
        plt.ylabel("Average Sentiment Score", fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/NLP_Q2_sentiment_trends_by_subreddit.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved: NLP_Q2_sentiment_trends_by_subreddit.png")
        
        # 4. Sentiment Distribution (Stacked Bar Chart)
        plt.figure(figsize=(14, 8))
        
        # Get top 20 subreddits by total posts
        top_20_subs = sentiment_by_subreddit_pd.nlargest(20, 'total_posts')['subreddit'].tolist()
        
        dist_data = sentiment_dist_by_subreddit_pd[
            sentiment_dist_by_subreddit_pd['subreddit'].isin(top_20_subs)
        ]
        
        pivot_dist = dist_data.pivot(index='subreddit', columns='sentiment_category', values='count')
        pivot_dist = pivot_dist.fillna(0)
        
        # Normalize to percentages
        pivot_dist_pct = pivot_dist.div(pivot_dist.sum(axis=1), axis=0) * 100
        
        # Define colors
        colors_dict = {'negative': '#d73027', 'neutral': '#fee08b', 'positive': '#1a9850'}
        colors_list = [colors_dict.get(col, 'gray') for col in pivot_dist_pct.columns]
        
        pivot_dist_pct.plot(
            kind='barh',
            stacked=True,
            figsize=(14, 8),
            color=colors_list,
            alpha=0.85
        )
        
        plt.xlabel("Percentage (%)", fontsize=12)
        plt.ylabel("Subreddit", fontsize=12)
        plt.title("Sentiment Distribution Across Top 20 Subreddits", fontsize=16, weight='bold', pad=20)
        plt.legend(title='Sentiment', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/NLP_Q2_sentiment_distribution_stacked.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved: NLP_Q2_sentiment_distribution_stacked.png")
        
        # 5. Sentiment Heatmap Over Time
        plt.figure(figsize=(16, 10))
        
        # Select top 15 subreddits
        top_15_subs = sentiment_by_subreddit_pd.nlargest(15, 'total_posts')['subreddit'].tolist()
        
        heatmap_data = sentiment_subreddit_time_pd[
            sentiment_subreddit_time_pd['subreddit'].isin(top_15_subs)
        ].pivot(index='subreddit', columns='year_month', values='avg_sentiment')
        
        sns.heatmap(
            heatmap_data,
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Average Sentiment'},
            linewidths=0.5,
            annot=False
        )
        
        plt.title("Sentiment Heatmap: Top 15 Subreddits Over Time", fontsize=16, weight='bold', pad=20)
        plt.xlabel("Month", fontsize=12)
        plt.ylabel("Subreddit", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/NLP_Q2_sentiment_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved: NLP_Q2_sentiment_heatmap.png")
        
        logger.info("\n" + "=" * 80)
        logger.info("Analysis completed successfully!")
        logger.info(f"Results saved to S3: {output_base}")
        logger.info(f"Visualizations saved to: {OUTDIR}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        spark.stop()
        logger.info("Spark session stopped")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 NLP_Q2.py <net-id> [spark-master-url]")
        print("Example: python3 NLP_Q2.py cg1372 spark://172.31.91.143:7077")
        sys.exit(1)
    
    net_id = sys.argv[1]
    spark_master = sys.argv[2] if len(sys.argv) > 2 else None
    
    main(net_id, spark_master)