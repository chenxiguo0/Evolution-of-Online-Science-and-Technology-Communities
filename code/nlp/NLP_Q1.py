#!/usr/bin/env python3
"""
Topic modeling using TF-IDF and LDA 
"""

import sys
import logging
import os
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lower, regexp_replace, trim, udf, explode, 
    count, desc, year, month, concat_ws, to_date
)
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, CountVectorizer, IDF, HashingTF
)
from pyspark.ml.clustering import LDA
from pyspark.ml import Pipeline
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

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


def preprocess_text_column(df, input_col, output_col):
    """
    Preprocess text: lowercase, remove URLs, special chars, extra whitespace.
    """
    df = df.withColumn(
        output_col,
        lower(col(input_col))
    )
    
    # Remove URLs
    df = df.withColumn(
        output_col,
        regexp_replace(col(output_col), r'http\S+|www\.\S+', ' ')
    )
    
    # Remove special characters and numbers, keep only letters and spaces
    df = df.withColumn(
        output_col,
        regexp_replace(col(output_col), r'[^a-z\s]', ' ')
    )
    
    # Remove extra whitespace
    df = df.withColumn(
        output_col,
        regexp_replace(col(output_col), r'\s+', ' ')
    )
    
    df = df.withColumn(output_col, trim(col(output_col)))
    
    return df


def simple_lemmatize(word):
    """
    Simple lemmatization rules (basic stemming approach).
    For production, consider using spark-nlp library.
    """
    # Remove common suffixes
    if word.endswith('ing'):
        word = word[:-3]
    elif word.endswith('ed'):
        word = word[:-2]
    elif word.endswith('es'):
        word = word[:-2]
    elif word.endswith('s') and len(word) > 3:
        word = word[:-1]
    return word


def build_nlp_pipeline(input_col, min_doc_freq=5, num_topics=10):
    """
    Build ML pipeline for topic modeling:
    1. Tokenization
    2. Stop words removal
    3. TF-IDF
    4. LDA
    """
    # Tokenizer
    tokenizer = Tokenizer(inputCol=input_col, outputCol="tokens")
    
    # Stop words remover (English + custom)
    custom_stopwords = [
        'would', 'could', 'get', 'like', 'one', 'think', 'know', 'see',
        'im', 'dont', 'thats', 'ive', 'youre', 'cant', 'also', 'really',
        'even', 'make', 'need', 'want', 'much', 'people', 'thing', 'things',
        'use', 'work', 'way', 'going', 'time', 'just', 'good', 'new'
    ]
    remover = StopWordsRemover(
        inputCol="tokens", 
        outputCol="filtered_tokens",
        stopWords=StopWordsRemover.loadDefaultStopWords("english") + custom_stopwords
    )
    
    # CountVectorizer for term frequency
    cv = CountVectorizer(
        inputCol="filtered_tokens", 
        outputCol="raw_features",
        minDF=min_doc_freq,
        vocabSize=5000  # Limit vocabulary size
    )
    
    # IDF
    idf = IDF(inputCol="raw_features", outputCol="features")
    
    # LDA
    lda = LDA(
        k=num_topics, 
        maxIter=20,
        optimizer="online",
        featuresCol="features"
    )
    
    pipeline = Pipeline(stages=[tokenizer, remover, cv, idf, lda])
    
    return pipeline


def extract_topics(lda_model, cv_model, num_words=10):
    """Extract top words for each topic."""
    topics = lda_model.describeTopics(maxTermsPerTopic=num_words)
    vocab = cv_model.vocabulary
    
    topic_words = []
    for row in topics.collect():
        topic_id = row['topic']
        term_indices = row['termIndices']
        term_weights = row['termWeights']
        
        words = [vocab[idx] for idx in term_indices]
        topic_info = {
            'topic_id': topic_id,
            'words': words,
            'weights': term_weights
        }
        topic_words.append(topic_info)
    
    return topic_words


def analyze_topics_by_time(transformed_df, date_col='date'):
    """Analyze how topic distributions change over time."""
    # Get dominant topic for each document
    dominant_topic_df = transformed_df.select(
        col(date_col),
        col('subreddit'),
        col('topicDistribution')
    )
    
    # Extract dominant topic (topic with highest probability)
    from pyspark.sql.functions import udf
    from pyspark.sql.types import IntegerType
    
    def get_dominant_topic(topic_dist):
        if topic_dist:
            return int(max(enumerate(topic_dist), key=lambda x: x[1])[0])
        return -1
    
    get_dominant_topic_udf = udf(get_dominant_topic, IntegerType())
    
    dominant_topic_df = dominant_topic_df.withColumn(
        'dominant_topic',
        get_dominant_topic_udf(col('topicDistribution'))
    )
    
    # Add year_month for temporal analysis
    dominant_topic_df = dominant_topic_df.withColumn(
        'year_month',
        concat_ws('-', year(col(date_col)), month(col(date_col)))
    )
    
    # Count topics by month
    topic_trends = dominant_topic_df.groupBy('year_month', 'dominant_topic') \
        .agg(count('*').alias('count')) \
        .orderBy('year_month', 'dominant_topic')
    
    # Count topics by subreddit
    topic_by_subreddit = dominant_topic_df.groupBy('subreddit', 'dominant_topic') \
        .agg(count('*').alias('count')) \
        .orderBy('subreddit', desc('count'))
    
    return topic_trends, topic_by_subreddit


def main(net_id, spark_master=None):
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("NLP Q1: Topic Modeling Analysis")
    logger.info("=" * 80)
    
    # Setup output directory (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    OUTDIR = os.path.join(script_dir,"..", "..", "data", "plots")
    os.makedirs(OUTDIR, exist_ok=True)
    logger.info(f"Output directory: {OUTDIR}")
    
    # Create Spark session
    spark = create_spark_session("Reddit_Topic_Modeling", spark_master)
    logger.info("Spark session created")
    
    try:
        # Load filtered data
        comments_path = f"s3a://{net_id}-dsan6000-datasets/project/reddit/parquet/comments/"
        submissions_path = f"s3a://{net_id}-dsan6000-datasets/project/reddit/parquet/submissions/"
        
        logger.info(f"Loading comments from: {comments_path}")
        comments_df = spark.read.parquet(comments_path)
        
        logger.info(f"Loading submissions from: {submissions_path}")
        submissions_df = spark.read.parquet(submissions_path)
        
        # Sample data for faster processing (optional - remove for full analysis)
        # comments_df = comments_df.sample(0.1, seed=42)
        
        logger.info(f"Comments count: {comments_df.count():,}")
        logger.info(f"Submissions count: {submissions_df.count():,}")
        
        # Filter out deleted/removed content and null text
        comments_df = comments_df.filter(
            (col('body').isNotNull()) & 
            (~col('body').isin('[deleted]', '[removed]')) &
            (col('body') != '')
        )
        
        submissions_df = submissions_df.filter(
            (col('title').isNotNull()) & 
            (col('title') != '')
        )
        
        # Combine title and selftext for submissions
        submissions_df = submissions_df.withColumn(
            'text',
            concat_ws(' ', col('title'), col('selftext'))
        )
        
        # Prepare comments text
        comments_df = comments_df.withColumnRenamed('body', 'text')
        
        # Select relevant columns
        comments_processed = comments_df.select('text', 'subreddit', 'date', 'created_utc')
        submissions_processed = submissions_df.select('text', 'subreddit', 'date', 'created_utc')
        
        # Union both datasets
        all_text = comments_processed.union(submissions_processed)
        logger.info(f"Total documents for analysis: {all_text.count():,}")
        
        # Preprocess text
        logger.info("Preprocessing text...")
        all_text = preprocess_text_column(all_text, 'text', 'clean_text')
        
        # Filter out very short texts
        all_text = all_text.filter(
            (col('clean_text').isNotNull()) & 
            (col('clean_text') != '')
        )
        
        # Cache for performance
        all_text = all_text.repartition(200).cache()
        
        # Build and fit pipeline
        logger.info("Building NLP pipeline...")
        pipeline = build_nlp_pipeline('clean_text', min_doc_freq=10, num_topics=10)
        
        logger.info("Fitting LDA model (this may take a while)...")
        model = pipeline.fit(all_text)
        
        # Get the trained models
        cv_model = model.stages[2]  # CountVectorizer
        lda_model = model.stages[4]  # LDA
        
        # Extract topics
        logger.info("\n" + "=" * 80)
        logger.info("DISCOVERED TOPICS")
        logger.info("=" * 80)
        
        topics = extract_topics(lda_model, cv_model, num_words=15)
        for topic in topics:
            logger.info(f"\nTopic {topic['topic_id']}:")
            words_with_weights = list(zip(topic['words'], topic['weights']))
            for word, weight in words_with_weights[:15]:
                logger.info(f"  {word}: {weight:.4f}")
        
        # Transform data
        logger.info("\nTransforming data with LDA model...")
        transformed = model.transform(all_text)
        
        # Analyze topics over time
        logger.info("\nAnalyzing topic trends...")
        topic_trends, topic_by_subreddit = analyze_topics_by_time(transformed, 'date')
        
        # Save results
        output_base = f"s3a://{net_id}-dsan6000-datasets/project/nlp_analysis/q1_topics/"
        
        logger.info(f"\nSaving results to: {output_base}")
        
        # Save topic trends
        topic_trends.coalesce(1).write.mode('overwrite') \
            .option('header', 'true') \
            .csv(output_base + 'topic_trends/')
        
        # Save topic by subreddit
        topic_by_subreddit.coalesce(1).write.mode('overwrite') \
            .option('header', 'true') \
            .csv(output_base + 'topic_by_subreddit/')
        
        # Display sample results
        logger.info("\n" + "=" * 80)
        logger.info("TOPIC TRENDS OVER TIME (Sample)")
        logger.info("=" * 80)
        topic_trends.show(20, truncate=False)
        
        logger.info("\n" + "=" * 80)
        logger.info("TOP TOPICS BY SUBREDDIT (Sample)")
        logger.info("=" * 80)
        topic_by_subreddit.show(30, truncate=False)
        
        # ==============================
        # VISUALIZATION AND CSV EXPORT
        # ==============================
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 80)
        
        # Convert to pandas for visualization
        topic_trends_pd = topic_trends.toPandas()
        topic_by_subreddit_pd = topic_by_subreddit.toPandas()
        
        # Save CSVs locally
        topic_trends_pd.to_csv(f"{OUTDIR}/NLP_Q1_topic_trends.csv", index=False)
        topic_by_subreddit_pd.to_csv(f"{OUTDIR}/NLP_Q1_topic_by_subreddit.csv", index=False)
        logger.info(f"CSVs saved to {OUTDIR}")
        
        # Set seaborn style
        sns.set_theme(style="whitegrid", palette="Set2")
        
        # 1. Topic Trends Over Time
        plt.figure(figsize=(14, 8))
        
        # Pivot data for better visualization
        pivot_data = topic_trends_pd.pivot(index='year_month', columns='dominant_topic', values='count')
        pivot_data = pivot_data.fillna(0)
        
        # Plot stacked area chart
        pivot_data.plot(kind='area', stacked=True, alpha=0.7, figsize=(14, 8), colormap='tab10')
        
        plt.title("Topic Distribution Over Time", fontsize=16, weight='bold', pad=20)
        plt.xlabel("Month", fontsize=12)
        plt.ylabel("Number of Documents", fontsize=12)
        plt.legend(title='Topic ID', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/NLP_Q1_topic_trends_over_time.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved: NLP_Q1_topic_trends_over_time.png")
        
        # 2. Top Topics by Subreddit (Heatmap)
        plt.figure(figsize=(14, 10))
        
        # Get top 20 subreddits by total posts
        top_subreddits = (
            topic_by_subreddit_pd.groupby('subreddit')['count']
            .sum()
            .nlargest(20)
            .index
        )
        
        heatmap_data = topic_by_subreddit_pd[
            topic_by_subreddit_pd['subreddit'].isin(top_subreddits)
        ].pivot(index='subreddit', columns='dominant_topic', values='count')
        heatmap_data = heatmap_data.fillna(0)
        
        sns.heatmap(
            heatmap_data, 
            cmap='YlOrRd', 
            annot=False, 
            fmt='.0f',
            cbar_kws={'label': 'Number of Documents'},
            linewidths=0.5
        )
        
        plt.title("Topic Distribution across Top 20 Subreddits", fontsize=16, weight='bold', pad=20)
        plt.xlabel("Topic ID", fontsize=12)
        plt.ylabel("Subreddit", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/NLP_Q1_topic_heatmap_by_subreddit.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved: NLP_Q1_topic_heatmap_by_subreddit.png")
        
        # 3. Topic Keywords Visualization
        fig, axes = plt.subplots(5, 2, figsize=(16, 20))
        axes = axes.flatten()
        
        for idx, topic in enumerate(topics):
            if idx >= 10:
                break
            
            ax = axes[idx]
            words = topic['words'][:10]
            weights = topic['weights'][:10]
            
            # Normalize weights for better visualization
            weights_norm = [w / max(weights) for w in weights]
            
            colors = plt.cm.viridis(weights_norm)
            ax.barh(range(len(words)), weights_norm, color=colors)
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words, fontsize=10)
            ax.invert_yaxis()
            ax.set_xlabel('Relative Weight', fontsize=10)
            ax.set_title(f'Topic {topic["topic_id"]}', fontsize=12, weight='bold')
            ax.grid(axis='x', alpha=0.3)
        
        plt.suptitle('Top Keywords for Each Discovered Topic', fontsize=18, weight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(f"{OUTDIR}/NLP_Q1_topic_keywords.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved: NLP_Q1_topic_keywords.png")
        
        # 4. Topic Distribution Pie Chart (Overall)
        plt.figure(figsize=(10, 10))
        
        overall_topic_dist = topic_trends_pd.groupby('dominant_topic')['count'].sum().sort_values(ascending=False)
        
        colors = plt.cm.tab10(range(len(overall_topic_dist)))
        plt.pie(
            overall_topic_dist.values,
            labels=[f'Topic {i}' for i in overall_topic_dist.index],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        plt.title('Overall Topic Distribution', fontsize=16, weight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/NLP_Q1_topic_distribution_pie.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved: NLP_Q1_topic_distribution_pie.png")
        
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
        print("Usage: python3 NLP_Q1.py <net-id> [spark-master-url]")
        print("Example: python3 NLP_Q1.py cg1372 spark://172.31.91.143:7077")
        sys.exit(1)
    
    net_id = sys.argv[1]
    spark_master = sys.argv[2] if len(sys.argv) > 2 else None
    
    main(net_id, spark_master)