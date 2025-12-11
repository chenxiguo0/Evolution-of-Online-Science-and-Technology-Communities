#!/usr/bin/env python3
"""
Reddit Data Filtering Example (CLUSTER VERSION).

This script demonstrates how to:
1. Read Reddit comments and submissions data from S3
2. Filter by subreddits of interest
3. Select commonly useful columns
4. Save filtered data back to S3

This is a starting point example. Students can modify this to:
- Add different subreddit filters
- Select additional columns
- Add date range filters
- Combine comments with submissions
- Perform initial exploratory data analysis
"""

import logging
import os
import sys
import time
from typing import (
    List,
    Optional,
)

from pyspark.sql import (
    DataFrame,
    SparkSession,
)
from pyspark.sql.functions import (
    col,
    from_unixtime,
    to_date,
    date_format
)

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


# Example subreddits of interest (students should customize this)
EXAMPLE_SUBREDDITS: List[str] = [
    # AI / ML / Deep Learning (highly active / popular)
    "MachineLearning",          # ML research and discussions
    "ArtificialInteligence",    # AI news and debate
    "OpenAI",                   # OpenAI-related discussions
    "ChatGPT",                  # ChatGPT discussions
    "Futurology",               # Future tech and AI predictions
    "technology",               # Broad tech topics
    "AIethics",                 # Ethical AI discussions
    "deeplearning",             # Deep learning research and projects
    "AI_Agents",                # AI agent development
    "GenerativeAI",             # Generative AI models
    "genai",                    # Generative AI discussions
    "DeepLearningAI",           # Deep Learning AI community
    "NeuralNetworks",           # Neural network research
    "AI_Research",              # AI research papers/discussion
    "MLQuestions",              # Machine learning Q&A
    "ReinforcementLearning",    # RL topics and papers
    "ComputerVision",           # Computer vision research
    "AI_Art",                   # AI-generated art (high engagement) # NEW
    "LLM",                      # Large Language Model discussions # NEW
    "AI_Programming",           # AI applied to coding # NEW

    # Programming / Data / Computing (highly active)
    "programming",              # General programming
    "datascience",              # Data science topics
    "bigdata",                  # Big data discussion
    "computerscience",          # Computer science discussions
    "coding",                   # Coding discussions
    "learnprogramming",         # Beginners learning programming
    "robotics",                 # Robotics and automation
    "DataIsBeautiful",          # Data visualization community
    "technews",                 # Tech news discussions
    "Python",                   # Python programming
    "JavaScript",               # JavaScript programming
    "Rust",                     # Rust language
    "GoLang",                   # Go programming
    "Linux",                    # Linux OS and usage
    "OpenSource",               # Open source software
    "CyberSecurity",            # Cybersecurity topics
    "Hacking",                  # Ethical hacking discussions
    "DevOps",                   # DevOps and cloud practices
    "CloudComputing",           # Cloud technologies
    "webdev",                   # Web development, active # NEW
    "programmerhumor",          # Programming humor, very active # NEW
    "cscareerquestions",        # CS career questions, high activity # NEW
    "learnpython",              # Python beginner questions # NEW
    "learnjava",                # Java beginner questions # NEW

    # Broad Science / STEM (moderate to high activity)
    "science",                  # General science
    "Physics",                  # Physics discussions
    "Chemistry",                # Chemistry community
    "Biology",                  # Biology discussions
    "Mathematics",              # Math discussions
    "Astronomy",                # Astronomy discussions, popular # NEW
    "Space",                    # Space exploration
    "EarthScience",             # Earth science topics
    "Geology",                  # Geology discussions
    "Neuroscience",             # Brain research
    "EnvironmentalScience",     # Environmental topics
    "Engineering",              # Engineering broad topics
    "ElectricalEngineering",    # Electrical engineering
    "MechanicalEngineering",    # Mechanical engineering
    "CivilEngineering",         # Civil engineering
    "MaterialsScience",         # Material science
    "Statistics",               # Statistical discussions
    "QuantumComputing",         # Quantum computing
    "Genetics",                 # Genetics research
    "Ecology",                  # Ecology discussions
    "Bioinformatics",           # Bioinformatics research
    "Nanotechnology",           # Nano-tech discussions
    "Energy",                   # Energy tech
    "RoboticsEngineering",      # Engineering robotics
    "ComputerEngineering",      # Computer engineering
    "TechInnovation",           # Science & tech innovation
    "Astrophysics",             # Astrophysics research
    "ScientificResearch",       # General scientific research
    "ScienceNews",              # Science news
    "FutureTechnology",         # Future tech
    "AIinScience",              # AI applications in science
    "SpaceX",                   # SpaceX updates, popular
    "NASA",                     # NASA discussions
    "Cosmology",                # Cosmology research
    "futurescience",            # Emerging science topics # NEW
    "biologymemes",             # Humorous biology posts, active # NEW
    "physicsmemes",             # Humorous physics posts, active # NEW
    "spaceflight",              # Spaceflight enthusiasts, high activity # NEW

    # General Technology / Future Trends / Science Popularization (high engagement)
    "TechNewsToday",            # Daily tech news
    "TechTalk",                 # Tech discussions
    "Innovation",               # Innovative tech
    "Futurism",                 # Futuristic tech & society
    "TechCulture",              # Tech culture discussions
    "AIforGood",                # AI for social good
    "TechEntrepreneur",         # Tech entrepreneurship
    "Startups",                 # Startup discussions
    "EdTech",                   # Education tech
    "QuantumPhysics",           # Quantum physics discussions
    "Blockchain",               # Blockchain discussions
    "Cryptography",             # Cryptography discussions
    "MachineLearningMemes",     # ML humor
    "DataScienceMemes",         # Data science humor
    "gadgets",                  # Tech gadget enthusiasts # NEW
    "tech",                     # Broad tech discussions, very active # NEW
    "futurewhatif",             # Speculative future/tech # NEW
    "Entrepreneur"              # Tech/business innovation # NEW
]







# Commonly interesting columns for analysis
COMMENT_COLUMNS: List[str] = [
    "id",
    "subreddit",
    "author",
    "body",
    "score",
    "created_utc",
    "parent_id",
    "link_id",
    "controversiality",
    "gilded",
    "total_awards_received",
    "permalink"
]

SUBMISSION_COLUMNS: List[str] = [
    "id",
    "subreddit",
    "author",
    "title",
    "selftext",
    "score",
    "created_utc",
    "num_comments",
    "url",
    "over_18",
    "upvote_ratio",
    "link_flair_text",
    "total_awards_received",
    "permalink"
]



def _create_spark_session(
    master_url: str,
) -> SparkSession:
    """
    Create a Spark session optimized for Reddit data processing.

    Args:
        master_url: URL of the Spark master node

    Returns:
        Configured SparkSession
    """
    spark = (
        SparkSession.builder
        .appName("Reddit_Data_Filter_Example")
        .master(master_url)
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.executor.cores", "2")
        .config("spark.cores.max", "6")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.InstanceProfileCredentialsProvider")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    logger.info("Spark session created successfully")
    return spark


def _read_reddit_data(
    spark: SparkSession,
    data_type: str,
    net_id: str,
) -> DataFrame:
    """
    Read Reddit data (comments or submissions) from S3.

    Args:
        spark: Active SparkSession
        data_type: Either "comments" or "submissions"
        net_id: Student's net ID for S3 bucket access

    Returns:
        DataFrame with Reddit data
    """
    s3_path = f"s3a://{net_id}-dsan6000-datasets-2025/reddit/parquet/{data_type}/"

    logger.info(f"Reading {data_type} data from: {s3_path}")
    print(f"Reading {data_type} data from S3...")

    df = spark.read.parquet(s3_path)

    row_count = df.count()
    logger.info(f"Loaded {row_count:,} rows of {data_type} data")
    print(f"Loaded {row_count:,} rows of {data_type} data")

    return df


def _filter_by_subreddits(
    df: DataFrame,
    subreddits: List[str],
    data_type: str,
) -> DataFrame:
    """
    Filter DataFrame by subreddits of interest.

    Args:
        df: Input DataFrame
        subreddits: List of subreddit names to keep
        data_type: Type of data being filtered

    Returns:
        Filtered DataFrame
    """
    logger.info(f"Filtering {data_type} by {len(subreddits)} subreddits")
    print(f"\nFiltering {data_type} by subreddits of interest...")
    print(f"Target subreddits: {', '.join(subreddits)}")

    filtered_df = df.filter(col("subreddit").isin(subreddits))

    row_count = filtered_df.count()
    logger.info(f"After filtering: {row_count:,} rows")
    print(f"After filtering: {row_count:,} rows")

    return filtered_df


def _select_columns(
    df: DataFrame,
    columns: List[str],
    data_type: str,
) -> DataFrame:
    """
    Select specific columns and add date/year_month columns.

    Args:
        df: Input DataFrame
        columns: List of column names to select
        data_type: Type of data being processed ('comment' or 'submission')

    Returns:
        DataFrame with selected columns and additional date columns
    """
    logger.info(f"Selecting {len(columns)} columns from {data_type}")

    available_columns = [c for c in columns if c in df.columns]
    df_selected = df.select(*available_columns)

    df_selected = (
        df_selected
        .withColumn("date", to_date(from_unixtime(col("created_utc"))))
        .withColumn("year_month", date_format(col("date"), "yyyy-MM"))
    )

    logger.info(f"Selected columns: {', '.join(available_columns)} + ['date', 'year_month']")
    return df_selected


def _save_filtered_data(
    df: DataFrame,
    data_type: str,
    net_id: str,
) -> None:
    """
    Save filtered data back to S3.

    Args:
        df: Filtered DataFrame to save
        data_type: Either "comments" or "submissions"
        net_id: Student's net ID for S3 bucket access
    """
    output_path = f"s3a://{net_id}-dsan6000-datasets-2025/project/reddit/parquet/{data_type}/"

    logger.info(f"Saving filtered {data_type} to: {output_path}")
    print(f"\nSaving filtered {data_type} to S3...")
    print(f"Output path: {output_path}")

    df.write.mode("overwrite").parquet(output_path)

    logger.info(f"Successfully saved filtered {data_type}")
    print(f"Successfully saved filtered {data_type}")


def _show_sample_data(
    df: DataFrame,
    data_type: str,
    num_rows: int = 5,
) -> None:
    """
    Display sample data for verification.

    Args:
        df: DataFrame to sample
        data_type: Type of data being displayed
        num_rows: Number of rows to display
    """
    print(f"\nSample {data_type} data (first {num_rows} rows):")
    print("=" * 80)
    df.show(num_rows, truncate=50)


def _process_data_type(
    spark: SparkSession,
    data_type: str,
    subreddits: List[str],
    net_id: str,
) -> None:
    """
    Process Reddit data (comments or submissions).

    Args:
        spark: Active SparkSession
        data_type: Either "comments" or "submissions"
        subreddits: List of subreddits to filter
        net_id: Student's net ID
    """
    logger.info(f"Processing {data_type} data")
    print(f"\n{'=' * 80}")
    print(f"Processing {data_type.upper()} Data")
    print(f"{'=' * 80}")

    start_time = time.time()

    df = _read_reddit_data(spark, data_type, net_id)

    if data_type == "comments":
        columns = COMMENT_COLUMNS
    else:
        columns = SUBMISSION_COLUMNS

    filtered_df = _filter_by_subreddits(df, subreddits, data_type)

    selected_df = _select_columns(filtered_df, columns, data_type)

    _show_sample_data(selected_df, data_type)

    _save_filtered_data(selected_df, data_type, net_id)

    elapsed_time = time.time() - start_time
    logger.info(f"Processed {data_type} in {elapsed_time:.1f} seconds")
    print(f"\nProcessed {data_type} in {elapsed_time:.1f} seconds")


def main() -> int:
    """Main function for Reddit data filtering example."""
    logger.info("Starting Reddit Data Filtering Example")
    print("=" * 80)
    print("REDDIT DATA FILTERING EXAMPLE (CLUSTER MODE)")
    print("=" * 80)

    if len(sys.argv) < 2:
        print("❌ Error: NET_ID not provided")
        print("Usage: python reddit_data_filter_example.py <NET_ID> [master_url]")
        print("\nExample:")
        print("  python reddit_data_filter_example.py abc123 spark://10.0.1.100:7077")
        print("  or")
        print("  export MASTER_PRIVATE_IP=10.0.1.100")
        print("  python reddit_data_filter_example.py abc123")
        return 1

    net_id = sys.argv[1]
    logger.info(f"Using NET_ID: {net_id}")
    print(f"\nNET_ID: {net_id}")

    if len(sys.argv) > 2:
        master_url = sys.argv[2]
    else:
        master_private_ip = os.getenv("MASTER_PRIVATE_IP")
        if master_private_ip:
            master_url = f"spark://{master_private_ip}:7077"
        else:
            print("❌ Error: Master URL not provided")
            print("Provide as argument or set MASTER_PRIVATE_IP environment variable")
            return 1

    print(f"Spark Master: {master_url}")
    logger.info(f"Using Spark master URL: {master_url}")

    print(f"\nExample subreddits to filter: {', '.join(EXAMPLE_SUBREDDITS)}")
    print("\n⚠️  NOTE: Customize EXAMPLE_SUBREDDITS list for your analysis!")

    overall_start = time.time()

    spark = _create_spark_session(master_url)

    try:
        _process_data_type(spark, "comments", EXAMPLE_SUBREDDITS, net_id)

        _process_data_type(spark, "submissions", EXAMPLE_SUBREDDITS, net_id)

        success = True
        logger.info("Data filtering completed successfully")

    except Exception as e:
        logger.exception(f"Error during data processing: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        success = False

    spark.stop()
    logger.info("Spark session stopped")

    overall_time = time.time() - overall_start

    print("\n" + "=" * 80)
    if success:
        print("✅ REDDIT DATA FILTERING COMPLETED SUCCESSFULLY!")
        print(f"\nTotal execution time: {overall_time:.1f} seconds")
        print("\nFiltered data saved to:")
        print(f"  - s3://{net_id}-dsan6000-datasets-2025/project/reddit/parquet/comments/")
        print(f"  - s3://{net_id}-dsan6000-datasets-2025/project/reddit/parquet/submissions/")
        print("\nNext steps:")
        print("  1. Verify the filtered data in S3")
        print("  2. Customize the subreddit list and columns for your analysis")
        print("  3. Begin EDA on the filtered dataset")
        print("  4. Develop NLP and ML pipelines")
    else:
        print("❌ Data filtering failed - check error messages above")
    print("=" * 80)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
