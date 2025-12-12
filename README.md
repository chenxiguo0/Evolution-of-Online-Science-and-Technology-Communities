# Project: Evolution of Online Science and Technology Communities on Reddit

**Team:** Chenxi Guo, Linjin He, Xiaoya Meng

## High-Level Problem Statement

This project analyzes how online science and technology communities on Reddit evolve, interact, and structure their discussions around AI and emerging technologies.

## Dataset

The analysis is based on a large-scale dataset of over 13 million Reddit comments and submissions from June 2023 to July 2024. The data is filtered to include discussions from 100 subreddits related to science, technology, AI, and programming. All processing and analysis were performed using a distributed Apache Spark cluster to handle the scale of the data.

## Project Summary

This project is divided into three main analytical components:

1.  **Exploratory Data Analysis (EDA):** We investigated community activity, user engagement, attention concentration, and user overlap across different subreddits to understand the structural dynamics of these online communities.
2.  **Natural Language Processing (NLP):** We applied topic modeling (LDA) and sentiment analysis (VADER) to uncover the dominant themes of discussion, track emotional patterns, and assess the impact of external events on community discourse.
3.  **Machine Learning (ML):** We developed classification and clustering models to predict comment quality and identify distinct discussion communities based on language use, providing tools for content moderation and audience segmentation.

All findings, visualizations, and detailed analyses are presented on the project website.

## Key Findings

- **Event-Driven Engagement:** Community activity, particularly in AI-focused subreddits, spikes in response to major industry events (e.g., new model releases), but the overall emotional tone of the communities remains remarkably stable and neutral-to-positive.
- **Core AI Hub:** A few key AI subreddits (`r/ChatGPT`, `r/OpenAI`) form a highly interconnected core, driving the majority of the AI-related discourse on the platform.
- **Dominant Discussion Themes:** The most prevalent topics revolve around human-AI relations, future technologies, and career/learning opportunities. These themes have remained consistent over the one-year analysis period.
- **Predictive Modeling:** While predicting comment "quality" with high precision is challenging, our models show high recall, making them suitable for flagging content for human review. Clustering models effectively separated broad, casual conversations from niche, technical discussions.

## Repository Structure

```
.
├── README.md
├── code/
│   ├── eda/
│   ├── nlp/
│   └── ml/
├── data/
│   ├── csv/
│   └── plots/
├── website-source/
│   ├── *.qmd
│   └── _quarto.yml
└── docs/
    └── (Generated website files)
```

-   **`code/`**: Contains all PySpark scripts for EDA, NLP, and ML analysis.
-   **`data/`**: Contains the results of our analysis, including generated CSV files and plots.
-   **`website/`**: Contains the Quarto source files (`.qmd`) for the project website.
-   **`docs/`**: Contains the rendered HTML files for the final website, suitable for hosting on GitHub Pages.

## How to Build the Website

The project website is built using Quarto.

1.  **Install Quarto:** Follow the instructions at [quarto.org](https://quarto.org/docs/get-started/).

2.  **Navigate to the website directory:**
    ```bash
    cd website-source
    ```

3.  **Render the website:**
    ```bash
    quarto render
    ```
    This command will generate the final HTML files in the `docs/` directory. To preview the site with live reloading, you can use `quarto preview`.