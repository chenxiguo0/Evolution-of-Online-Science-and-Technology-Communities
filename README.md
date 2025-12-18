# Project: Evolution of Online Science and Technology Communities on Reddit

## Project Motivation and Overview

In the digital era, Reddit has become one of the world’s most influential hubs for science and technology discussion. Subreddits devoted to AI, programming, and emerging tech not only aggregate cutting-edge knowledge and best practices, but also reflect the evolving landscape of online expertise, public sentiment, and the social impact of technology. However, the ways these science and technology communities evolve, interact, and structure their collective discourse over time remain underexplored at scale. Questions about the internal dynamics of these communities—what drives user engagement, how do discussion themes change in response to real-world events, and how does sentiment flow through key technology hubs—are increasingly relevant as online dialogue shapes innovation and opinion.

This project aims to systematically analyze how online science and technology communities on Reddit evolved between June 2023 and July 2024, with a special focus on discussions concerning artificial intelligence and emerging technologies. Leveraging big data tools, we track structural change, thematic evolution, user collaboration and clustering across 100 of the most active science and tech subreddits, providing insights into the “digital pulse” of this rapidly shifting sphere.

**Team:** Chenxi Guo, Linjin He, Xiaoya Meng

---

## High-Level Problem Statement

This project investigates how online science and technology communities on Reddit develop, interact, and organize their discussions around AI and emerging technologies, using data-driven methods to uncover hidden patterns of engagement, sentiment, and knowledge diffusion.

## Dataset

Our study is based on a large-scale dataset exceeding 13 million Reddit comments and posts collected from June 2023 to July 2024, filtered to include 100 subreddits dedicated to science, technology, artificial intelligence, and programming. All data processing and analysis was performed using distributed Apache Spark clusters to efficiently handle the vast data volume.

## Project Summary

The project consists of three main analytical components:

1.  **Exploratory Data Analysis (EDA):** Community activity, user participation, attention centralization, and cross-subreddit overlap are quantified to illustrate the underlying social structure of these online spaces.
2.  **Natural Language Processing (NLP):** Topic modeling (LDA) and sentiment analysis (VADER) are used to extract dominant themes, trace sentiment dynamics, and assess the impact of major real-world events on community mood.
3.  **Machine Learning (ML):** Classification and clustering models are built to predict comment quality and to automatically identify distinct discussion communities, supporting moderation and audience research.

The full results, interactive graphs, and detailed methodology are available on the project website.

## Key Findings

- **Event-Driven Engagement:** Spikes in community activity—especially in AI-related subreddits—often correspond to industry breakthroughs, though sentiment remains stably neutral-to-positive overall.
- **AI Hub Centralization:** Subreddits such as `r/ChatGPT` and `r/OpenAI` function as a highly interconnected core, anchoring much of the public discourse around AI.
- **Stable Themes:** Across the year, recurring discussion topics cluster around human-AI relations, future technologies, and educational/career opportunities.
- **Predictive Success & Limitations:** While predicting comment quality with high precision is challenging, our models excel at recall, making them practical for flagging questionable content. Clustering models reveal clear divides between broad general discussions and technical niche conversations.

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

-   **`code/`**: PySpark scripts for EDA, NLP, and ML analyses.
-   **`data/`**: Results of all analyses, including processed CSVs and visualizations.
-   **`website-source/`**: Quarto source files (`.qmd`) for the project website.
-   **`docs/`**: Rendered HTML files, ready for GitHub Pages hosting.

## How to Build the Website

1.  **Install Quarto:** See [quarto.org](https://quarto.org/docs/get-started/).
2.  **Navigate to the website directory:**
    ```bash
    cd website-source
    ```
3.  **Render the website:**
    ```bash
    quarto render
    ```
    The final HTML files will appear in the `docs/` directory. To preview locally with live reload, use `quarto preview`.

---

Thank you for your interest! Feedback and collaboration are welcome.