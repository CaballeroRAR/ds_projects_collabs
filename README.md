# Data Science projects in collaboration.

# 1. Retail Customer Segmentation – UCI Machine Learning Repository

## Overview
End-to-end data engineering and unsupervised learning project focused on customer segmentation and analysis of transactional data from a UK-based non-store online retailing company. The project implements a custom Python data pipeline to clean raw transaction logs, followed by K-Means clustering to identify distinct customer groups based on recency, frequency, and monetary (RFM) patterns.

## Dataset
- **Source:** UCI Machine Learning Repository – Online Retail Data Set
- **Period:** Dec 1, 2010 – Dec 9, 2011
- **Records:** ~541,909 transactions
- **Scope:** International sales (mostly UK) with product descriptions and quantities
- **Data Quality:** Initial dataset required extensive cleaning (nulls, duplicates, invalid stock codes, zero prices)

## Tech Stack
![Python](https://img.shields.io/badge/Python-Medium-182625?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Medium-3D5A73?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Medium-28403D?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Medium-011F26?style=flat&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Medium-F2380F?style=flat&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Medium-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Medium-28403D?style=flat&logo=jupyter&logoColor=white)

## Methodology

### Data Preparation
- Loaded raw dataset and inspected structure using Pandas
- **Data Cleaning Pipeline:**
  - Normalized column names (lowercase, underscores)
  - Dropped rows with missing Customer IDs
  - Identified and removed duplicate transactions
  - Filtered out transactions with zero or negative prices
  - Cleaned Stock Codes by removing non-standard characters (abnormal codes)
  - Filtered for valid Invoice formats (6-digit numeric codes)
- Generated Pickle files (`.pkl`) to store cleaned data states

### Feature Engineering
- **RFM Analysis Construction:**
  - **Recency:** Calculated days since last purchase for each customer
  - **Frequency:** Counted total number of unique invoices per customer
  - **Monetary:** Summed total transaction value per customer
- Aggregated transaction-level data into customer-level metrics for clustering

### Pipeline Development
- Developed a custom modular Python pipeline (`pipeline.py`)
- Created reusable functions for string cleaning, regex filtering, and data validation
- Automated sequential execution of cleaning steps with state management (handling dependencies like abnormal stock code lists)
- Integrated pipeline into Jupyter workflows for iterative testing

### Analysis
**Exploratory Data Analysis (EDA)**
- Analyzed sales distribution by country and top-selling products
- Investigated unit price distributions and quantity patterns
- Examined customer geographic distribution and purchase frequency

**Unsupervised Learning: K-Means Clustering**
- Applied K-Means clustering to the RFM dataset
- Segmented customers into distinct groups (e.g., High Value, At Risk, New, Lost)
- Determined optimal cluster count using Elbow Method (implied)

**Visualizations**
- Distribution plots for RFM variables
- Cluster scatter plots visualizing customer segments
- Bar charts for sales by country and category

## Key Findings
- **Data Cleaning Impact:** Significant portion of data removed due to missing customer IDs, cancellations (negative quantities), and zero-price transactions
- **Customer Segmentation:** Identified distinct clusters allowing for targeted marketing strategies (e.g., VIP retention vs. re-engagement campaigns)
- **Geographic Concentration:** Majority of transactions and customers based in the UK, with a significant long tail of international sales
- **Stock Code Anomalies:** Isolated and handled non-transactional entries (e.g., bank charges, gifts, test codes) to ensure analysis accuracy

## Project Structure
```
1-cluster_retail_uci/
├── src/
│   ├── __init__.py
│   ├── pipeline.py          # Main cleaning pipeline
│   ├── functions.py         # Utility functions
│   ├── k_means_function.py  # Clustering functions
│   ├── viz_functions.py     # Visualization functions
│   ├── elbow_method.py      # Cluster optimization
│   ├── plot_save.py         # Plot utilities
│   └── notebook_pipeline_test.ipynb  # Pipeline testing
├── notebook/
│   ├── cluster_retail.ipynb  # Main analysis
│   └── df_2010-2011.pkl     # Cleaned data
├── dataset/                 # Raw data files
├── pyrightconfig.json       # Type checking configuration
└── README.md
```

## Deliverables
Modular Python cleaning pipeline, clean datasets in Pickle format, trained clustering model, and RFM analysis enabling customer segmentation strategies for targeted marketing and retention planning.

**Skills Demonstrated:** Data Engineering (ETL), Custom Pipeline Development, Python Scripting, Data Cleaning & Preprocessing, Regex, Exploratory Data Analysis (EDA), Unsupervised Machine Learning (K-Means), RFM Analysis, Customer Segmentation

# 2. The Astroturfing Report: Analyzing Authenticity in Top Comments

## Overview
A forensic Data Science investigation aimed at quantifying and analyzing the authenticity of public discourse on social media. By combining metadata heuristics and Natural Language Processing (NLP), the project identifies suspicious account behavior, detects bot-driven "astroturfing" campaigns, and segments organic sentiment from manufactured consensus.

## Dataset
- **Source:** Reddit Threads (Scraped via Direct JSON Access without relying on official API)
- **Scope:** Comments and Author Information
- **Features Extracted:** Body text, karma scores, account age, controversiality, etc.

## Tech Stack
![Python](https://img.shields.io/badge/Python-Medium-182625?style=flat&logo=python&logoColor=white)
![GCP](https://img.shields.io/badge/GCP-Cloud-4285F4?style=flat&logo=google-cloud&logoColor=white)
![Reddit](https://img.shields.io/badge/Reddit-API-FF4500?style=flat&logo=reddit&logoColor=white)
![Transformers](https://img.shields.io/badge/NLP-Transformers-yellow?style=flat&logo=huggingface&logoColor=white)
![BigQuery](https://img.shields.io/badge/BigQuery-Analytics-blue?style=flat&logo=google-cloud&logoColor=white)

## Methodology

### Data Collection & Ingestion (Phase 1)
- Scraped deeply nested Reddit comment trees using local Python scripts to bypass API limits (`manual_scrapping.py`).
- Extracted and computed author "Trust Scores" using account metrics like karma and creation date constraints.
- Flattened the extracted JSON into CSV outputs format (`data_transformation.py`).
- Ingested the locally cleaned CSV into Google BigQuery as the `comments_structured` table (`gcp_ingestion.py`).

### Data Transformations - Medallion Architecture (Phase 2)
- Built a GCP transformation pipeline routing the raw ingested tables into conceptual layers:
  - **Bronze:** Exposes raw ingested tables.
  - **Silver:** Cleans, deduplicates, and strongly-types the comments (`silver_comments_clean`).
  - **Gold:** Aggregates data by author profiles (`gold_author_profiles`) and formats clean data for NLP (`gold_nlp`).

### Deep NLP Processing & Reporting (Phase 3 & 4)
- Orchestrated heavy ML operations out-of-core using mapped Google Colab Notebooks with GPU processing.
- Pulled the `gold_nlp` tables back from BigQuery.
- Used `paraphrase-multilingual-MiniLM-L12-v2` (BERT) to embed Spanish/English comments.
- Mapped comments to clustered narrative structures using `UMAP` dimensionality reduction and `HDBSCAN` density-based clustering.
- Evaluated emotional trajectory with Multilingual `XLM-RoBERTa` for sentiment tracking.
- Synced the unified `gold_nlp_results` dataframe back to BigQuery with their assigned cluster IDs and sentiment probabilities.
- Generated localized Python visualizations (`src/visualization/plots.py`) combining author trust heuristics and NLP results (Quadrant Maps, Timelines, Actionable Wordclouds) to clearly expose bot-driven narratives.

## Project Structure
```
2-nlp-astroturfing-report/
├── src/
│   ├── forensics/
│   │   ├── manual_scrapping.py     # Local Reddit JSON Scraper
│   │   └── trust_scoring.py        # Karma/Age heuristics 
│   ├── infra/
│   │   ├── data_transformation.py  # Flattens nested JSON trees
│   │   ├── gcp_ingestion.py        # BigQuery integration
│   │   └── run_transformations.py  # Orchestrates SQL Medallion scripts
│   ├── bq_sql_transformations/
│   │   ├── 01_bronze.sql           
│   │   ├── 02_silver.sql           
│   │   ├── 03_gold.sql             
│   │   ├── 04_gold_nlp.sql         
│   │   └── 05_gold_reporting.sql   # Unified View for BI
│   ├── nlp/
│   │   ├── data_loader.py          # BigQuery cloud I/O
│   │   ├── embeddings_cluster.py   # UMAP & HDBSCAN
│   │   ├── sentiment_analysis.py   # HuggingFace RoBERTa
│   │   └── nlp_pipeline.py         # Full NLP orchestration
│   └── visualization/
│       └── plots.py                # Python Statistical Dashboards
├── notebook/
│   └── control_notebook.ipynb      # Central control interface (Local + Colab)
├── data/                           # Ignored local raw/structured CSV files
├── requirements.txt
└── README.md
```

## Deliverables
Detailed forensics and extraction python scripts, BigQuery integrated Medallion tables, scalable multidimensional ML models (Embeddings, Clustering, Sentiment), and a final interactive data visualization suite explicitly detailing astroturfing narratives and behaviors.

**Skills Demonstrated:** Advanced NLP (Multilingual BERT, HDBSCAN, Sentiment), Cloud Data Engineering (GCP Medallion Architecture, BigQuery), Network Forensics, Behavioral Clustering, API Mining (Direct JSON extraction), Data Visualization (Seaborn, Wordclouds).
