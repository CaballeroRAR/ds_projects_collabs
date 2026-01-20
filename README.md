# ds_projects_collabs

Data Science projects in collaboration.

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
├───dataset
│   └───online_retail_II.xlsx (or .csv)
├───functions
│   ├───__init__.py
│   ├───pipeline.py
│   └───functions.py
├───notebook
│   ├───cluster_retail.ipynb
│   └───df_2010-2011.pkl
└───README.md
```

## Deliverables
Modular Python cleaning pipeline, clean datasets in Pickle format, trained clustering model, and RFM analysis enabling customer segmentation strategies for targeted marketing and retention planning.

**Skills Demonstrated:** Data Engineering (ETL), Custom Pipeline Development, Python Scripting, Data Cleaning & Preprocessing, Regex, Exploratory Data Analysis (EDA), Unsupervised Machine Learning (K-Means), RFM Analysis, Customer Segmentation
