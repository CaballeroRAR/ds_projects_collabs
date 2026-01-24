***

# Dataset Overview: Online Retail II

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)  
###

## 1. **Business Context**

### 1.a Description
This dataset contains all transactions occurring for a UK-based and registered non-store online retail between **01/12/2009** and **09/12/2011**. The company primarily sells unique all-occasion gifts, and many customers are wholesalers.

### 1.b Purpose
This analysis goal is to: cluster clients successfully, leave a ready to use cleaning process for future datasets, and provide insights for marketing strategies.
The delivery, description of the clusters based on RFM, a clear explanation of the cleaning process, a ready-to-use model for future predictions.
Optional: A interactive dashboard that allows to explore the clusters.

### 1.c Project Structure
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
###

## 2. **EDA**

### 2.a Structure & Format
*   **Type:** Multivariate, Time-Series.
*   **Attribute Type:** Categorical, Integer, Real.
*   **Format:** 2 Excel files (.xlsx).
    *   `Year 2009-2010`
    *   `Year 2010-2011`
*   **Missing Values:** Yes (specifically in CustomerID).

### 2.b Key Characteristics
*   **Volume:** The dataset contains approximately 1 million records (541,909 rows in 2009-10; 525,461 rows in 2010-11).
*   **Imbalance:** The dataset is heavily imbalanced, as the majority of transactions originate from the United Kingdom. There are extreme outliers in total sales per client.
*   **Noise:** It contains multiple out-of-scope transactions (for example: Invoice numbers that are cancellations, other sort of balance adjustments, and abnormal stock codes) and some null values that need to be cleaned for analysis. More details in the notebook.

### 2.c Data Quality & Anomalies (Findings)

### Missing Values & Dimensionality
*   **Total Entries:** `541,910`
*   **Missing Data:** The columns `Description` and `Customer ID` contain missing values and do not match the total entry count.
*   **Product Inconsistency:** There is a mismatch between unique `StockCode` and `Description` counts:
    *   `4,070` unique `StockCode` values (expected products).
    *   `4,223` unique `Description` values.
    *   *Note:* This suggests duplicates in naming or similar products with slight description variations.

### Statistical Outliers
*   **Quantity:** Contains negative values, indicating returns or adjustments.
    *   Min value: `-80,995`
    *   Max value: `80,995`
    *   *Observation:* The absolute minimum and maximum quantities are identical, likely corresponding to a massive return/cancellation transaction.
*   **UnitPrice:** Contains negative values.
    *   Min value: `-11,062.06`
    *   *Note:* Negative prices typically represent bad debts, adjustments, or system errors.

### Transaction Types
*   **Invoices:**
    *   **Total Unique:** `25,900`
    *   **Cancellations:** Identified a subset `df_cancellation_invoices` containing entries marked as cancellations based on the dataset description.
*   **Invoice Prefixes:** Analysis reveals invoices starting with `'C'` (Cancellation) and `'A'` (Adjustments/Other).

### Abnormal Stock Codes
Several non-standard stock codes were found that do not represent standard retail products. These include transaction fees, postage, and internal bank charges.

**Current State:** Identifying what these codes represent and determining if they should be included in the final analysis.

**List of Abnormal Stock Codes:**
```text
['POST', 'D', 'C2', 'DOT', 'M', 'BANK CHARGES', 'S', 'AMAZONFEE', 
 'DCGS0076', 'DCGS0003', 'gift_0001_40', 'DCGS0070', 'm', 'gift_0001_50', 
 'gift_0001_30', 'gift_0001_20', 'DCGS0055', 'DCGS0072', 'DCGS0074', 
 'DCGS0069', 'DCGS0057', 'DCGSSBOY', 'DCGSSGIRL', 'gift_0001_10', 'PADS', 
 'DCGS0004', 'DCGS0073', 'DCGS0071', 'DCGS0066P', 'DCGS0068', 'DCGS0067', 
 'B', 'CRUK']
```

###

## 3. **Feature Engineering**

## 3.a Features (Variables)
The clean dataset consists of 8 columns:

| Column Name | Type | Description |
| :--- | :--- | :--- |
| **Invoice** | String | A 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter 'c', it indicates a cancellation. |
| **StockCode** | String | A 5-digit integral number uniquely assigned to each distinct product. Added 5 consecutive digits + letter to this logic after EDA |
| **Description** | String | Product (item) name. |
| **Quantity** | Integer | The quantities of each product (item) per transaction. |
| **InvoiceDate** | DateTime | The day and time when a transaction was generated. |
| **UnitPrice** | Float | Product price per unit in sterling (£). |
| **CustomerID** | Integer | A 5-digit float number uniquely assigned to each customer. (Changed to integer) |
| **Country** | String | The name of the country where each customer resides. |

***

