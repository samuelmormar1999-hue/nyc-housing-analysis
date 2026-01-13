
# NYC Housing Market Analysis

## Overview
This project analyzes housing transactions dataset obtained from **Kaggle**.
The dataset contains information such as:
- Sale price
- Location category (borough codes)
- Building size and surface area
- Additional building characteristics

The original CSV file was downloaded from Kaggle and subsequently cleaned and prepared for analysis.

---

## Objectives
- Understand how housing prices are distributed across the NYC market.
- Identify differences in transaction volume across location categories.
- Compare price levels and affordability between cities.
- Distinguish between volume-driven and value-driven housing markets.

---

## Methodology

### 1. Exploratory Data Analysis (Python)
- Data inspection and cleaning using **pandas**
- Analysis of price distributions using histograms and boxplots
- Comparison of housing prices and transaction volumes across location categories
- Visualization of key patterns using **matplotlib**

### 2. SQL-Based Market Analysis
- The cleaned dataset was stored in a local **SQLite** database.
- SQL queries were used to simulate realistic market-driven questions, such as:
  - Where is housing market activity concentrated?
  - Which cities show higher average housing prices?
  - Are there high-price, low-volume markets?
- Queries are documented in `sql/queries.sql`, and selected queries are executed in the notebook to support the analytical narrative.

### 3. Reporting
- Key findings and market implications are summarized in an executive report located in the `reports/` folder.
- Final visualizations supporting the conclusions are saved as image files for easy reuse.

---

## Key Findings
- Housing prices are not uniformly inflated across the NYC market.
- Queens (QN) and Brooklyn (BK) concentrate the highest number of housing transactions, indicating highly liquid markets.
- Manhattan (MN) represents a high-value, low-volume market, characterized by fewer transactions at significantly higher prices.
- Bronx (BX) and Staten Island (SI) show lower transaction volumes and more affordable average prices.
- Significant disparities exist across cities in terms of price levels, price per square foot, and transaction activity.

---

## Project Structure

nyc-housing-market-analysis/
│
├── data/
│ └── nyc_housing_base.csv
│
├── notebooks/
│ └── nyc_housing_analysis.ipynb
│
├── sql/
│ ├── nyc_housing.db
│ └── queries.sql
│
├── reports/
│ ├── summary.md
│ └── figures/
│ ├── price_distribution.png
│ ├── sales_by_location.png
│ 
│
├── README.md
└── requirements.txt

## Technologies used 

- **Python**
    - pandas 
    -matplotlib 
- **SQL** 
    -SQLite (via Python standard library)


## How to Run the Project
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt