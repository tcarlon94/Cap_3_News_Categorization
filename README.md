# News Categorization with NLP

## Overview
This project builds a supervised text classification pipeline to assign news articles to content categories. It demonstrates data cleaning, exploratory data analysis, NLP preprocessing, feature engineering, and model evaluation.

## Business Problem
News publishers and media platforms need automated ways to categorize articles for search, recommendations, and content organization. This project shows how a machine learning classifier can reduce manual tagging and improve content workflows.

## Repository structure
- `data/raw/` — original dataset source files
- `data/processed/` — cleaned and transformed dataset outputs
- `notebooks/` — sequential analysis and modeling notebooks
- `reports/` — visual summaries and export figures
- `.gitignore` — files and directories excluded from Git
- `requirements.txt` — Python dependencies for reproducibility

## Data
- Source: `data/raw/News_Category_Dataset_v3.json`
- Dataset: news articles with category labels
- Notes: the raw dataset is imported, cleaned, and then exported to `data/processed/`

## Workflow
1. `notebooks/01_data_cleaning.ipynb` — ingest raw data and clean text fields
2. `notebooks/02_preprocessing.ipynb` — preprocess text, remove stop words, and build TF-IDF features
3. `notebooks/03_eda.ipynb` — explore distributions, category imbalance, and text statistics
4. `notebooks/04_modeling.ipynb` — train, evaluate, and compare classifiers

## Key results
| Model | Accuracy | Macro F1 | Notes |
|---|---:|---:|---|
| Naive Bayes | `69.0%` | `0.63` | Baseline text classifier |
| Logistic Regression | `77.0%` | `0.71` | Strong linear baseline |
| LightGBM | `83.0%` | `0.78` | Best tree-based baseline |
| Tuned LightGBM | `85.0%` | `0.80` | Best overall performance after Optuna tuning |

## Key Findings
- LightGBM produced the strongest classification performance, outperforming Naive Bayes and Logistic Regression.
- Optuna tuning improved the final LightGBM model to `85.0%` test accuracy and `0.80` macro F1.
- The model is strong overall, but smaller categories still lag behind larger classes, suggesting further work on imbalance handling could help.

## Screenshots
![Category distribution](reports/figures/class_distribution.png/)
![Category Keywords](reports/figures/keywords_by_category.png)
![Confusion matrix](reports/figures/lightgbm_confusion_matrix.png)

## How to Run
1. Clone the repo
2. Install dependencies
3. Run notebooks in order

## Future Improvements
- Add transformer-based model
- Deploy as a simple Streamlit app
- Add real-time article classification pipeline
