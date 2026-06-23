# Telco Customer Churn Predictor

**Live Demo:** [https://kadfygzfk6spvkvfbmjn2o.streamlit.app/](https://kadfygzfk6spvkvfbmjn2o.streamlit.app/)

This project demonstrates a complete end-to-end machine learning workflow predicting customer churn for a telecommunications company. It covers exploratory data analysis (EDA), feature engineering, model training (Logistic Regression, Random Forest, XGBoost), evaluation, and a deployed Streamlit web application.

## Key Findings

- **Model Performance:** The final XGBoost model achieves an 84.6% ROC-AUC, correctly identifying 79.4% of actual churners.
- **Contract Type:** Month-to-month contracts are the strongest predictor of churn. Customers on these contracts churn at a rate of 42.7%, compared to 11.3% for one-year and 2.8% for two-year contracts.
- **Internet Service:** Fiber optic customers churn at 41.9%, which is nearly double the rate for DSL customers (19.0%). This suggests a potential pricing or value mismatch for fiber optic services.
- **Tenure Risk:** The first year represents the highest-risk window for customer attrition. Median tenure for churners is 10.0 months versus 38.0 months for non-churners.
- **Add-on Services:** Customers without TechSupport churn at 41.6% compared to 15.2% for those with it, showing that add-on services are strongly protective.

## Data

The dataset used is the IBM Telco Customer Churn dataset.

**Note on Data Availability:** Due to size and licensing, the raw CSV data files and processed Parquet files are gitignored and not included in this repository. You can download the raw dataset directly from Kaggle:
[Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Place the downloaded `WA_Fn-UseC_-Telco-Customer-Churn.csv` into the `data/raw/` directory before running the notebooks.

## Tech Stack

The project is built using the following technologies:
- Python for the core programming language.
- Pandas and NumPy for data manipulation and preprocessing.
- Scikit-learn and XGBoost for machine learning modeling.
- Streamlit for the interactive web application frontend.
- Matplotlib and Seaborn for data visualization.
- SHAP for model explainability.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [EDA & Preprocessing](notebooks/01_eda_and_preprocessing.ipynb) | Data cleaning, feature engineering, class imbalance analysis |
| 2 | [Model Training](notebooks/02_model_training.ipynb) | Logistic Regression, Random Forest, XGBoost comparison and selection |
| 3 | [Evaluation & Explainability](notebooks/03_evaluation_and_explainability.ipynb) | Confusion matrix, ROC-AUC, SHAP global and local explanations |

## Project Structure

```
telco-churn-predictor/
├── app.py                  # Streamlit web application
├── requirements.txt
├── data/
│   ├── raw/                # CSV from Kaggle (gitignored)
│   └── processed/          # Parquet splits (gitignored)
├── models/
│   ├── churn_model.joblib
│   └── feature_columns.joblib
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation_and_explainability.ipynb
├── figures/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── shap_summary.png
└── src/
    ├── preprocess.py
    └── predict.py
```

## Installation and Usage

To run this project locally, clone the repository and install the dependencies:

```bash
git clone https://github.com/farisazhar181/telco-customer-churn.git
cd telco-customer-churn
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

To run the Streamlit app:
```bash
streamlit run app.py
```
