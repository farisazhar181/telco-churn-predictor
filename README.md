# Telco Customer Churn Predictor

**Live Demo:** https://your-app.streamlit.app

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
