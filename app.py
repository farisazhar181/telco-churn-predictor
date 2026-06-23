import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Telco Customer Churn Predictor", layout="wide")

@st.cache_resource
def load_model_and_features():
    # Use relative paths
    model = joblib.load('models/churn_model.joblib')
    features = joblib.load('models/feature_columns.joblib')
    explainer = shap.TreeExplainer(model)
    return model, features, explainer

try:
    model, feature_columns, explainer = load_model_and_features()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Telco Customer Churn Prediction")

# --- Sidebar ---
st.sidebar.header("Customer Attributes")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 1)
MonthlyCharges = st.sidebar.slider("Monthly Charges ($)", 18.25, 118.75, 50.0)
TotalCharges = float(tenure * MonthlyCharges)

Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
InternetService = st.sidebar.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
TechSupport = st.sidebar.selectbox("TechSupport", ["Yes", "No", "No internet service"])
OnlineSecurity = st.sidebar.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
OnlineBackup = st.sidebar.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
DeviceProtection = st.sidebar.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])
StreamingTV = st.sidebar.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
StreamingMovies = st.sidebar.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])
PaymentMethod = st.sidebar.selectbox("PaymentMethod", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
PhoneService = st.sidebar.selectbox("PhoneService", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("MultipleLines", ["Yes", "No", "No phone service"])
SeniorCitizen_check = st.sidebar.checkbox("Senior Citizen")
SeniorCitizen = 1 if SeniorCitizen_check else 0
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
PaperlessBilling = st.sidebar.selectbox("PaperlessBilling", ["Yes", "No"])
gender = st.sidebar.selectbox("gender", ["Male", "Female"])

# --- Preprocessing ---
input_dict = {
    'tenure': tenure,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges,
    'Contract': Contract,
    'InternetService': InternetService,
    'TechSupport': TechSupport,
    'OnlineSecurity': OnlineSecurity,
    'OnlineBackup': OnlineBackup,
    'DeviceProtection': DeviceProtection,
    'StreamingTV': StreamingTV,
    'StreamingMovies': StreamingMovies,
    'PaymentMethod': PaymentMethod,
    'PhoneService': PhoneService,
    'MultipleLines': MultipleLines,
    'SeniorCitizen': SeniorCitizen,
    'Partner': Partner,
    'Dependents': Dependents,
    'PaperlessBilling': PaperlessBilling,
    'gender': gender
}

from src.preprocess import preprocess_input
from src.predict import predict_churn

df_encoded = preprocess_input(input_dict, feature_columns)

# --- Prediction ---
prob, risk_label, color = predict_churn(model, df_encoded)

st.header("Prediction Result")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"### Churn Probability: **{prob * 100:.1f}%**")
    st.markdown(f"### Risk Level: <span style='color:{color}'>{risk_label}</span>", unsafe_allow_html=True)
    
    # SHAP Explanation
    shap_values = explainer(df_encoded)
    
    # We want to extract top 5 features
    shap_vals = shap_values.values[0]
    # np.argsort returns indices in ascending order, we want absolute descending
    abs_shap_vals = np.abs(shap_vals)
    top_indices = np.argsort(abs_shap_vals)[-5:][::-1]
    
    # Filter low predictive signal from highlight if we wanted to, but they shouldn't have high SHAP anyway.
    # Generate summary
    top_feature_1_idx = top_indices[0]
    top_feature_2_idx = top_indices[1]
    
    f1_name = feature_columns[top_feature_1_idx].replace('_', ' ')
    f2_name = feature_columns[top_feature_2_idx].replace('_', ' ')
    
    direction_1 = "increases" if shap_vals[top_feature_1_idx] > 0 else "decreases"
    direction_2 = "increases" if shap_vals[top_feature_2_idx] > 0 else "decreases"
    
    st.markdown("### Summary")
    st.write(
        f"This customer is considered **{risk_label}** risk. "
        f"The primary factor is their **{f1_name}** which {direction_1} their risk, "
        f"followed by their **{f2_name}** which {direction_2} their risk."
    )

with col2:
    st.markdown("### Top Factors Driving Prediction")
    # SHAP waterfall plot
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # SHAP waterfall requires a single Explanation object
    # We pass max_display=6 to show top 5 + "other"
    shap.plots.waterfall(shap_values[0], max_display=6, show=False)
    
    st.pyplot(fig, bbox_inches='tight')
