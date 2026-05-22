import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load('models/best_xgb.pkl')
scaler = joblib.load('models/scaler.pkl')

# Page config
st.set_page_config(page_title="Churn Predictor", layout="wide")

# Title
st.title("Customer Churn Predictor")
st.markdown("Upload a CSV of customer data to predict churn probability.")

# File uploader
uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df)

    # Show predictions
    scaled = scaler.fit_transform(df)
    predictions = model.predict(scaled)
    input, ans = st.columns(2)
    input = st.dataframe(df)
    ans = st.dataframe(predictions)

else:
    st.info("Please upload a CSV file to get started.")

