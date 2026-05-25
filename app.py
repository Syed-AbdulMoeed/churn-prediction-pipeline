import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt    

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

    #--- Preprocessing -------------------------------------------
    df_processed = df.copy()
    
    # Drop specific columns 
    cols_to_drop = ['customerID', 'Churn']
    df_processed.drop(columns=[col for col in cols_to_drop if col in df_processed.columns], inplace=True)

    # One hot encoding Categoricals
    df_processed = pd.get_dummies( df_processed, columns = ['Contract', 'PaymentMethod', 'InternetService'], dtype=int )    

    # Encoding Binary
    binary_cols = df_processed.select_dtypes(include='object').columns
    
    df_processed[binary_cols] = df_processed[binary_cols].apply(
        lambda col : col.map({
        'Yes': 1, 'No': 0, 'Female': 1, 'Male': 0,
        'No phone service': 0, 'No internet service': 0
        })
    )

    # Scaling Numericals
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df_processed[num_cols] = scaler.transform(df_processed[num_cols])

    # Aligning columns to match the training data exactly
    model_cols = model.get_booster().feature_names
    df_processed = df_processed.reindex(columns=model_cols, fill_value=0)

    #--- Predictions ----------------------------------------------
    churn_prob = model.predict_proba(df_processed)[:, 1]
    churn_pred = model.predict(df_processed)

    #--- Display Results ------------------------------------------
    st.subheader("Predictions")

    # Adding prediction cols 
    results = df.copy()
    results["Churn Probability"] = (churn_prob * 100).round(1).astype(str) + '%'
    results['Prediction'] = ['Will Churn' if p == 1 else 'Will Stay' for p in churn_pred]

    # Showing only the relevant columns
    display_cols = ['Churn Probability', 'Prediction']
    if 'customerID' in df.columns:
        display_cols = ['customerID'] + display_cols

    st.dataframe(results[display_cols], use_container_width=True)

    # Summary metrics

    #--- SHAP Chart ------------------------------------------------
    st.subheader("Top Churn Risk Factors")
    st.markdown("What is driving churn predictions across all uploaded customers?")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(df_processed)

    fig, ax = plt.subplots(figsize=(10, 6))

else:
    st.info("Please upload a CSV file to get started.")

