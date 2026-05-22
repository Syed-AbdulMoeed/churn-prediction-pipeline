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

    #--- Preprocessing -------------------------------------------
    df_processed = df.copy()
    
    # Drop customerID if present
    if 'customerID' in df_processed.columns:
        df_processed.drop(columns=['customerID'], inplace=True)

    # One hot encoding Categoricals
    df_processed = pd.get_dummies( df, columns = ['Contract', 'PaymentMethod', 'InternetService'], dtype=int )    

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
    df_processed.reindex(columns=model_cols, fill_value=0)

    #--- Predictions ----------------------------------------------
    churn_prob = model.predict_proba(df_processed)[:, 1]
    churn_pred = model.predict(df_processed)


else:
    st.info("Please upload a CSV file to get started.")

