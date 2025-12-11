# app.py - Streamlit demo for Fraud Detection
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# load model + scaler
rf = joblib.load("rf_fraud_model.joblib")
scaler = joblib.load("scaler.joblib")

st.set_page_config(page_title="Fraud Detector", layout="centered")
st.title("Fraud Transaction Detector — Demo")

st.markdown("""
Adjust the inputs below and click **Predict**.  
Threshold used in demo: **0.1325** (balanced).
""")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Amount", min_value=0.0, value=100.0, step=1.0)
    oldbalanceOrg = st.number_input("Old Balance (origin)", min_value=0.0, value=500.0, step=1.0)
    newbalanceOrig = st.number_input("New Balance (after tx)", min_value=0.0, value=max(0.0, oldbalanceOrg - amount), step=1.0)
    hour = st.slider("Hour of day", 0, 23, 12)

with col2:
    isForeign = st.selectbox("Is Foreign Transaction?", options=[0,1], index=0)
    isHighRiskCountry = st.selectbox("High Risk Country?", options=[0,1], index=0)
    device_age_days = st.number_input("Device age (days)", min_value=0.0, value=365.0, step=1.0)
    num_prev_txn_1d = st.number_input("Previous txns in 1 day", min_value=0, value=0, step=1)

if st.button("Predict"):
    df = pd.DataFrame([{
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "hour": hour,
        "isForeign": isForeign,
        "isHighRiskCountry": isHighRiskCountry,
        "device_age_days": device_age_days,
        "num_prev_txn_1d": num_prev_txn_1d
    }])
    num_cols = ["amount","oldbalanceOrg","newbalanceOrig","device_age_days","num_prev_txn_1d"]
    df[num_cols] = scaler.transform(df[num_cols])
    prob = rf.predict_proba(df)[:,1][0]
    threshold = 0.1325  # chosen balanced threshold
    pred = "FRAUD" if prob >= threshold else "NOT FRAUD"

    st.metric("Fraud Probability", f"{prob:.4f}")
    st.write("Threshold used:", threshold)
    if pred == "FRAUD":
        st.error("PREDICTION: FRAUD")
    else:
        st.success("PREDICTION: NOT FRAUD")

st.write("---")
st.write("Model note: RandomForest trained on synthetic data; threshold tuned to balance recall.")