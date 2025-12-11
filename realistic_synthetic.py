import joblib
import pandas as pd

# Load saved model and scaler
rf = joblib.load("rf_fraud_model.joblib")
scaler = joblib.load("scaler.joblib")

def predict(amount, oldbalanceOrg, newbalanceOrig, hour,
            isForeign, isHighRiskCountry, device_age_days, num_prev_txn_1d):

    # Create dataframe
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

    # Scale numerical columns
    num_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "device_age_days", "num_prev_txn_1d"]
    df[num_cols] = scaler.transform(df[num_cols])

    # -------- Threshold-based prediction --------
    threshold = 0.1325   # <-- balanced threshold we selected

    prob = rf.predict_proba(df)[:, 1][0]
    pred = 1 if prob >= threshold else 0

    print("\n--- FRAUD PREDICTION RESULT ---")
    print(f"Fraud Probability: {prob:.4f}")
    print(f"Threshold Used: {threshold}")
    print(f"Prediction: {'FRAUD' if pred == 1 else 'NOT FRAUD'}")
    print("--------------------------------\n")

# Example 1 — Risky transaction (should lean fraud)
predict(900, 200, 0, 2, 1, 0, 10, 0)

# Example 2 — Safe transaction (should not be fraud)
predict(50, 1000, 950, 14, 0, 0, 400, 0)