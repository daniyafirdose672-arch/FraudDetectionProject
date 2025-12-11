import joblib
import pandas as pd

# Load saved objects
rf = joblib.load("rf_fraud_model.joblib")
scaler = joblib.load("scaler.joblib")

def predict(amount, oldbalanceOrg, newbalanceOrig, hour,
            isForeign, isHighRiskCountry, device_age_days, num_prev_txn_1d):
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
    threshold = 0.30   # try 0.3 or 0.25 for better recall
prob = rf.predict_proba(df)[:,1][0]
pred = 1 if prob >= threshold else 0
print(f"Pred: {pred}  |  Fraud probability: {prob:.4f}")

# Example call — change values to test different cases
if __name__ == "__main__":
    # Example: risky -> high amount, low balance, foreign
    predict(900, 200, 0, 2, 1, 0, 10, 0)

    # Example: safe -> small amount, big balance
    predict(50, 1000, 950, 14, 0, 0, 400, 0)
    