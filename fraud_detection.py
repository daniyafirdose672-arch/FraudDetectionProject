print("Fraud detection script started...")

# ==============================
# FRAUD TRANSACTION DETECTION - BASIC VERSION
# ==============================

# 1. IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 2. CREATE A FAKE FRAUD DATASET (NO FILE NEEDED)
# ------------------------------------------------
np.random.seed(42)  # so results are same every time

n_samples = 600

# Random transaction amounts (most are small, some are big)
amount = np.random.exponential(scale=80, size=n_samples)

# Random account balances before transaction
oldbalanceOrg = np.random.exponential(scale=200, size=n_samples)

# Simple rule to create "fraud" labels:
# Fraud if amount is high AND old balance is low
is_fraud = ((amount > 200) & (oldbalanceOrg < 150)).astype(int)

# Create DataFrame
data = pd.DataFrame({
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "isFraud": is_fraud
})

print("\nFirst 5 rows of our fake dataset:")
print(data.head())
print("\nClass distribution (0 = not fraud, 1 = fraud):")
print(data["isFraud"].value_counts())

# 3. SPLIT FEATURES & TARGET
# ------------------------------------------------
X = data[["amount", "oldbalanceOrg"]]  # input features
y = data["isFraud"]                    # target label

# 4. TRAIN-TEST SPLIT
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# 5. FEATURE SCALING
# ------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. TRAIN A MODEL (LOGISTIC REGRESSION)
# ------------------------------------------------
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 7. PREDICT ON TEST DATA
# ------------------------------------------------
y_pred = model.predict(X_test_scaled)

# 8. EVALUATE MODEL
# ------------------------------------------------
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# 9. RANDOM FOREST MODEL
# ==============================
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))
# 10. ROC-AUC Curve for Random Forest
# ==============================
from sklearn.metrics import roc_curve, auc

rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, rf_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest ROC Curve")
plt.legend()
plt.show()