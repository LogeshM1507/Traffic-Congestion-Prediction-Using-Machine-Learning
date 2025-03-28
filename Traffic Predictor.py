import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib

# Load dataset
file_path = "C:\ML_No_Imports\GTFS_Data.csv"
df = pd.read_csv(file_path)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df["arrival_time"] = pd.to_datetime(df["arrival_time"], errors="coerce")
df["arrival_time"] = df["arrival_time"].fillna(df["arrival_time"].median()).astype(int)

# Data Pre Processing
for col in ["speed", "Number_of_trips"]:
    df[col].fillna(df[col].median(), inplace=True)

# Dropping SRI column, as it is directly connected to Degree_of_congestion
df.drop(columns=["SRI"], inplace=True)

# Encode categorical
label_encoders = {}
for col in ["trip_id", "Degree_of_congestion"]:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

#features and target
X = df.drop(columns=["Degree_of_congestion"])
y = df["Degree_of_congestion"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.4f}")
print(classification_report(y_test, y_pred))

# Feature Importance
importances = rf_model.feature_importances_
plt.figure(figsize=(10, 5))
plt.barh(X.columns, importances, color="skyblue")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.show()

for feature, importance in zip(X.columns, importances):
    print(f"{feature}: {importance:.4f}")
joblib.dump(rf_model, "congestion_model.pkl")
