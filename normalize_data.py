import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load datasets
train = pd.read_csv("train.csv")
val = pd.read_csv("validation.csv")
test = pd.read_csv("test.csv")

# Create scaler
scaler = StandardScaler()

# Fit ONLY on training data
train["dengue_cases"] = scaler.fit_transform(train[["dengue_cases"]])

# Apply the same scaler to validation and test
val["dengue_cases"] = scaler.transform(val[["dengue_cases"]])
test["dengue_cases"] = scaler.transform(test[["dengue_cases"]])

# Save normalized datasets
train.to_csv("train_scaled.csv", index=False)
val.to_csv("validation_scaled.csv", index=False)
test.to_csv("test_scaled.csv", index=False)

# Save scaler for later (needed to convert predictions back)
joblib.dump(scaler, "scaler.pkl")

print("Normalization complete!")
print(train.head())
