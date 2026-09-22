import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============================================================
# File paths
# ============================================================

BASE_PATH = "../"

DATA_FILE = BASE_PATH + "Brazil_Final_Dataset.csv"

LSTM_FILE = BASE_PATH + "results/lstm_predictions.csv"
TRANSFORMER_FILE = BASE_PATH + "results/transformer_predictions.csv"
DLINEAR_FILE = BASE_PATH + "results/dlinear_predictions.csv"


# ============================================================
# Load data
# ============================================================

data = pd.read_csv(DATA_FILE)

lstm = pd.read_csv(LSTM_FILE)
transformer = pd.read_csv(TRANSFORMER_FILE)
dlinear = pd.read_csv(DLINEAR_FILE)


# ============================================================
# Identify date column
# ============================================================

def find_date_column(df):
    possible = ["year_month", "date", "month", "Year_Month"]

    for col in possible:
        if col in df.columns:
            return col

    raise ValueError(
        f"No date column found. Available columns: {list(df.columns)}"
    )


# ============================================================
# Identify state column
# ============================================================

def find_state_column(df):
    possible = ["state", "State"]

    for col in possible:
        if col in df.columns:
            return col

    raise ValueError(
        f"No state column found. Available columns: {list(df.columns)}"
    )


# ============================================================
# Identify actual dengue column in final dataset
# ============================================================

def find_dengue_column(df):
    possible = [
        "dengue_cases",
        "Dengue_Cases",
        "cases",
        "Cases",
        "dengue"
    ]

    for col in possible:
        if col in df.columns:
            return col

    raise ValueError(
        f"No dengue column found. Available columns: {list(df.columns)}"
    )


# ============================================================
# Prepare prediction data
# ============================================================

def prepare_predictions(df, model_name):

    state_col = find_state_column(df)
    date_col = find_date_column(df)

    if "actual" not in df.columns:
        raise ValueError(
            f"'actual' column not found in {model_name} predictions."
        )

    if "predicted" not in df.columns:
        raise ValueError(
            f"'predicted' column not found in {model_name} predictions."
        )

    result = df.copy()

    result["state"] = result[state_col].astype(str).str.upper()
    result["year_month"] = pd.to_datetime(result[date_col])

    result = result[
        ["state", "year_month", "actual", "predicted"]
    ]

    result["model"] = model_name

    return result


lstm = prepare_predictions(lstm, "LSTM")
transformer = prepare_predictions(transformer, "Transformer")
dlinear = prepare_predictions(dlinear, "DLinear")


# ============================================================
# Combine predictions
# ============================================================

predictions = pd.concat(
    [lstm, transformer, dlinear],
    ignore_index=True
)


# ============================================================
# Remove the two RS observations
# ============================================================

excluded_dates = pd.to_datetime([
    "2024-08-01",
    "2024-10-01"
])

sensitivity_predictions = predictions[
    ~(
        (predictions["state"] == "RS") &
        (predictions["year_month"].isin(excluded_dates))
    )
].copy()


# ============================================================
# Calculate metrics
# ============================================================

results = []

for model in ["LSTM", "Transformer", "DLinear"]:

    original = predictions[
        predictions["model"] == model
    ]

    sensitivity = sensitivity_predictions[
        sensitivity_predictions["model"] == model
    ]

    original_rmse = np.sqrt(
        mean_squared_error(
            original["actual"],
            original["predicted"]
        )
    )

    original_mae = mean_absolute_error(
        original["actual"],
        original["predicted"]
    )

    sensitivity_rmse = np.sqrt(
        mean_squared_error(
            sensitivity["actual"],
            sensitivity["predicted"]
        )
    )

    sensitivity_mae = mean_absolute_error(
        sensitivity["actual"],
        sensitivity["predicted"]
    )

    results.append({
        "Model": model,
        "Original RMSE": original_rmse,
        "Excluding observations RMSE": sensitivity_rmse,
        "Original MAE": original_mae,
        "Excluding observations MAE": sensitivity_mae
    })


# ============================================================
# Results table
# ============================================================

results_df = pd.DataFrame(results)

print("\n==============================================")
print("SENSITIVITY ANALYSIS")
print("==============================================")
print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:,.2f}"
    )
)


# ============================================================
# Save results
# ============================================================

output_file = BASE_PATH + "results/sensitivity_analysis.csv"

results_df.to_csv(
    output_file,
    index=False
)

print("\nSaved to:")
print(output_file)