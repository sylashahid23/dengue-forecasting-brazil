import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================================
# Configuration
# ==========================================================

DATA_FILE = "Brazil_Final_Dataset.csv"

LSTM_FILE = "results/lstm_predictions.csv"
TRANSFORMER_FILE = "results/transformer_predictions.csv"
DLINEAR_FILE = "results/dlinear_predictions.csv"

LSTM_TRANSFORMER_STACKING_FILE = (
    "results/lstm_transformer_stacking_predictions.csv"
)

LSTM_DLINEAR_STACKING_FILE = (
    "results/lstm_dlinear_stacking_predictions.csv"
)

TRANSFORMER_DLINEAR_STACKING_FILE = (
    "results/transformer_dlinear_stacking_predictions.csv"
)

ALL_THREE_STACKING_FILE = (
    "results/all_three_stacking_predictions.csv"
)

OUTPUT_DIR = Path("results/additional_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Colours
# ==========================================================

MODEL_COLORS = {
    "LSTM": "#A8D8F0",
    "Transformer": "#F7D774",
    "DLinear": "#F8BBD9"
}

STATE_COLORS = {
    "AM": "#A8B86B",
    "BA": "#F6D36F",
    "GO": "#F2A8CB",
    "RS": "#6FA8DC",
    "SP": "#D96FA3"
}


# ==========================================================
# Load dataset
# ==========================================================

df = pd.read_csv(DATA_FILE)

df["year_month"] = pd.to_datetime(df["year_month"])

df = df.sort_values(
    ["state", "year_month"]
).reset_index(drop=True)

states = ["AM", "BA", "GO", "RS", "SP"]

print("Dataset loaded.")
print(df.shape)


# ==========================================================
# 1. Mean climate variables by state
# ==========================================================

climate_summary = df.groupby("state")[
    ["temperature", "humidity", "precipitation"]
].mean()

ax = climate_summary.plot(
    kind="bar",
    figsize=(10, 6),
    color=[
        STATE_COLORS["AM"],
        STATE_COLORS["BA"],
        STATE_COLORS["GO"]
    ]
)

ax.set_xlabel("State")
ax.set_ylabel("Mean value")
ax.set_title(
    "Mean climate variables by state, 2015–2024"
)

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "climate_variables_by_state.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Created: climate_variables_by_state.png")


# ==========================================================
# 2. Monthly dengue cases by state
# ==========================================================

plt.figure(figsize=(12, 6))

for state in states:

    state_data = df[
        df["state"] == state
    ].sort_values("year_month")

    plt.plot(
        state_data["year_month"],
        state_data["dengue_cases"],
        label=state,
        color=STATE_COLORS[state]
    )

plt.xlabel("Year")
plt.ylabel("Dengue cases")

plt.title(
    "Monthly dengue cases by state, 2015–2024"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "monthly_dengue_cases_by_state.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Created: monthly_dengue_cases_by_state.png")


# ==========================================================
# 3. High- and low-dengue periods
#
# Threshold is calculated separately for each state
# using the full 2015–2024 study period.
# ==========================================================

df["high_threshold"] = df.groupby(
    "state"
)["dengue_cases"].transform(
    lambda x: x.quantile(0.75)
)

df["period"] = np.where(
    df["dengue_cases"] >= df["high_threshold"],
    "High dengue",
    "Low dengue"
)

period_summary = df.groupby(
    ["state", "period"]
)["dengue_cases"].mean().unstack()

ax = period_summary.plot(
    kind="bar",
    figsize=(10, 6),
    color=[
        "#F8BBD9",
        "#A8D8F0"
    ]
)

ax.set_xlabel("State")
ax.set_ylabel("Mean dengue cases")

ax.set_title(
    "Mean dengue cases during high- and low-dengue periods"
)

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "high_low_dengue_periods.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Created: high_low_dengue_periods.png")


# ==========================================================
# 4. Seasonal dengue pattern
# ==========================================================

df["month"] = df["year_month"].dt.month

seasonal = df.groupby(
    ["state", "month"]
)["dengue_cases"].mean().reset_index()

plt.figure(figsize=(11, 6))

for state in states:

    state_data = seasonal[
        seasonal["state"] == state
    ]

    plt.plot(
        state_data["month"],
        state_data["dengue_cases"],
        marker="o",
        label=state,
        color=STATE_COLORS[state]
    )

plt.xlabel("Month")
plt.ylabel("Mean dengue cases")

plt.title(
    "Average monthly dengue pattern by state, 2015–2024"
)

plt.xticks(range(1, 13))

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "seasonal_dengue_pattern.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Created: seasonal_dengue_pattern.png")


# ==========================================================
# Load model predictions
# ==========================================================

model_files = {
    "LSTM": LSTM_FILE,
    "Transformer": TRANSFORMER_FILE,
    "DLinear": DLINEAR_FILE
}

predictions = {}

for model, file in model_files.items():

    predictions[model] = pd.read_csv(file)

    predictions[model]["year_month"] = pd.to_datetime(
        predictions[model]["year_month"]
    )

    predictions[model] = predictions[model].sort_values(
        ["state", "year_month"]
    ).reset_index(drop=True)

    print(f"Loaded {model} predictions.")


# ==========================================================
# Load stacking predictions
# ==========================================================

stacking_files = {
    "LSTM + Transformer": LSTM_TRANSFORMER_STACKING_FILE,
    "LSTM + DLinear": LSTM_DLINEAR_STACKING_FILE,
    "Transformer + DLinear": TRANSFORMER_DLINEAR_STACKING_FILE,
    "LSTM + Transformer + DLinear": ALL_THREE_STACKING_FILE
}

stacking_predictions = {}

for model, file in stacking_files.items():

    stacking_predictions[model] = pd.read_csv(file)

    stacking_predictions[model]["year_month"] = pd.to_datetime(
        stacking_predictions[model]["year_month"]
    )

    stacking_predictions[model] = stacking_predictions[model].sort_values(
        ["state", "year_month"]
    ).reset_index(drop=True)

    print(f"Loaded {model} stacking predictions.")


# ==========================================================
# 5. High-dengue prediction analysis
#
# IMPORTANT:
# Thresholds are calculated using training data only
# (2015–2021), then applied to the test period (2023–2024).
# ==========================================================

train_df = df[
    df["year_month"] < "2022-01-01"
].copy()

test_actual = df[
    df["year_month"] >= "2023-01-01"
].copy()

training_thresholds = train_df.groupby(
    "state"
)["dengue_cases"].quantile(0.75)

print("\nTraining-period high-dengue thresholds:")

for state in states:
    print(
        f"{state}: "
        f"{training_thresholds[state]:.2f}"
    )


# ==========================================================
# 5a. High-dengue predictions - base models
# ==========================================================

for state in states:

    state_actual = test_actual[
        test_actual["state"] == state
    ].copy()

    threshold = training_thresholds[state]

    high_period = state_actual[
        state_actual["dengue_cases"] >= threshold
    ].copy()

    if high_period.empty:
        print(
            f"No high-dengue test months found for {state}."
        )
        continue

    plt.figure(figsize=(12, 6))

    # Plot all test-period actual observations
    plt.plot(
        state_actual["year_month"],
        state_actual["dengue_cases"],
        label="Actual",
        color="#7A7A7A",
        linewidth=2
    )

    # Highlight high-dengue actual observations
    plt.scatter(
        high_period["year_month"],
        high_period["dengue_cases"],
        label="High dengue",
        color="#D96FA3",
        s=45,
        zorder=5
    )

    for model, pred in predictions.items():

        state_pred = pred[
            (pred["state"] == state) &
            (pred["year_month"].isin(
                high_period["year_month"]
            ))
        ]

        if not state_pred.empty:

            plt.scatter(
                state_pred["year_month"],
                state_pred["predicted"],
                label=model,
                color=MODEL_COLORS[model],
                s=45
            )

    plt.xlabel("Month")
    plt.ylabel("Dengue cases")

    plt.title(
        f"Actual and predicted dengue cases during high-dengue months - {state}"
    )

    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"high_dengue_predictions_{state}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("Created high-dengue prediction graphs.")


# ==========================================================
# 5b. High-dengue predictions - stacking models
# ==========================================================

for state in states:

    state_actual = test_actual[
        test_actual["state"] == state
    ].copy()

    threshold = training_thresholds[state]

    high_period = state_actual[
        state_actual["dengue_cases"] >= threshold
    ].copy()

    if high_period.empty:
        continue

    plt.figure(figsize=(12, 6))

    plt.plot(
        state_actual["year_month"],
        state_actual["dengue_cases"],
        label="Actual",
        color="#7A7A7A",
        linewidth=2
    )

    plt.scatter(
        high_period["year_month"],
        high_period["dengue_cases"],
        label="High dengue",
        color="#D96FA3",
        s=45,
        zorder=5
    )

    for model, pred in stacking_predictions.items():

        state_pred = pred[
            (pred["state"] == state) &
            (pred["year_month"].isin(
                high_period["year_month"]
            ))
        ]

        if not state_pred.empty:

            plt.scatter(
                state_pred["year_month"],
                state_pred["stacking_predicted"],
                label=model,
                s=45
            )

    plt.xlabel("Month")
    plt.ylabel("Dengue cases")

    plt.title(
        f"Stacking predictions during high-dengue months - {state}"
    )

    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"high_dengue_stacking_predictions_{state}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("Created stacking high-dengue prediction graphs.")


# ==========================================================
# 6. Peak magnitude comparison
# ==========================================================

peak_results = []

for state in states:

    actual_state = test_actual[
        test_actual["state"] == state
    ]

    if actual_state.empty:
        continue

    actual_peak = actual_state.loc[
        actual_state["dengue_cases"].idxmax()
    ]

    actual_peak_value = actual_peak["dengue_cases"]
    actual_peak_date = actual_peak["year_month"]

    for model, pred in predictions.items():

        pred_state = pred[
            pred["state"] == state
        ]

        if pred_state.empty:
            continue

        predicted_peak = pred_state.loc[
            pred_state["predicted"].idxmax()
        ]

        peak_results.append({
            "state": state,
            "model": model,
            "actual_peak": actual_peak_value,
            "predicted_peak": predicted_peak["predicted"],
            "actual_peak_month": actual_peak_date,
            "predicted_peak_month":
                predicted_peak["year_month"]
        })

peak_df = pd.DataFrame(peak_results)

peak_df["peak_magnitude_error"] = (
    peak_df["predicted_peak"]
    - peak_df["actual_peak"]
)

peak_df.to_csv(
    OUTPUT_DIR / "peak_magnitude_results.csv",
    index=False
)

print("Created: peak_magnitude_results.csv")


# ==========================================================
# 7. Prediction bias
# ==========================================================

bias_results = []

for model, pred in predictions.items():

    temp = pred.copy()

    temp["bias"] = (
        temp["predicted"]
        - temp["actual"]
    )

    bias_summary = temp.groupby(
        "state"
    )["bias"].mean().reset_index()

    bias_summary["model"] = model

    bias_results.append(bias_summary)

bias_df = pd.concat(
    bias_results,
    ignore_index=True
)

bias_df.to_csv(
    OUTPUT_DIR / "prediction_bias.csv",
    index=False
)

print("Created: prediction_bias.csv")


# ==========================================================
# 8. Actual-predicted correlation
# ==========================================================

correlation_results = []

for model, pred in predictions.items():

    for state in states:

        state_pred = pred[
            pred["state"] == state
        ]

        if len(state_pred) < 2:
            continue

        correlation = (
            state_pred["actual"]
            .corr(state_pred["predicted"])
        )

        correlation_results.append({
            "state": state,
            "model": model,
            "correlation": correlation
        })

correlation_df = pd.DataFrame(
    correlation_results
)

correlation_df.to_csv(
    OUTPUT_DIR / "actual_predicted_correlation.csv",
    index=False
)

print("Created: actual_predicted_correlation.csv")


# ==========================================================
# 9. Peak timing error
# ==========================================================

timing_results = []

for state in states:

    actual_state = test_actual[
        test_actual["state"] == state
    ].copy()

    if actual_state.empty:
        continue

    actual_peak_date = actual_state.loc[
        actual_state["dengue_cases"].idxmax(),
        "year_month"
    ]

    for model, pred in predictions.items():

        state_pred = pred[
            pred["state"] == state
        ].copy()

        if state_pred.empty:
            continue

        predicted_peak_date = state_pred.loc[
            state_pred["predicted"].idxmax(),
            "year_month"
        ]

        timing_error = (
            predicted_peak_date
            - actual_peak_date
        ).days / 30.4375

        timing_results.append({
            "state": state,
            "model": model,
            "actual_peak_month": actual_peak_date,
            "predicted_peak_month":
                predicted_peak_date,
            "peak_timing_error_months":
                timing_error
        })

timing_df = pd.DataFrame(
    timing_results
)

timing_df.to_csv(
    OUTPUT_DIR / "peak_timing_error.csv",
    index=False
)

print("Created: peak_timing_error.csv")


# ==========================================================
# 10. Summary
# ==========================================================

print("\nAdditional analysis completed.")

print(
    f"Files saved to: {OUTPUT_DIR}"
)
# ==========================================================
# 11. Pairwise Equal-Weight Ensemble Performance
# ==========================================================

PAIRWISE_FILE = "results/pairwise_ensemble_metrics.csv"

pairwise_df = pd.read_csv(PAIRWISE_FILE)

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(pairwise_df))
width = 0.35

ax.bar(
    x - width / 2,
    pairwise_df["RMSE"],
    width,
    label="RMSE",
    color="#A8D8F0"
)

ax.bar(
    x + width / 2,
    pairwise_df["MAE"],
    width,
    label="MAE",
    color="#F7D774"
)

ax.set_xlabel("Pairwise Ensemble")
ax.set_ylabel("Error")

ax.set_title(
    "RMSE and MAE Comparison of Pairwise Equal-Weight Ensemble Models"
)

ax.set_xticks(x)
ax.set_xticklabels(
    [
        "LSTM + Transformer",
        "LSTM + DLinear",
        "Transformer + DLinear"
    ],
    rotation=20,
    ha="right"
)

ax.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "pairwise_ensemble_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Created: pairwise_ensemble_performance.png")