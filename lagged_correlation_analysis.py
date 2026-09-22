import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "Brazil_Final_Dataset.csv"
OUTPUT_DIR = Path("results/additional_analysis")

STATES = ["AM", "BA", "GO", "RS", "SP"]

CLIMATE_VARIABLES = [
    "temperature",
    "humidity",
    "precipitation"
]

MAX_LAG = 3


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["year_month"] = pd.to_datetime(df["year_month"])

df = df.sort_values(["state", "year_month"]).reset_index(drop=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CALCULATE LAGGED CORRELATIONS
# ============================================================

results = []

for state in STATES:

    state_df = df[df["state"] == state].copy()

    state_df = state_df.sort_values("year_month").reset_index(drop=True)

    for variable in CLIMATE_VARIABLES:

        for lag in range(MAX_LAG + 1):

            # Shift climate variable forward so that:
            # climate at t-lag is compared with dengue at t
            state_df[f"{variable}_lag{lag}"] = (
                state_df[variable].shift(lag)
            )

            temp = state_df[
                ["dengue_cases", f"{variable}_lag{lag}"]
            ].dropna()

            if len(temp) < 3:
                continue

            r, p = pearsonr(
                temp["dengue_cases"],
                temp[f"{variable}_lag{lag}"]
            )

            results.append({
                "state": state,
                "climate_variable": variable,
                "lag_months": lag,
                "correlation": r,
                "p_value": p,
                "n": len(temp)
            })


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

results_df = pd.DataFrame(results)

detailed_file = OUTPUT_DIR / "lagged_correlation_results.csv"

results_df.to_csv(
    detailed_file,
    index=False
)


# ============================================================
# CREATE STATE-LEVEL SUMMARY
# ============================================================

summary_df = results_df.pivot_table(
    index=["state", "climate_variable"],
    columns="lag_months",
    values="correlation"
).reset_index()

summary_df.columns.name = None

summary_df = summary_df.rename(
    columns={
        0: "lag_0",
        1: "lag_1",
        2: "lag_2",
        3: "lag_3"
    }
)

summary_file = OUTPUT_DIR / "lagged_correlation_summary.csv"

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# CREATE P-VALUE SUMMARY
# ============================================================

pvalue_df = results_df.pivot_table(
    index=["state", "climate_variable"],
    columns="lag_months",
    values="p_value"
).reset_index()

pvalue_df.columns.name = None

pvalue_df = pvalue_df.rename(
    columns={
        0: "lag_0_p",
        1: "lag_1_p",
        2: "lag_2_p",
        3: "lag_3_p"
    }
)

pvalue_file = OUTPUT_DIR / "lagged_correlation_pvalues.csv"

pvalue_df.to_csv(
    pvalue_file,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\nLagged correlation analysis completed.")

print("\nState-level correlations:")
print(summary_df.to_string(index=False))

print("\nDetailed results saved to:")
print(detailed_file)

print("\nCorrelation summary saved to:")
print(summary_file)

print("\nP-value summary saved to:")
print(pvalue_file)