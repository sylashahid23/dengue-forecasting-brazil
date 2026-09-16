import pandas as pd

# Load datasets
dengue = pd.read_csv("State_Dengue_Monthly.csv")
climate = pd.read_csv("Climate_Monthly.csv")

print("Dengue shape:", dengue.shape)
print("Climate shape:", climate.shape)

# Merge on state and month
merged = pd.merge(
    dengue,
    climate,
    on=["state", "year_month"],
    how="inner"
)

print("Merged shape:", merged.shape)

# Sort for modelling
merged = merged.sort_values(
    ["state", "year_month"]
)

# Save
merged.to_csv(
    "Brazil_Climate_Dengue_Monthly.csv",
    index=False
)

print("\nFirst rows:")
print(merged.head())

print("\nMissing values:")
print(merged.isna().sum())