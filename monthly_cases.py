import pandas as pd
import glob

files = sorted(glob.glob("DENGBR*.csv"))

monthly_counts = {}

for file in files:
    print(f"Processing {file}...")

    df = pd.read_csv(
        file,
        usecols=["DT_NOTIFIC"],
        low_memory=False
    )

    df["DT_NOTIFIC"] = pd.to_datetime(df["DT_NOTIFIC"], errors="coerce")
    df = df.dropna(subset=["DT_NOTIFIC"])

    counts = df["DT_NOTIFIC"].dt.strftime("%Y-%m").value_counts()

    for month, count in counts.items():
        monthly_counts[month] = monthly_counts.get(month, 0) + count

monthly = (
    pd.DataFrame(
        sorted(monthly_counts.items()),
        columns=["year_month", "dengue_cases"]
    )
)

monthly.to_csv("Brazil_Monthly.csv", index=False)

print("\nDone!")
print(monthly.head())
print(monthly.tail())
print(f"\nTotal months: {len(monthly)}")
