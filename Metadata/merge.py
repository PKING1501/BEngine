import pandas as pd
import glob

# ---- Load all CSV files ----
files = sorted(glob.glob("*.csv"), key=lambda x: int(x.split(".")[0]))

dfs = [pd.read_csv(f) for f in files]

# ---- Find common columns across all files ----
common_cols = set(dfs[0].columns)
for df in dfs[1:]:
    common_cols &= set(df.columns)

common_cols = list(common_cols)
print("Common columns:", common_cols)

# Remove unwanted columns
exclude_cols = ['% Change', 'Revenue']
common_cols = [col for col in common_cols if col not in exclude_cols]

# ---- Keep only common columns ----
dfs = [df[common_cols] for df in dfs]

# ---- Merge vertically ----
merged_df = pd.concat(dfs, ignore_index=True)

# ---- Rebuild "No." column ----
if "No." in merged_df.columns:
    merged_df["No."] = range(1, len(merged_df) + 1)

# ---- Optional: sort columns (keep No. first if exists) ----
if "No." in merged_df.columns:
    cols = ["No."] + [c for c in merged_df.columns if c != "No."]
    merged_df = merged_df[cols]

# ---- Save output ----
merged_df.to_csv("merged.csv", index=False)

print("Merged CSV saved as merged.csv")