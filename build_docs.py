import pandas as pd

# 1. Load your CSV
df = pd.read_csv("shl_assessments_final.csv")

# 2. Split 'test_type' into Python lists
df["test_type"] = df["test_type"].str.split(",")

# 3. Build a single 'doc' column for embedding
def make_doc(row):
    parts = [
        row["name"],
        row["description"],
        *row["test_type"],        # expand list of test types
        row["job_levels"],
        row["languages"]
    ]
    # join non-null parts with periods
    return ". ".join([str(p).strip() for p in parts if pd.notna(p)])

df["doc"] = df.apply(make_doc, axis=1)

# 4. Quick sanity check
print(df[["name", "test_type", "doc"]].head())
