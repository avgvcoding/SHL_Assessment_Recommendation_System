import pandas as pd

df = pd.read_csv("shl_assessments_final.csv")

df["test_type"] = df["test_type"].str.split(",")

def make_doc(row):
    parts = [
        row["name"],
        row["description"],
        *row["test_type"],        
        row["job_levels"],
        row["languages"]
    ]
    return ". ".join([str(p).strip() for p in parts if pd.notna(p)])

df["doc"] = df.apply(make_doc, axis=1)

print(df[["name", "test_type", "doc"]].head())
