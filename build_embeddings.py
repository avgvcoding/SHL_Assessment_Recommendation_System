import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# 1. Load the cleaned DataFrame
df = pd.read_csv("shl_assessments_final.csv")
df["test_type"] = df["test_type"].str.split(",")
# recreate 'doc' if not already in CSV
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

# 2. Load the SBERT model
print("Loading model all-mpnet-base-v2...")
model = SentenceTransformer("all-mpnet-base-v2")

# 3. Compute embeddings
print("Computing embeddings for each document (this may take a few minutes)...")
embeddings = model.encode(df["doc"].tolist(), show_progress_bar=True, convert_to_numpy=True)

# 4. Save embeddings and metadata for FAISS
#   - embeddings.npy for the vectors
#   - meta.pkl for the metadata columns
np.save("shl_embeddings.npy", embeddings)
df_meta = df[[
  "name", "url", "remote_support", 
  "adaptive_support", "test_type", 
  "duration", "description"
]]

with open("shl_meta.pkl", "wb") as f:
    pickle.dump(df_meta, f)

print("Embeddings saved to shl_embeddings.npy and metadata to shl_meta.pkl")
