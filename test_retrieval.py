import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# 1. Load model, index, and metadata
model = SentenceTransformer("all-mpnet-base-v2")
index = faiss.read_index("shl_index.faiss")
with open("shl_meta.pkl", "rb") as f:
    df_meta = pickle.load(f)

# 2. Define a sample query
query = "Entry-level analyst role: cognitive ability and personality test under 45 minutes"

# 3. Embed the query
q_vec = model.encode([query], convert_to_numpy=True)

# 4. Search the FAISS index (top 5)
D, I = index.search(q_vec, k=5)   # distances D and indices I

# 5. Print out the results
print(f"Query: {query}\n")
for rank, idx in enumerate(I[0], start=1):
    meta = df_meta.iloc[idx]
    print(f"{rank}. {meta['name']} ({meta['duration']} mins)")
    print(f"   URL: {meta['url']}")
    print(f"   Remote: {meta['remote_support']}, Adaptive: {meta['adaptive_support']}")
    print(f"   Type: {meta['test_type']}\n")
