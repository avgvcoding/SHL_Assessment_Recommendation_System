import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")
index = faiss.read_index("shl_index.faiss")
with open("shl_meta.pkl", "rb") as f:
    df_meta = pickle.load(f)

query = "Entry-level analyst role: cognitive ability and personality test under 45 minutes"

q_vec = model.encode([query], convert_to_numpy=True)

D, I = index.search(q_vec, k=5)  

print(f"Query: {query}\n")
for rank, idx in enumerate(I[0], start=1):
    meta = df_meta.iloc[idx]
    print(f"{rank}. {meta['name']} ({meta['duration']} mins)")
    print(f"   URL: {meta['url']}")
    print(f"   Remote: {meta['remote_support']}, Adaptive: {meta['adaptive_support']}")
    print(f"   Type: {meta['test_type']}\n")
