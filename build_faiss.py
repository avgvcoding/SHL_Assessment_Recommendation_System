import faiss
import numpy as np
import pickle

# 1. Load your embeddings and metadata
embeddings = np.load("shl_embeddings.npy")           # shape: (542, 768)
with open("shl_meta.pkl", "rb") as f:
    df_meta = pickle.load(f)

# 2. Build a Flat L2 index (simple, exact nearest neighbors)
dim = embeddings.shape[1]     # 768 for MPNet
index = faiss.IndexFlatL2(dim)
index.add(embeddings)         # add all vectors

# 3. Save the index to disk
faiss.write_index(index, "shl_index.faiss")

print(f"Built FAISS index with {index.ntotal} vectors and saved to shl_index.faiss")
