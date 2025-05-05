import faiss
import numpy as np
import pickle

embeddings = np.load("shl_embeddings.npy")          
with open("shl_meta.pkl", "rb") as f:
    df_meta = pickle.load(f)

dim = embeddings.shape[1]    
index = faiss.IndexFlatL2(dim)
index.add(embeddings)         

faiss.write_index(index, "shl_index.faiss")

print(f"Built FAISS index with {index.ntotal} vectors and saved to shl_index.faiss")
