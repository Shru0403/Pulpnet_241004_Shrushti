import json
import torch
from sentence_transformers import SentenceTransformer

# load dataset
with open("dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# load embedder
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# generate and save embeddings
print("Generating embeddings")
embeddings = embedder.encode(dataset, convert_to_tensor=True)
torch.save(embeddings, "chunk_embeddings.pt")
print("Embeddings saved")