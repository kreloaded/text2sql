from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import argparse
import os

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FILE = "faiss.index"
EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "metadata.json"

def load_data(input_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
    return data

def embed_texts(texts, model):
    return model.encode(texts, show_progress_bar=True)

def save_faiss_index(embeddings, index_path):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return index

def main(input_path, index_path=INDEX_FILE):
    data = load_data(input_path)
    texts = [item['text'] for item in data]
    metadata = [item.get('meta', {}) for item in data]
    model = SentenceTransformer(MODEL_NAME)
    embeddings = embed_texts(texts, model)
    embeddings = np.array(embeddings).astype('float32')
    save_faiss_index(embeddings, index_path)
    np.save(EMBEDDINGS_FILE, embeddings)
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f)
    print(f"Ingested {len(texts)} items into FAISS index at {index_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest schema/examples into FAISS index.")
    parser.add_argument('--input', required=True, help='Path to input JSON file (list of {text, meta})')
    parser.add_argument('--index', default=INDEX_FILE, help='Path to output FAISS index file')
    args = parser.parse_args()
    main(args.input, args.index)
