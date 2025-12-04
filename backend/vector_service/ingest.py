from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import argparse
import os

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUTPUT_DIR = "output"
INDEX_FILE = "faiss.index"
EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "metadata.json"


def load_data(input_path):
    print(f"\nLoading data from {input_path}...")
    with open(input_path, "r") as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} entries")
    return data


def embed_texts(texts, model):
    print(f"\nGenerating embeddings for {len(texts)} texts...")
    print("This may take a few minutes...")
    return model.encode(texts, show_progress_bar=True)


def save_faiss_index(embeddings, index_path):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return index


def main(input_path, output_dir=None):
    # Determine output directory
    if output_dir is None:
        # Use same directory as input file, or default output dir
        input_dir = os.path.dirname(input_path)
        output_dir = input_dir if input_dir else DEFAULT_OUTPUT_DIR

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Created output directory: {output_dir}")

    # Build output file paths
    index_path = os.path.join(output_dir, INDEX_FILE)
    embeddings_path = os.path.join(output_dir, EMBEDDINGS_FILE)
    metadata_path = os.path.join(output_dir, METADATA_FILE)

    print("\n" + "=" * 60)
    print("FAISS INDEX GENERATION")
    print("=" * 60)
    print(f"Input file: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    # Load data

    data = load_data(input_path)
    texts = [item["text"] for item in data]
    metadata = []
    for item in data:
        meta = item.get("meta", {}) or {}
        meta = dict(meta)
        meta["text"] = item.get("text", "")
        metadata.append(meta)

    # Generate embeddings
    model = SentenceTransformer(MODEL_NAME)
    embeddings = embed_texts(texts, model)
    embeddings = np.array(embeddings).astype("float32")

    # Save outputs
    print(f"\nSaving FAISS index to {index_path}...")
    save_faiss_index(embeddings, index_path)
    print(f"✓ Saved FAISS index")

    print(f"\nSaving embeddings to {embeddings_path}...")
    np.save(embeddings_path, embeddings)
    print(f"✓ Saved embeddings")

    print(f"\nSaving metadata to {metadata_path}...")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata")

    print("\n" + "=" * 60)
    print("✓ INGESTION COMPLETE!")
    print("=" * 60)
    print(f"Processed {len(texts)} items")
    print(f"Output files:")
    print(f"  - {index_path}")
    print(f"  - {embeddings_path}")
    print(f"  - {metadata_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest schema/examples into FAISS index."
    )
    parser.add_argument(
        "--input", required=True, help="Path to input JSON file (list of {text, meta})"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save output files (default: same as input file directory)",
    )
    args = parser.parse_args()
    main(args.input, args.output_dir)
