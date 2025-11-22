# Vector Ingestion Service

This service ingests schema and example data, generates embeddings using sentence-transformers, and stores them in a local FAISS index.

## How it works
- Reads schema and example data (JSON/CSV/other)
- Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- Stores vectors in FAISS index (local disk)

## Usage
- Build/run with Docker for easy local deployment
- Exposes CLI or REST endpoints for ingestion

## Requirements
- Python 3.8+
- sentence-transformers
- faiss-cpu
- fastapi (if REST API)
- uvicorn (if REST API)

## Preprocessing Spider Dataset
First, preprocess the Spider dataset into the required format:
```sh
python preprocess_spider.py --tables /path/to/spider/tables.json --dataset /path/to/spider/train.json --output spider_processed.json
```

Options:
- `--schemas-only`: Only process schemas, skip example question-SQL pairs

## Example CLI usage
```sh
python ingest.py --input spider_processed.json --index faiss.index
```

This will create:
- `faiss.index`: FAISS vector index
- `embeddings.npy`: Raw embeddings
- `metadata.json`: Metadata for each entry

## Example REST API usage
```sh
uvicorn ingest:app --reload
# POST /ingest with data
```
