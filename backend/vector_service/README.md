# Vector Ingestion Service

This service ingests Spider dataset schema and example data, generates embeddings using sentence-transformers, and stores them in a local FAISS index for semantic search.

## What Gets Stored in FAISS

The preprocessing creates **9,535 entries** from the Spider dataset:

1. **Schema Entries (876 entries)**
   - One entry per table from 166 databases
   - Contains: table name, columns with types, primary keys, foreign keys
   - Format: `Database: {db_id}\nTable: {table_name}\nColumns: col1 (type) [PRIMARY KEY], col2 (type) | Foreign Keys: col -> other_table.col`

2. **Example Entries (8,659 entries)**
   - Question-SQL pairs from training data (train_spider.json + train_others.json)
   - Contains: database name, natural language question, SQL query
   - Format: `Database: {db_id}\nQuestion: {question}\nSQL: {sql}`

This allows the system to:
- Find relevant schemas for new questions
- Retrieve similar example queries
- Get complete schema information with relationships

## Setup

### 1. Create virtual environment
```sh
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```sh
pip install -r requirements.txt
```

### 3. Preprocess Spider Dataset
The Spider dataset is in `input/spider_data/`. Process it with:

```sh
python preprocess_spider.py
```

This will:
- Load schemas from `tables.json`
- Load training examples from `train_spider.json` and `train_others.json`
- Create `output/spider_processed.json` with 9,535 entries

**Options:**
- `--spider-dir`: Path to Spider dataset directory (default: `input/spider_data`)
- `--database-dir`: Path to Spider database folder (default: `spider_dir/database`)
- `--output`: Output file path (default: `output/spider_processed.json`)
- `--schemas-only`: Only process schemas, skip examples
- `--examples-only`: Only process examples, skip schemas
- `--no-train-others`: Skip train_others.json

**Example with custom Spider directory:**
```sh
python preprocess_spider.py --spider-dir /path/to/spider_data
```

### 4. Generate FAISS Index
Create the vector index from preprocessed data:

```sh
python ingest.py --input output/spider_processed.json
```

By default, this will create the output files in the same directory as the input file (`output/`).

You can also specify a custom output directory:
```sh
python ingest.py --input output/spider_processed.json --output-dir output
```

This creates:
- `output/faiss.index`: FAISS vector index (9,535 vectors)
- `output/embeddings.npy`: Raw embeddings
- `output/metadata.json`: Metadata for each entry

## Files Structure

```
vector_service/
├── input/
│   └── spider_data/           # Spider dataset
│       ├── tables.json        # Database schemas
│       ├── train_spider.json  # Training examples
│       ├── train_others.json  # Additional training examples
│       └── database/          # SQLite database files
├── output/
│   ├── spider_processed.json  # Preprocessed data (9,535 entries)
│   ├── faiss.index            # Generated vector index
│   ├── embeddings.npy         # Generated embeddings
│   └── metadata.json          # Generated metadata
├── preprocess_spider.py       # Preprocessing script
└── ingest.py                  # FAISS ingestion script
```

## Usage

After setup, the FAISS index can be queried to:
1. Find relevant schemas for a user question
2. Retrieve similar example queries
3. Get complete schema information for SQL generation
