# Text2SQL Backend

FastAPI backend for converting natural language to SQL queries using CodeT5+ and RAG (Retrieval-Augmented Generation).

## Architecture

- **Backend**: FastAPI (local)
- **Vector Store**: FAISS with 7,140 Spider dataset entries
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Model**: CodeT5+ (tzaware/codet5p-spider) running on Google Colab GPU
- **Connection**: ngrok tunnel from Colab to backend

## Prerequisites

- Python 3.10+ (avoid 3.14 due to tokenizer compatibility issues)
- Google Colab account (free tier with T4 GPU)
- ngrok account (free tier)

## Setup

### 1. Create virtual environment
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate  # On Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Preprocess Spider dataset (if not done)
```bash
cd vector_service
python preprocess_spider_hf.py
```

### 4. Generate FAISS index (if not done)
```bash
python ingest.py
```
This creates:
- `vector_service/faiss.index` (7,140 vectors)
- `vector_service/metadata.json`
- `vector_service/embeddings.npy`

### 5. Set up Colab model server
Follow instructions in `colab_model_server.ipynb`:
1. Import the colab notebook
2. Get ngrok auth token from https://dashboard.ngrok.com/get-started/your-authtoken
3. Run all cells to start model server
4. Copy the ngrok URL (e.g., `https://xxx.ngrok-free.app`)

### 6. Configure environment variables
```bash
cp .env.example .env
```

Edit `.env` and set:
```bash
CUSTOM_MODEL_API_URL=https://your-ngrok-url.ngrok-free.app
```

## Running the API

### Start the backend server
```bash
cd backend
source .venv/bin/activate
PYTHONPATH=. python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

**Important**: Keep the Colab cell running while using the backend!

### Alternative: Using Docker (local model only)
```bash
docker build -t text2sql-backend .
docker run -p 8000:8000 text2sql-backend
```
Note: Docker setup runs model locally (slow on CPU). Colab setup recommended.

## API Endpoints

### `POST /generate`
Generate SQL from natural language question.

**Request:**
```json
{
  "question": "Show all singers from USA",
  "db_id": "concert_singer",
  "use_retrieval": true,
  "top_k": 5
}
```

**Response:**
```json
{
  "question": "Show all singers from USA",
  "generated_sql": "SELECT name FROM singers WHERE country = 'USA'",
  "retrieved_context": [],
  "db_id": "concert_singer"
}
```

**Parameters:**
- `question`: Natural language query
- `db_id`: Database identifier
- `use_retrieval`: Enable RAG (retrieves similar examples from FAISS)
- `top_k`: Number of examples to retrieve (default: 5)

### `POST /retrieve`
Retrieve similar examples without generating SQL (for debugging).

**Request:**
```json
{
  "question": "Show all singers from USA",
  "top_k": 3
}
```

### `GET /health`
Health check endpoint. Returns backend and retriever status.

### `GET /`
Root endpoint with welcome message.

## Testing

### Test SQL generation (without retrieval)
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"question": "Show all singers from USA", "db_id": "concert_singer", "use_retrieval": false}'
```

### Test SQL generation (with retrieval)
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the names of all artists?", "db_id": "concert_singer", "use_retrieval": true, "top_k": 3}'
```

### Test retrieval only
```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question": "Show all singers", "top_k": 5}'
```

## Documentation

Interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Troubleshooting

### "Empty reply from server"
- Check if Colab cell is still running
- Verify ngrok URL is correct in `.env`
- Test Colab health: `curl https://your-ngrok-url.ngrok-free.app/health`

### "502 Bad Gateway"
- Colab Flask server not running (run Cell 5 in Colab notebook)
- ngrok tunnel expired (restart Colab cells)

### "Tokenizer not found"
- Uses fallback tokenizer (Salesforce/codet5-base) automatically
- This is expected behavior due to Python 3.14 compatibility

### FAISS index not found
- Run `python vector_service/preprocess_spider_hf.py`
- Then run `python vector_service/ingest.py`

## Project Structure

```
backend/
├── main.py                 # FastAPI application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not in git)
├── .env.example          # Environment template
├── models/
│   └── schemas.py        # Pydantic models
├── services/
│   ├── retriever.py      # FAISS retrieval service
│   └── sql_generator.py  # CodeT5+ inference service
└── vector_service/
    ├── preprocess_spider_hf.py  # Spider dataset preprocessing
    ├── ingest.py                # FAISS index generation
    ├── faiss.index              # Generated FAISS index
    ├── metadata.json            # Vector metadata
    └── embeddings.npy           # Raw embeddings
```

## Notes

- Keep Colab cell running during backend usage
- ngrok free tier URLs expire after inactivity
- FAISS index contains 7,140 entries (140 schemas + 7,000 examples)
- Model uses T4 GPU on Colab (free tier)
