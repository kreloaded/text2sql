# Text2SQL

Convert natural language questions to SQL queries using AI-powered semantic search and code generation.

## 🚀 Features

- **Natural Language Processing**: Convert plain English questions to SQL queries
- **Semantic Search**: Uses FAISS vector store with sentence transformers for intelligent context retrieval
- **Spider Dataset**: Trained on 166 databases with 9,535+ examples
- **Side-by-Side View**: Modern UI showing question, SQL, schema, and similar examples
- **FastAPI Backend**: High-performance API with automatic documentation
- **Flask Frontend**: Clean, responsive web interface

## 📋 Prerequisites

- Python 3.8 or higher
- Spider dataset (download from [Yale Spider](https://yale-lily.github.io/spider))
- 4GB+ RAM for FAISS index

## 🛠️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/kreloaded/text2sql.git
cd text2sql
```

### 2. Download Spider Dataset

Download the Spider dataset and extract it:
```bash
# Download from https://yale-lily.github.io/spider
# Extract to your preferred location, e.g., ~/Downloads/spider_data
```

### 3. Setup Vector Service

```bash
cd backend/vector_service

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Process Spider dataset
python preprocess_spider.py \
  --spider-dir /path/to/spider/data/directory \
  --database-dir /path/to/spider/data/directory/database

# Generate FAISS index (this may take 5-10 minutes)
python ingest.py --input output/spider_processed.json
```

### 4. Setup Backend

```bash
cd ../  # Move to backend directory

# Use the same virtual environment
source vector_service/venv/bin/activate

# Install backend dependencies (if not already installed)
pip install -r requirements.txt
```

### 5. Setup Frontend

```bash
cd ../frontend

# Create virtual environment for frontend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Running the Application

You'll need to run both the backend and frontend servers.

### Terminal 1: Start Backend Server

```bash
cd backend
source vector_service/venv/bin/activate
PYTHONPATH=. python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Backend will be available at: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/health

### Terminal 2: Start Frontend Server

```bash
cd frontend
source venv/bin/activate
python app.py
```

Frontend will be available at: http://localhost:5001

## 💡 Usage

1. Open your browser and navigate to http://localhost:5001
2. Enter a natural language question (e.g., "How many singers do we have?")
3. Click "Generate SQL"
4. View the results in the side-by-side comparison:
   - **Left Panel**: Your question and generated SQL
   - **Right Panel**: Database schema and similar examples

## 📁 Project Structure

```
text2sql/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── config.py              # Configuration settings
│   ├── requirements.txt       # Backend dependencies
│   ├── models/
│   │   └── schemas.py         # Pydantic models
│   ├── services/
│   │   ├── retriever.py       # FAISS vector search
│   │   └── sql_generator.py   # SQL generation logic
│   └── vector_service/
│       ├── preprocess_spider.py  # Spider dataset processor
│       ├── ingest.py            # FAISS index generator
│       └── output/              # Generated files
│           ├── spider_processed.json
│           ├── faiss.index
│           ├── embeddings.npy
│           └── metadata.json
└── frontend/
    ├── app.py                 # Flask application
    ├── requirements.txt       # Frontend dependencies
    ├── templates/
    │   └── index.html         # Main UI template
    └── static/
        ├── css/
        │   └── style.css      # Styling
        └── js/
            └── main.js        # Frontend logic
```

## 🔧 Configuration

### Backend Configuration (`backend/config.py`)

- `FAISS_INDEX_PATH`: Path to FAISS index file
- `METADATA_PATH`: Path to metadata JSON file
- `EMBEDDING_MODEL`: Sentence transformer model name
- `TOP_K`: Number of similar examples to retrieve

### Environment Variables

You can set these environment variables to override defaults:
```bash
export FAISS_INDEX_PATH=/custom/path/to/faiss.index
export METADATA_PATH=/custom/path/to/metadata.json
```

## 🧪 Testing

### Test Backend API

```bash
cd backend
source vector_service/venv/bin/activate
python test_api.py
```

### Test Vector Service

```bash
cd backend
source vector_service/venv/bin/activate
python debug_payload.py
```

## 📊 Dataset Information

- **Databases**: 166 different database schemas
- **Total Entries**: 9,535 (876 schemas + 8,659 examples)
- **Source**: Spider dataset from Yale NLP
- **Format**: JSON with schema details, foreign keys, primary keys, and SQL examples

## 🐛 Troubleshooting

### FAISS Index Not Found
```bash
# Regenerate the index
cd backend/vector_service
source venv/bin/activate
python ingest.py --input output/spider_processed.json
```

### Port Already in Use
```bash
# Backend (change port)
uvicorn main:app --port 8001

# Frontend (edit app.py to change port)
```

### Module Import Errors
```bash
# Make sure PYTHONPATH is set for backend
cd backend
PYTHONPATH=. python -m uvicorn main:app
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Spider Dataset**: Yale NLP Group
- **FAISS**: Facebook AI Research
- **Sentence Transformers**: UKPLab

## 📧 Contact

For questions or issues, please open an issue on GitHub.