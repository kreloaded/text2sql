# Text2SQL Frontend

Flask-based web interface for the Text2SQL system.

## Features

- 🎨 Modern, responsive chat-style UI
- 🔍 Database selection dropdown (10+ Spider databases)
- ⚡ Real-time SQL generation via FastAPI backend
- 📊 Context display (schema + similar examples)
- 📋 One-click SQL copy to clipboard
- 🚀 Easy deployment

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the frontend server:**
   ```bash
   python app.py
   ```

   The frontend will be available at: http://localhost:5000

3. **Ensure backend is running:**
   Make sure the FastAPI backend is running on http://localhost:8000

## Configuration

Set the backend URL via environment variable (optional):
```bash
export BACKEND_URL=http://localhost:8000
python app.py
```

## Usage

1. Open http://localhost:5000 in your browser
2. (Optional) Select a database from the dropdown
3. Enter your natural language question
4. Click "Generate SQL" or press Ctrl+Enter
5. View the generated SQL, schema, and examples
6. Copy SQL with one click

## Project Structure

```
frontend/
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main UI template
└── static/
    ├── css/
    │   └── style.css     # Styling
    └── js/
        └── main.js       # Client-side logic
```

## API Endpoints

- `GET /` - Main UI
- `POST /api/generate` - Generate SQL (proxies to backend)
- `GET /api/health` - Health check

## Development

Run in debug mode (auto-reload on changes):
```bash
python app.py
```

## Example Queries

Try these with the **concert_singer** database:
- "How many singers do we have?"
- "List all singers from France"
- "What are the names of concerts?"
- "Show me singers who have performed in more than 2 concerts"
