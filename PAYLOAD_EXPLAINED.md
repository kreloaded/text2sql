# What Gets Sent to the Colab Model

## Overview
When you make a request to the FastAPI backend, here's the flow:
1. **User** → sends question to FastAPI `/generate` endpoint
2. **FastAPI** → optionally retrieves context from FAISS
3. **FastAPI** → builds prompt and sends to Colab
4. **Colab** → receives prompt, runs model, returns SQL
5. **FastAPI** → returns SQL to user

---

## Example 1: WITHOUT Retrieval

### User Request
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"question": "Show all singers from USA", "db_id": "concert_singer", "use_retrieval": false}'
```

### What FastAPI Sends to Colab

**HTTP Request:**
```
POST https://your-ngrok-url.ngrok-free.app/generate
Content-Type: application/json
```

**JSON Payload:**
```json
{
  "prompt": "Question: Show all singers from USA\nSQL:",
  "max_new_tokens": 128
}
```

**Explanation:**
- `prompt`: Simple format with just the question
- `max_new_tokens`: How many tokens the model can generate (128 = ~50-80 words)

---

## Example 2: WITH Retrieval (RAG)

### User Request
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the names of all artists?", "db_id": "concert_singer", "use_retrieval": true, "top_k": 3}'
```

### What Happens in FastAPI Backend

1. **FAISS Retrieval**: Searches vector store for similar examples
2. **Context Building**: Combines schemas + examples
3. **Prompt Building**: Creates a rich prompt with context

### What FastAPI Sends to Colab

**HTTP Request:**
```
POST https://your-ngrok-url.ngrok-free.app/generate
Content-Type: application/json
```

**JSON Payload:**
```json
{
  "prompt": "Database Schema:\n- concert_singer\n\nRelevant Examples:\n1. Q: What are the names of all singers?\n   SQL: SELECT name FROM singer\n2. Q: List all artist names.\n   SQL: SELECT Name FROM Artist\n3. Q: Show me all concert names.\n   SQL: SELECT concert_name FROM concert\n\nQuestion: What are the names of all artists?\nSQL:",
  "max_new_tokens": 128
}
```

**Explanation:**
- `prompt`: Now includes:
  - **Database Schema**: Which database we're querying
  - **Relevant Examples**: Top 3 similar question-SQL pairs from FAISS
  - **Actual Question**: The user's question
  - **SQL:** Prompt ending to trigger SQL generation
- This is **RAG (Retrieval-Augmented Generation)** - using retrieved examples to improve results

---

## What Happens in Colab

### Flask Server Receives
```python
# In Colab Flask server (Cell 5)
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')              # Gets the full prompt
    max_new_tokens = data.get('max_new_tokens', 128)  # Gets token limit
    
    # Tokenize the prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    ).to(device)
    
    # Generate SQL
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask', None),
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode to text
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Return result
    return jsonify({
        "generated_sql": sql_query.strip(),
        "status": "success"
    })
```

### Colab Sends Back
```json
{
  "generated_sql": "SELECT Name FROM Artist",
  "status": "success"
}
```

---

## Key Parameters Explained

### `prompt` (string)
- The text input to the model
- Can be simple ("Question: X\nSQL:") or complex (with schema + examples)
- Model was trained on this format during fine-tuning on Spider dataset

### `max_new_tokens` (integer, default: 128)
- Maximum number of tokens to generate
- 128 tokens ≈ 50-80 words
- Enough for most SQL queries
- Prevents runaway generation

### Model Generation Parameters (in Colab)
- `num_beams=4`: Beam search for better quality (explores 4 alternatives)
- `early_stopping=True`: Stops when EOS token is generated
- `max_length=512`: Maximum input length (truncates if longer)
- `truncation=True`: Cuts off prompt if > 512 tokens
- `padding=True`: Adds padding for batching

---

## Size Comparison

| Scenario | Prompt Length | Example |
|----------|---------------|---------|
| No retrieval | ~30-50 chars | "Question: Show all singers\nSQL:" |
| With retrieval (3 examples) | ~300-800 chars | Includes schema + 3 Q&A pairs + question |
| With retrieval (5 examples) | ~500-1200 chars | More examples = better context |

---

## Why This Format?

The CodeT5+ model (`tzaware/codet5p-spider`) was **fine-tuned on the Spider dataset** with this exact format:
- Database schemas
- Example question-SQL pairs
- Question to answer
- Expected output: SQL query

The model learned to:
1. Parse the schema structure
2. Learn from similar examples
3. Generate correct SQL for the new question

This is why retrieval helps - giving the model similar examples improves accuracy!

---

## Debug Tips

### See exactly what's sent:
Check your backend terminal logs - it prints:
```
DEBUG: Prompt length: 456 chars
DEBUG: Calling HuggingFace API...
DEBUG: Status code: 200
DEBUG: Generated SQL: SELECT ...
```

### Test Colab health:
```bash
curl https://your-ngrok-url.ngrok-free.app/health
```

### Test with custom prompt:
You can send a raw prompt to Colab:
```bash
curl -X POST https://your-ngrok-url.ngrok-free.app/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Question: List all tables\nSQL:", "max_new_tokens": 50}'
```
