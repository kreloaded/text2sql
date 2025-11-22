"""
Flask frontend for Text2SQL system
Provides a web interface for natural language to SQL conversion
"""

from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(__name__)

# Backend API configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


@app.route("/")
def index():
    """Render the main chat interface"""
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate_sql():
    """
    Proxy endpoint to call backend SQL generation API
    Accepts: {"question": str, "db_id": str}
    Returns: Backend response with generated SQL and context
    """
    try:
        data = request.json
        question = data.get("question", "")
        if question:
            question = question.strip()
        db_id = data.get("db_id", "")
        if db_id:
            db_id = db_id.strip()

        if not question:
            return jsonify({"error": "Question is required"}), 400

        # Call backend API
        response = requests.post(
            f"{BACKEND_URL}/generate",
            json={"question": question, "db_id": db_id if db_id else None},
            timeout=30,
        )

        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return (
                jsonify(
                    {
                        "error": f"Backend error: {response.status_code}",
                        "details": response.text,
                    }
                ),
                response.status_code,
            )

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to connect to backend: {str(e)}"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    """Check health of both frontend and backend"""
    try:
        backend_response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        backend_status = (
            backend_response.json() if backend_response.status_code == 200 else None
        )

        return jsonify(
            {
                "frontend": "healthy",
                "backend": backend_status,
                "backend_url": BACKEND_URL,
            }
        )
    except Exception as e:
        return jsonify(
            {
                "frontend": "healthy",
                "backend": "unreachable",
                "error": str(e),
                "backend_url": BACKEND_URL,
            }
        )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
