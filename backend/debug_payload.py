"""
Debug script to show exactly what is sent to the Colab model
"""

import sys

sys.path.append(".")

from services.sql_generator import SQLGenerator
from services.retriever import VectorRetriever
import json

# Initialize services
print("Initializing services...")
retriever = VectorRetriever(
    index_path="vector_service/output/faiss.index",
    metadata_path="vector_service/output/metadata.json",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
)

generator = SQLGenerator(model_name="tzaware/codet5p-spider", use_api=True)

# Test 1: Without retrieval
print("\n" + "=" * 80)
print("TEST 1: WITHOUT RETRIEVAL")
print("=" * 80)
question1 = "Show all singers from USA"
prompt1 = f"Question: {question1}\nSQL:"
print("\nPrompt sent to model:")
print("-" * 80)
print(prompt1)
print("-" * 80)

payload1 = {"prompt": prompt1, "max_new_tokens": 128}
print("\nJSON payload sent to Colab:")
print(json.dumps(payload1, indent=2))

# Test 2: With retrieval
print("\n" + "=" * 80)
print("TEST 2: WITH RETRIEVAL")
print("=" * 80)
question2 = "What are the names of all artists?"
print(f"\nOriginal question: {question2}")

# Retrieve context
print("\nRetrieving similar examples...")
context = retriever.retrieve(question2, top_k=3)
print(f"Retrieved {len(context)} items")

# Build prompt
prompt2 = generator.build_prompt(question2, context)
print("\nPrompt sent to model:")
print("-" * 80)
print(prompt2)
print("-" * 80)

payload2 = {"prompt": prompt2, "max_new_tokens": 128}
print("\nJSON payload sent to Colab:")
print(json.dumps(payload2, indent=2))

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Without retrieval: {len(prompt1)} characters")
print(f"With retrieval: {len(prompt2)} characters")
print(f"\nThe Colab Flask server receives a POST request to /generate")
print(f"with the above JSON payload containing 'prompt' and 'max_new_tokens'")
