"""
Vector Retrieval Service
Loads FAISS index and retrieves relevant schema/examples for queries.
"""

import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import os


class VectorRetriever:
    def __init__(self, index_path: str, metadata_path: str, embedding_model: str):
        """
        Initialize the vector retriever.

        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata JSON file
            embedding_model: Name of the sentence-transformers model
        """
        self.index_path = index_path
        self.metadata_path = metadata_path

        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)

        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        print(f"Loading embedding model: {embedding_model}...")
        self.model = SentenceTransformer(embedding_model)

        print(f"✓ Vector retriever initialized with {self.index.ntotal} vectors")

    def retrieve(
        self, query: str, top_k: int = 5, filter_type: str = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant entries for a query.

        Args:
            query: Natural language query
            top_k: Number of results to return
            filter_type: Optional filter for entry type ('schema' or 'example')

        Returns:
            List of metadata dictionaries for the most relevant entries
        """
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype(
            "float32"
        )

        # Search FAISS index
        distances, indices = self.index.search(
            query_embedding, top_k * 2
        )  # Get more than needed for filtering

        # Retrieve metadata
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                meta = self.metadata[idx].copy()
                meta["distance"] = float(distance)

                # Apply filter if specified
                if filter_type is None or meta.get("type") == filter_type:
                    results.append(meta)

                # Stop when we have enough results
                if len(results) >= top_k:
                    break

        return results

    def retrieve_by_db(
        self, query: str, db_id: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant entries for a specific database.
        Falls back to general examples if not enough db-specific results.

        Args:
            query: Natural language query
            db_id: Database identifier
            top_k: Number of results to return

        Returns:
            List of metadata dictionaries filtered by database
        """
        # Get target schema directly (not via semantic search)
        target_schemas = self.get_schema_for_db(db_id)

        # Get examples using semantic search
        all_results = self.retrieve(query, top_k=top_k * 5, filter_type=None)

        print(f"DEBUG: Retrieved {len(all_results)} results before filtering")
        print(f"DEBUG: Found {len(target_schemas)} schemas for db_id='{db_id}'")

        # Separate examples by database
        db_examples = [
            r
            for r in all_results
            if r.get("db_id") == db_id and r.get("type") == "example"
        ]
        general_examples = [r for r in all_results if r.get("type") == "example"]

        print(f"DEBUG: Found {len(db_examples)} examples for {db_id}")

        # Build result: schema + examples
        results = []

        # Always include target schema (directly looked up, not from semantic search)
        if target_schemas:
            schema_meta = target_schemas[0]
            results.append(
                {
                    "type": "schema",
                    "db_id": db_id,
                    "text": schema_meta.get("schema", ""),
                    "distance": 0.0,  # Direct lookup, no distance
                    "question": None,
                    "sql": None,
                }
            )

        # Try to get db-specific examples, fall back to general if needed
        if len(db_examples) >= top_k - 1:
            # Enough db-specific examples
            results.extend(db_examples[: top_k - 1])
        else:
            # Not enough db-specific examples, use general ones
            results.extend(db_examples)  # Add all db-specific
            needed = top_k - len(results)
            if needed > 0:
                print(f"DEBUG: Need {needed} more examples, using general examples")
                results.extend(general_examples[:needed])

        print(
            f"DEBUG: Returning {len(results)} total results ({sum(1 for r in results if r.get('type')=='schema')} schemas, {sum(1 for r in results if r.get('type')=='example')} examples)"
        )

        return results[:top_k]

    def get_schema_for_db(self, db_id: str) -> List[Dict[str, Any]]:
        """
        Get all schema entries for a specific database.

        Args:
            db_id: Database identifier

        Returns:
            List of schema metadata for the database
        """
        schemas = []
        for meta in self.metadata:
            if meta.get("type") == "schema" and meta.get("db_id") == db_id:
                schemas.append(meta)
        return schemas
