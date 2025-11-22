"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class QueryRequest(BaseModel):
    """Request model for generating SQL from natural language."""
    question: str = Field(..., description="Natural language question", min_length=1)
    db_id: Optional[str] = Field(None, description="Optional database identifier for context filtering")
    top_k: int = Field(5, description="Number of context entries to retrieve", ge=1, le=20)
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "Show all singers from USA",
                "db_id": "concert_singer",
                "top_k": 5
            }
        }


class RetrievedContext(BaseModel):
    """Model for retrieved context entry."""
    type: str = Field(..., description="Entry type: 'schema' or 'example'")
    db_id: str = Field(..., description="Database identifier")
    distance: float = Field(..., description="Distance score from query")
    question: Optional[str] = Field(None, description="Example question (if type is 'example')")
    sql: Optional[str] = Field(None, description="Example SQL (if type is 'example')")


class QueryResponse(BaseModel):
    """Response model for SQL generation."""
    question: str = Field(..., description="Original natural language question")
    generated_sql: str = Field(..., description="Generated SQL query")
    retrieved_context: Optional[List[RetrievedContext]] = Field(None, description="Retrieved context used for generation")
    db_id: Optional[str] = Field(None, description="Database identifier used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "Show all singers from USA",
                "generated_sql": "SELECT * FROM singer WHERE nationality = 'United States'",
                "retrieved_context": [
                    {
                        "type": "schema",
                        "db_id": "concert_singer",
                        "distance": 0.15
                    },
                    {
                        "type": "example",
                        "db_id": "concert_singer",
                        "distance": 0.23,
                        "question": "List all singers",
                        "sql": "SELECT * FROM singer"
                    }
                ],
                "db_id": "concert_singer"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    index_loaded: bool = Field(..., description="Whether the FAISS index is loaded")
    total_vectors: int = Field(..., description="Total number of vectors in index")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
