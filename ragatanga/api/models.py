"""
Pydantic models for API requests and responses.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="The query string")

class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    retrieved_facts_sparql: List[str] = Field(default_factory=list, description="Facts retrieved from SPARQL queries")
    retrieved_facts_semantic: List[str] = Field(default_factory=list, description="Facts retrieved from semantic search")
    retrieved_facts: List[str] = Field(default_factory=list, description="All retrieved facts")
    answer: str = Field("", description="The generated answer")

    class Config:
        allow_mutation = True

class StatusResponse(BaseModel):
    """Response model for status messages."""
    message: str = Field(..., description="Status message")
    success: bool = Field(True, description="Whether the operation was successful")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")

class OntologyStatistics(BaseModel):
    """Model for ontology statistics."""
    statistics: Dict[str, int] = Field(..., description="Overall statistics")
    classes: Dict[str, Dict[str, Any]] = Field(..., description="Class information")
    properties: Dict[str, Dict[str, Any]] = Field(..., description="Property information")
    individuals: Dict[str, Dict[str, Any]] = Field(..., description="Individual information")
    metadata: Dict[str, Any] = Field(..., description="Metadata about the ontology")