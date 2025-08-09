from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Any, Dict
from datetime import datetime

class QueryRequest(BaseModel):
    """Request model for the /hackrx/run endpoint"""
    documents: str = Field(
        ...,
        description="URL to the document (PDF/DOCX) to be processed",
        example="https://hackrx.blob.core.windows.net/assets/policy.pdf"
    )
    questions: List[str] = Field(
        ...,
        description="List of questions to be answered based on the document",
        min_items=1,
        example=[
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?"
        ]
    )

class QueryResponse(BaseModel):
    """Response model for the /hackrx/run endpoint"""
    success: bool = Field(
        True,
        description="Success status of the operation"
    )
    answers: List[str] = Field(
        ...,
        description="List of answers corresponding to the input questions",
        example=[
            "A grace period of thirty days is provided for premium payment.",
            "There is a waiting period of thirty-six months for pre-existing diseases."
        ]
    )
    processing_time: Optional[float] = Field(
        None,
        description="Processing time in seconds"
    )
    document_processed: Optional[bool] = Field(
        None,
        description="Whether document was successfully processed"
    )
    questions_count: Optional[int] = Field(
        None,
        description="Number of questions processed"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Response timestamp"
    )

class DocumentChunk(BaseModel):
    """Model for document chunks used in processing"""
    content: str = Field(..., description="Text content of the chunk")
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    page_number: Optional[int] = Field(None, description="Page number where chunk is located")
    start_char: Optional[int] = Field(None, description="Starting character position")
    end_char: Optional[int] = Field(None, description="Ending character position")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

class EmbeddingResult(BaseModel):
    """Model for embedding results"""
    chunk_id: str = Field(..., description="ID of the embedded chunk")
    embedding: List[float] = Field(..., description="Vector embedding")
    similarity_score: Optional[float] = Field(None, description="Similarity score when used in search")

class SearchResult(BaseModel):
    """Model for search results"""
    chunk: DocumentChunk = Field(..., description="The relevant document chunk")
    score: float = Field(..., description="Relevance score")
    rank: int = Field(..., description="Rank in search results")

class LLMResponse(BaseModel):
    """Model for LLM responses"""
    answer: str = Field(..., description="Generated answer")
    confidence: Optional[float] = Field(None, description="Confidence score")
    reasoning: Optional[str] = Field(None, description="Reasoning behind the answer")
    sources: Optional[List[str]] = Field(default_factory=list, description="Source chunks used")

class ProcessingStatus(BaseModel):
    """Model for tracking processing status"""
    status: str = Field(..., description="Current status")
    progress: Optional[float] = Field(None, description="Progress percentage")
    message: Optional[str] = Field(None, description="Status message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")

class ErrorResponse(BaseModel):
    """Model for error responses"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
