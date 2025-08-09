from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Import our custom modules
from .services.document_processor import DocumentProcessor
from .services.embedding_service import EmbeddingService
from .services.llm_service import LLMService
from .services.query_engine import QueryEngine
from .models.schemas import QueryRequest, QueryResponse
from .core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize services
document_processor = DocumentProcessor()
embedding_service = EmbeddingService()
llm_service = LLMService()
query_engine = QueryEngine(document_processor, embedding_service, llm_service)

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the Bearer token"""
    expected_token = settings.api_token
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return credentials.credentials

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HackRx 6.0 - LLM Query Retrieval System",
        "status": "active",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "document_processor": "active",
            "embedding_service": "active",
            "llm_service": "active",
            "query_engine": "active"
        }
    }

@app.post("/hackrx/run")
@app.post("/api/v1/hackrx/run")
async def process_query(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """
    Main endpoint for processing document queries
    
    This endpoint:
    1. Downloads and processes documents from the provided URL
    2. Creates embeddings for document chunks
    3. Performs semantic search for each question
    4. Uses LLM to generate accurate answers
    5. Returns structured JSON responses
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Processing query request with {len(request.questions)} questions")
        
        # Process the query using our query engine
        answers = await query_engine.process_query(
            document_url=request.documents,
            questions=request.questions
        )
        
        processing_time = time.time() - start_time
        
        # Return response in exact HackRx format
        response = {
            "answers": answers
        }
        logger.info(f"Successfully processed {len(answers)} answers in {processing_time:.2f}s")
        
        return response
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error processing query: {str(e)}")
        
        # Return error response in HackRx format
        error_response = {
            "answers": [f"Error processing question: {str(e)}" for _ in request.questions]
        }
        return error_response

@app.get("/api/v1/status")
async def api_status():
    """API status endpoint"""
    return {
        "api_version": "v1",
        "status": "operational",
        "endpoints": {
            "hackrx_run": "/hackrx/run",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD
    )
