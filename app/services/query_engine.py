from typing import List, Dict, Any
import logging
import asyncio
from datetime import datetime

from .document_processor import DocumentProcessor
from .embedding_service import EmbeddingService
from .llm_service import LLMService
from ..models.schemas import DocumentChunk, SearchResult, LLMResponse

logger = logging.getLogger(__name__)

class QueryEngine:
    """Main query processing engine that orchestrates all services"""
    
    def __init__(self, document_processor: DocumentProcessor, 
                 embedding_service: EmbeddingService, 
                 llm_service: LLMService):
        self.document_processor = document_processor
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        
        # Cache for processed documents
        self.document_cache = {}
        
    async def process_query(self, document_url: str, questions: List[str]) -> List[str]:
        """
        Main method to process a query request
        
        Steps:
        1. Process document and create chunks
        2. Create embeddings and add to search index
        3. For each question, find relevant context
        4. Generate answers using LLM
        5. Return structured responses
        """
        try:
            start_time = datetime.now()
            logger.info(f"Starting query processing for {len(questions)} questions")
            
            # Step 1: Process document
            chunks = await self._process_document_with_cache(document_url)
            
            # Step 2: Create embeddings and index
            await self._create_embeddings_for_chunks(chunks)
            
            # Step 3 & 4: Process each question
            answers = await self._process_questions(questions)
            
            # Log processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Query processing completed in {processing_time:.2f} seconds")
            
            return answers
            
        except Exception as e:
            logger.error(f"Error in query processing: {str(e)}")
            # Return error messages for all questions
            return [f"Error processing question: {str(e)}" for _ in questions]
    
    async def _process_document_with_cache(self, document_url: str) -> List[DocumentChunk]:
        """Process document with caching to avoid reprocessing same documents"""
        try:
            # Create a simple cache key from URL
            cache_key = document_url
            
            if cache_key in self.document_cache:
                logger.info("Using cached document chunks")
                return self.document_cache[cache_key]
            
            # Process new document
            logger.info("Processing new document")
            chunks = await self.document_processor.process_document(document_url)
            
            # Cache the results
            self.document_cache[cache_key] = chunks
            logger.info(f"Cached {len(chunks)} chunks for document")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise
    
    async def _create_embeddings_for_chunks(self, chunks: List[DocumentChunk]):
        """Create embeddings for document chunks"""
        try:
            # Clear previous index for new document
            self.embedding_service.clear_index()
            
            # Add chunks to embedding index
            self.embedding_service.add_chunks_to_index(chunks)
            
            logger.info("Embeddings created and indexed successfully")
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise
    
    async def _process_questions(self, questions: List[str]) -> List[str]:
        """Process all questions and generate answers"""
        try:
            logger.info(f"Processing {len(questions)} questions")
            
            # Process questions concurrently for better performance
            tasks = [self._process_single_question(question) for question in questions]
            answers = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            results = []
            for i, answer in enumerate(answers):
                if isinstance(answer, Exception):
                    logger.error(f"Error processing question {i}: {str(answer)}")
                    results.append(f"Error processing question: {str(answer)}")
                else:
                    results.append(answer)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing questions: {str(e)}")
            return [f"Error processing questions: {str(e)}" for _ in questions]
    
    async def _process_single_question(self, question: str) -> str:
        """Process a single question and return the answer"""
        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            # Step 1: Get relevant context using embedding search
            context = self.embedding_service.get_relevant_context(question)
            
            if not context:
                logger.warning(f"No relevant context found for question: {question[:50]}...")
                return "Unable to find relevant information in the document to answer this question."
            
            # Step 2: Generate answer using LLM
            llm_response = await self.llm_service.generate_answer(context, question)
            
            return llm_response.answer
            
        except Exception as e:
            logger.error(f"Error processing single question: {str(e)}")
            return f"Error processing question: {str(e)}"
    
    def get_search_results_for_question(self, question: str, top_k: int = 5) -> List[SearchResult]:
        """Get detailed search results for a question (useful for debugging/analysis)"""
        try:
            return self.embedding_service.search_similar_chunks(question, top_k)
        except Exception as e:
            logger.error(f"Error getting search results: {str(e)}")
            return []
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about the current processing state"""
        try:
            return {
                "cached_documents": len(self.document_cache),
                "indexed_chunks": self.embedding_service.index.ntotal if self.embedding_service.index else 0,
                "model_info": {
                    "embedding_model": self.embedding_service.model_name,
                    "llm_model": self.llm_service.model_name,
                    "embedding_dimension": self.embedding_service.dimension
                }
            }
        except Exception as e:
            logger.error(f"Error getting processing stats: {str(e)}")
            return {}
    
    def clear_cache(self):
        """Clear document cache and embeddings"""
        try:
            self.document_cache.clear()
            self.embedding_service.clear_index()
            logger.info("Cache and embeddings cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
    
    async def health_check(self) -> Dict[str, str]:
        """Perform health check on all services"""
        health_status = {}
        
        try:
            # Check document processor
            health_status["document_processor"] = "healthy"
            
            # Check embedding service
            if self.embedding_service.model:
                health_status["embedding_service"] = "healthy"
            else:
                health_status["embedding_service"] = "unhealthy - model not loaded"
            
            # Check LLM service
            if self.llm_service.client:
                health_status["llm_service"] = "healthy"
            else:
                health_status["llm_service"] = "unhealthy - client not initialized"
            
            health_status["overall"] = "healthy" if all(
                status.startswith("healthy") for status in health_status.values()
            ) else "degraded"
            
        except Exception as e:
            logger.error(f"Error in health check: {str(e)}")
            health_status["overall"] = "unhealthy"
            health_status["error"] = str(e)
        
        return health_status
