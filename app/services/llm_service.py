import google.generativeai as genai
from typing import List, Optional, Dict, Any
import logging
import asyncio

from ..models.schemas import LLMResponse
from ..core.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    """Service for interacting with Google Gemini models"""
    
    def __init__(self):
        """Initialize the LLM service"""
        self.client = None
        self.model = settings.gemini_model
        self.max_tokens = 2000  # Default max tokens
        self.temperature = 0.1  # Default temperature
        
        # Initialize Gemini client if API key is provided
        if settings.gemini_api_key:
            try:
                genai.configure(api_key=settings.gemini_api_key)
                self.client = genai.GenerativeModel(self.model)
                logger.info(f"Gemini client initialized with model: {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
        else:
            logger.warning("No Gemini API key provided. LLM service will return fallback responses.")
    
    def _initialize_client(self):
        """Initialize the OpenAI client"""
        # Client is already initialized in __init__ method
        # This method is kept for compatibility but not needed
        pass
    
    def create_system_prompt(self) -> str:
        """Create the system prompt for the LLM"""
        return """You are an expert document analysis assistant specializing in insurance policies, legal documents, HR policies, and compliance documents.

Your task is to answer questions based on the provided context from documents. Follow these guidelines:

1. ACCURACY: Provide precise, factual answers based ONLY on the provided context
2. COMPLETENESS: Include all relevant details, numbers, timeframes, and conditions
3. CLARITY: Use clear, professional language that's easy to understand
4. SPECIFICITY: Always include exact numbers, percentages, timeframes, and monetary amounts
5. CONTEXT-BASED: Never add information not present in the context
6. COMPREHENSIVE: Provide detailed explanations with all conditions and exceptions
7. STRUCTURE: Organize complex answers logically with proper flow

CRITICAL RULES:
- If specific information is not in the context, state "Information not specified in the document"
- Always quote exact figures, timeframes, and conditions from the document
- Include relevant policy clauses, section numbers, or references when available
- Explain any conditions, exclusions, or special circumstances mentioned

Format your answers as detailed, professional responses that fully address the question."""
    
    def create_user_prompt(self, context: str, question: str) -> str:
        """Create the user prompt with context and question"""
        return f"""Based on the following context from the document, please answer the question comprehensively:

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    
    async def generate_answer(self, context: str, question: str) -> LLMResponse:
        """Generate an answer using the LLM"""
        try:
            if not self.client:
                # Fallback response when OpenAI is not available
                return LLMResponse(
                    answer="Unable to process question - LLM service not available",
                    confidence=0.0,
                    reasoning="OpenAI API key not configured",
                    sources=[]
                )
            
            logger.info(f"Generating answer for question: {question[:100]}...")
            
            # Create prompts
            system_prompt = self.create_system_prompt()
            user_prompt = self.create_user_prompt(context, question)
            
            # Combine system and user prompt for Gemini
            full_prompt = f"{system_prompt}\n\nDocument Context:\n{context}\n\nQuestion: {question}"
            
            # Call Gemini API
            response = await asyncio.to_thread(
                self.client.generate_content,
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=300,  # Increased for detailed responses like the sample
                    temperature=self.temperature
                )
            )
            
            answer = response.text.strip()
            
            logger.info(f"Generated answer: {answer[:100]}...")
            
            return LLMResponse(
                answer=answer,
                confidence=0.9,  # High confidence when using Gemini
                reasoning=f"Answer generated from document context using {self.model}",
                sources=["document_context"]
            )
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            # Return a fallback response instead of raising exception
            return LLMResponse(
                answer=f"Unable to generate answer due to error: {str(e)}",
                confidence=0.0,
                reasoning="Error in LLM processing",
                sources=[]
            )
    
    async def generate_answers_batch(self, context_question_pairs: List[tuple]) -> List[LLMResponse]:
        """Generate answers for multiple questions in batch"""
        try:
            logger.info(f"Generating answers for {len(context_question_pairs)} questions")
            
            # Process questions concurrently
            tasks = [
                self.generate_answer(context, question)
                for context, question in context_question_pairs
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions in the batch
            results = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    logger.error(f"Error in batch question {i}: {str(response)}")
                    results.append(LLMResponse(
                        answer=f"Error processing question: {str(response)}",
                        confidence=0.0,
                        reasoning="Batch processing error",
                        sources=[]
                    ))
                else:
                    results.append(response)
            
            logger.info(f"Completed batch processing of {len(results)} questions")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            # Return fallback responses for all questions
            return [
                LLMResponse(
                    answer=f"Batch processing error: {str(e)}",
                    confidence=0.0,
                    reasoning="Batch processing failure",
                    sources=[]
                )
                for _ in context_question_pairs
            ]
    
    def extract_key_information(self, text: str, info_type: str = "general") -> Dict[str, Any]:
        """Extract key information from text for better processing"""
        try:
            # This could be enhanced with more sophisticated extraction
            # For now, we'll return basic text analysis
            
            word_count = len(text.split())
            char_count = len(text)
            
            # Look for common patterns
            patterns = {
                "numbers": len([w for w in text.split() if any(c.isdigit() for c in w)]),
                "percentages": text.count('%'),
                "dates": text.count('/') + text.count('-'),
                "currency": text.count('$') + text.count('₹') + text.lower().count('rupees')
            }
            
            return {
                "word_count": word_count,
                "char_count": char_count,
                "patterns": patterns,
                "complexity": "high" if word_count > 500 else "medium" if word_count > 200 else "low"
            }
            
        except Exception as e:
            logger.error(f"Error extracting key information: {str(e)}")
            return {}
