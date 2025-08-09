import google.generativeai as genai
from typing import List, Optional, Dict, Any
import logging
import asyncio
import re

from ..models.schemas import LLMResponse
from ..core.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    """Service for interacting with Google Gemini models"""
    
    def __init__(self):
        """Initialize the LLM service"""
        self.client = None
        self.model = None
        self.model_name = settings.gemini_model
        self.max_tokens = 2000  # Default max tokens
        self.temperature = 0.1  # Default temperature
        
        # Initialize Gemini client if API key is provided
        if settings.gemini_api_key:
            try:
                genai.configure(api_key=settings.gemini_api_key)
                self.model = genai.GenerativeModel(self.model_name)
                self.client = self.model  # For backward compatibility
                logger.info(f"Gemini client initialized with model: {self.model_name}")
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
    
    async def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using Gemini with enhanced prompting"""
        try:
            # Clean and prepare context
            cleaned_context = self.clean_context(context)
            
            system_prompt = self.create_system_prompt()
            
            # Enhanced prompt with explicit instructions
            prompt = f"""{system_prompt}

CONTEXT FROM DOCUMENTS:
{cleaned_context}

QUESTION: {query}

ANSWER REQUIREMENTS:
- Base your answer ONLY on the provided context
- Include all specific details, numbers, percentages, and timeframes
- Mention all conditions, exclusions, or special circumstances
- Use exact quotes for important figures or conditions
- If information is missing, explicitly state what is not available
- Provide a comprehensive, detailed response

ANSWER:"""

            response = await self.model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=400,  # Increased for more detailed answers
                    temperature=0.1,  # Lower temperature for more consistent answers
                    top_p=0.8,
                    top_k=20,
                )
            )
            
            if response and response.text:
                # Clean up the response
                answer = response.text.strip()
                # Remove any unwanted prefixes or formatting
                answer = self.clean_response(answer)
                return answer
            
            return "I apologize, but I couldn't generate a response. Please try again."
            
        except Exception as e:
            logger.error(f"Error generating answer with Gemini: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def clean_context(self, context: str) -> str:
        """Clean and prepare context for better processing"""
        if not context:
            return "No relevant context found in the documents."
        
        # Remove excessive whitespace and newlines
        context = re.sub(r'\n\s*\n', '\n\n', context)
        context = re.sub(r' +', ' ', context)
        
        # Remove redundant sections if context is too long
        if len(context) > 8000:  # Limit context length
            sentences = context.split('. ')
            # Keep first and last parts, remove middle if too long
            if len(sentences) > 50:
                context = '. '.join(sentences[:25] + ['[...relevant content continues...]'] + sentences[-25:])
        
        return context.strip()
    
    def clean_response(self, response: str) -> str:
        """Clean and format the LLM response"""
        # Remove common unwanted prefixes
        prefixes_to_remove = [
            "Based on the provided context,",
            "According to the document,",
            "The document states that",
            "From the context provided,",
            "As per the information given,",
        ]
        
        for prefix in prefixes_to_remove:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()
        
        # Ensure proper sentence structure
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response
    
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
