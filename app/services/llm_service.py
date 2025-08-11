import google.generativeai as genai
from typing import List, Optional, Dict, Any
import logging
import asyncio
import re

from ..models.schemas import LLMResponse
from ..core.config import settings
from .response_cache import ResponseCache

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
        
        # Initialize response cache
        self.cache = ResponseCache()
        
        # API key cycling setup
        self.available_keys = settings.get_available_gemini_keys()
        self.current_key_index = 0
        
        # Initialize Gemini client if API keys are available
        if self.available_keys:
            try:
                self._configure_current_key()
                logger.info(f"Gemini client initialized with model: {self.model_name}")
                logger.info(f"Available API keys: {len(self.available_keys)}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
        else:
            logger.warning("No Gemini API key provided. LLM service will return fallback responses.")
    
    def _configure_current_key(self):
        """Configure Gemini with the current API key"""
        current_key = self.available_keys[self.current_key_index]
        genai.configure(api_key=current_key)
        self.model = genai.GenerativeModel(self.model_name)
        self.client = self.model  # For backward compatibility
    
    def _cycle_to_next_key(self):
        """Cycle to the next available API key"""
        if len(self.available_keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(self.available_keys)
            self._configure_current_key()
            logger.info(f"Cycled to API key {self.current_key_index + 1}/{len(self.available_keys)}")
            return True
        return False
    
    def _initialize_client(self):
        """Initialize the OpenAI client"""
        # Client is already initialized in __init__ method
        # This method is kept for compatibility but not needed
        pass
    
    def create_system_prompt(self) -> str:
        """Create the system prompt for the LLM"""
        return """You are an expert insurance policy analyst. Provide comprehensive, detailed answers based strictly on the provided context.

ANSWER REQUIREMENTS:
1. COMPREHENSIVE: Provide complete answers with all relevant details, conditions, and exceptions
2. NATURAL FLOW: Write in natural, professional language without structured labels or formatting
3. SPECIFIC DETAILS: Include exact numbers, timeframes, percentages, and monetary amounts
4. ALL CONDITIONS: Mention all conditions, limitations, and exceptions that apply
5. CONTEXT-GROUNDED: Base answers entirely on the provided document context
6. COMPLETE COVERAGE: Address all aspects of the question thoroughly

STYLE GUIDELINES:
- Write in flowing, professional paragraphs
- Include specific details and exact figures from the document
- Explain conditions and limitations naturally within the answer
- Provide context for why certain conditions exist
- Use the exact terminology from the policy document
- Ensure answers are thorough but concise and readable

CRITICAL RULES:
- Never use structured labels like "[Direct Answer]" or "[Justification]"
- Write complete, comprehensive sentences that fully address the question
- Include all relevant conditions, waiting periods, and limitations
- Quote exact figures and timeframes as they appear in the document
- If information is not in the context, clearly state what is not specified

Provide detailed, professional answers that comprehensively address each question based solely on the document content."""
    
    def create_user_prompt(self, context: str, question: str) -> str:
        """Create the user prompt with context and question"""
        return f"""Based on the following context from the document, please answer the question comprehensively:

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    
    async def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using Gemini with caching"""
        try:
            # Check cache first
            cached_response = self.cache.get_cached_response(query, context)
            if cached_response:
                logger.info("Returning cached response")
                return cached_response
            
            # Clean and prepare context
            cleaned_context = self.clean_context(context)
            
            system_prompt = self.create_system_prompt()
            
            # Enhanced prompt with explicit instructions
            prompt = f"""{system_prompt}

DOCUMENT CONTEXT:
{cleaned_context}

QUESTION: {query}

INSTRUCTIONS:
Provide a comprehensive answer based on the document context above. Include all relevant details, conditions, timeframes, and limitations mentioned in the document. Write in natural, professional language without structured formatting. Ensure the answer fully addresses the question with specific details from the policy.

ANSWER:"""

            response = await self.model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=800,  # Increased for comprehensive answers with justification
                    temperature=0.0,  # Minimum temperature for maximum consistency
                    top_p=0.8,
                    top_k=10,  # More focused selection
                )
            )
            
            if response and response.text:
                # Clean up the response
                answer = response.text.strip()
                # Remove any unwanted prefixes or formatting
                answer = self.clean_response(answer)
                
                # Cache the response
                self.cache.cache_response(query, context, answer)
                
                return answer
            
            return "I apologize, but I couldn't generate a response. Please try again."
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if it's a quota/rate limit error
            if "quota" in error_str or "429" in error_str or "exceeded" in error_str:
                logger.warning(f"Quota exceeded for current API key. Error: {str(e)}")
                
                # Try cycling to next key
                if self._cycle_to_next_key():
                    logger.info("Retrying with next API key...")
                    try:
                        response = await self.model.generate_content_async(
                            prompt,
                            generation_config=genai.types.GenerationConfig(
                                max_output_tokens=800,
                                temperature=0.0,
                                top_p=0.8,
                                top_k=10,
                            )
                        )
                        
                        if response and response.text:
                            answer = response.text.strip()
                            answer = self.clean_response(answer)
                            self.cache.cache_response(query, context, answer)
                            return answer
                    except Exception as retry_e:
                        logger.error(f"Retry with next key also failed: {str(retry_e)}")
                
                return f"I apologize, but all API keys have reached their quota limits. Please try again later. Error: {str(e)}"
            
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
            "The grace period for premium payment is",
            "The waiting period for pre-existing diseases is",
            "The answer is",
        ]
        
        for prefix in prefixes_to_remove:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()
        
        # Remove excessive quotes around the main content
        response = response.strip('"').strip("'")
        
        # Clean up quote formatting - remove quotes around specific phrases but keep the content
        # Remove patterns like "thirty days (where premium is paid...)" and just return: thirty days (where premium is paid...)
        import re
        response = re.sub(r'^"([^"]+)"$', r'\1', response)
        response = re.sub(r"^'([^']+)'$", r'\1', response)
        
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
