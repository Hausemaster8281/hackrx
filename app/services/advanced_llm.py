"""
Advanced LLM service with API key rotation and pro model support
"""
import google.generativeai as genai
from typing import List, Optional, Dict, Any
import logging
import asyncio
import random
import time

from ..core.config import settings
from .response_cache import ResponseCache

logger = logging.getLogger(__name__)

class AdvancedLLMService:
    """LLM Service with API key rotation and advanced model support"""
    
    def __init__(self):
        self.cache = ResponseCache()
        self.api_keys = self._get_api_keys()
        self.current_key_index = 0
        self.model_name = settings.gemini_model
        self.clients = {}
        
        # Initialize all clients
        self._initialize_clients()
    
    def _get_api_keys(self) -> List[str]:
        """Get all available API keys"""
        keys = []
        if settings.gemini_api_key:
            keys.append(settings.gemini_api_key)
        if hasattr(settings, 'gemini_api_key_2') and settings.gemini_api_key_2:
            keys.append(settings.gemini_api_key_2)
        if hasattr(settings, 'gemini_api_key_3') and settings.gemini_api_key_3:
            keys.append(settings.gemini_api_key_3)
        return [key for key in keys if key and key != "your_second_api_key_here"]
    
    def _initialize_clients(self):
        """Initialize Gemini clients for all API keys"""
        for i, api_key in enumerate(self.api_keys):
            try:
                genai.configure(api_key=api_key)
                client = genai.GenerativeModel(self.model_name)
                self.clients[i] = {
                    'client': client,
                    'api_key': api_key,
                    'last_used': 0,
                    'error_count': 0
                }
                logger.info(f"Initialized client {i} with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize client {i}: {e}")
    
    def _get_next_client(self):
        """Get the next available client with rate limit awareness"""
        if not self.clients:
            return None
        
        # Find client with lowest error count and longest time since last use
        best_client = None
        best_score = float('-inf')
        
        current_time = time.time()
        
        for client_id, client_info in self.clients.items():
            # Score based on time since last use and error count
            time_score = current_time - client_info['last_used']
            error_penalty = client_info['error_count'] * 10
            score = time_score - error_penalty
            
            if score > best_score:
                best_score = score
                best_client = client_info
                self.current_key_index = client_id
        
        return best_client
    
    async def generate_answer_advanced(self, query: str, context: str) -> str:
        """Generate answer with advanced features"""
        try:
            # Check cache first
            cached_response = self.cache.get_cached_response(query, context)
            if cached_response:
                logger.info("Returning cached response")
                return cached_response
            
            # Get the best available client
            client_info = self._get_next_client()
            if not client_info:
                return "No available API clients"
            
            # Clean and prepare context
            cleaned_context = self._clean_context(context)
            
            # Enhanced prompt for Pro model
            prompt = self._create_advanced_prompt(query, cleaned_context)
            
            # Mark client as being used
            client_info['last_used'] = time.time()
            
            # Generate with Pro model settings
            response = await client_info['client'].generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1000,  # Higher for Pro model
                    temperature=0.0,
                    top_p=0.8,
                    top_k=10,
                    candidate_count=1,
                )
            )
            
            if response and response.text:
                answer = response.text.strip()
                answer = self._clean_response(answer)
                
                # Cache the response
                self.cache.cache_response(query, context, answer)
                
                # Reset error count on success
                client_info['error_count'] = 0
                
                return answer
            
            return "Unable to generate response"
            
        except Exception as e:
            # Increment error count for this client
            if client_info:
                client_info['error_count'] += 1
            
            logger.error(f"Error with advanced LLM service: {e}")
            
            # If it's a quota error, try next client
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                logger.info("Quota exceeded, trying next client...")
                # Remove current client temporarily
                if self.current_key_index in self.clients:
                    self.clients[self.current_key_index]['error_count'] += 5
                
                # Try with another client
                return await self._retry_with_different_client(query, context)
            
            return f"Error generating response: {str(e)}"
    
    async def _retry_with_different_client(self, query: str, context: str) -> str:
        """Retry with a different API client"""
        try:
            client_info = self._get_next_client()
            if not client_info or client_info['error_count'] > 3:
                return "All API clients exhausted"
            
            # Simple retry logic
            response = await client_info['client'].generate_content_async(
                self._create_advanced_prompt(query, context),
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=800,
                    temperature=0.0,
                    top_p=0.8,
                    top_k=10,
                )
            )
            
            if response and response.text:
                answer = response.text.strip()
                return self._clean_response(answer)
            
            return "Retry failed"
            
        except Exception as e:
            logger.error(f"Retry failed: {e}")
            return "All retry attempts failed"
    
    def _create_advanced_prompt(self, query: str, context: str) -> str:
        """Create advanced prompt optimized for Pro model"""
        return f"""You are an expert insurance policy analyst using the most advanced AI capabilities. Provide comprehensive, precise answers based strictly on the provided context.

CONTEXT FROM INSURANCE DOCUMENT:
{context}

QUESTION: {query}

ADVANCED ANALYSIS REQUIREMENTS:
1. COMPREHENSIVE COVERAGE: Analyze all relevant information in the context
2. PRECISE EXTRACTION: Use exact numbers, timeframes, and conditions as stated
3. LOGICAL STRUCTURE: Organize information clearly and logically
4. COMPLETE CONDITIONS: Include all conditions, exceptions, and limitations
5. PROFESSIONAL LANGUAGE: Use clear, professional insurance terminology
6. CONTEXTUAL ACCURACY: Base every statement on the provided context

Provide a detailed, professional answer that fully addresses the question with all relevant details from the policy document."""
    
    def _clean_context(self, context: str) -> str:
        """Clean and optimize context for Pro model"""
        if not context:
            return "No relevant context found"
        
        # More sophisticated cleaning for Pro model
        import re
        context = re.sub(r'\n\s*\n', '\n\n', context)
        context = re.sub(r' +', ' ', context)
        
        # Pro model can handle longer context
        if len(context) > 15000:  # Increased limit for Pro
            sentences = context.split('. ')
            if len(sentences) > 80:
                context = '. '.join(sentences[:40] + ['[...additional relevant content...]'] + sentences[-40:])
        
        return context.strip()
    
    def _clean_response(self, response: str) -> str:
        """Clean response for Pro model output"""
        import re
        
        # Remove unwanted prefixes
        prefixes = [
            "Based on the provided context,",
            "According to the insurance document,",
            "The policy states that",
            "From the document provided,",
        ]
        
        for prefix in prefixes:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()
        
        # Clean quotes
        response = re.sub(r'^"([^"]+)"$', r'\1', response)
        response = re.sub(r"^'([^']+)'$", r'\1', response)
        
        # Ensure proper sentence structure
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            'total_clients': len(self.clients),
            'active_clients': len([c for c in self.clients.values() if c['error_count'] < 3]),
            'current_model': self.model_name,
            'cache_stats': self.cache.get_cache_stats(),
            'client_status': {
                i: {
                    'error_count': info['error_count'],
                    'last_used': info['last_used']
                }
                for i, info in self.clients.items()
            }
        }
