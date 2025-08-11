"""
Alternative LLM service using Google AI Studio with higher rate limits
"""
import requests
import json
import logging
from typing import Optional
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

class GoogleAIStudioService:
    """Alternative service using Google AI Studio API with higher limits"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
        
    async def generate_content(self, prompt: str, max_tokens: int = 800) -> Optional[str]:
        """Generate content using Google AI Studio API"""
        try:
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.0,
                    "topK": 10,
                    "topP": 0.8,
                    "maxOutputTokens": max_tokens,
                    "stopSequences": []
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ]
            }
            
            headers = {
                "Content-Type": "application/json",
            }
            
            url = f"{self.base_url}?key={self.api_key}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'candidates' in data and len(data['candidates']) > 0:
                            content = data['candidates'][0]['content']['parts'][0]['text']
                            return content.strip()
                    else:
                        error_text = await response.text()
                        logger.error(f"AI Studio API error {response.status}: {error_text}")
                        return None
            
        except Exception as e:
            logger.error(f"Error with AI Studio API: {e}")
            return None

# Alternative configuration for higher limits
class HighLimitLLMService:
    """LLM Service with multiple API keys for higher limits"""
    
    def __init__(self):
        # List of API keys for rotation
        self.api_keys = [
            settings.gemini_api_key,
            # Add more keys here if available
        ]
        self.current_key_index = 0
        self.ai_studio_service = None
        
        if settings.gemini_api_key:
            self.ai_studio_service = GoogleAIStudioService(settings.gemini_api_key)
    
    async def generate_with_rotation(self, prompt: str, max_tokens: int = 800) -> Optional[str]:
        """Generate content with API key rotation"""
        for attempt in range(len(self.api_keys)):
            try:
                current_key = self.api_keys[self.current_key_index]
                service = GoogleAIStudioService(current_key)
                
                result = await service.generate_content(prompt, max_tokens)
                if result:
                    return result
                    
            except Exception as e:
                logger.warning(f"API key {self.current_key_index} failed: {e}")
                self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        
        return None
