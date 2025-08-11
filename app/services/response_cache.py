"""
Response caching service to reduce API calls
"""
import hashlib
import json
import os
import pickle
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ResponseCache:
    """Cache LLM responses to reduce API calls"""
    
    def __init__(self, cache_dir: str = "./cache", cache_duration_hours: int = 24):
        self.cache_dir = cache_dir
        self.cache_duration = timedelta(hours=cache_duration_hours)
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
    def _generate_cache_key(self, query: str, context: str) -> str:
        """Generate a cache key based on query and context"""
        # Create a hash of the query and context
        content = f"{query}|{context}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get the file path for a cache key"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def get_cached_response(self, query: str, context: str) -> Optional[str]:
        """Get cached response if available and not expired"""
        try:
            cache_key = self._generate_cache_key(query, context)
            cache_path = self._get_cache_path(cache_key)
            
            if not os.path.exists(cache_path):
                return None
            
            # Load cached data
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Check if cache is still valid
            if datetime.now() - cached_data['timestamp'] > self.cache_duration:
                os.remove(cache_path)  # Remove expired cache
                return None
            
            logger.info(f"Cache hit for query: {query[:50]}...")
            return cached_data['response']
            
        except Exception as e:
            logger.error(f"Error retrieving cached response: {e}")
            return None
    
    def cache_response(self, query: str, context: str, response: str):
        """Cache a response"""
        try:
            cache_key = self._generate_cache_key(query, context)
            cache_path = self._get_cache_path(cache_key)
            
            cached_data = {
                'query': query,
                'context_hash': hashlib.md5(context.encode()).hexdigest(),
                'response': response,
                'timestamp': datetime.now()
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
            
            logger.info(f"Cached response for query: {query[:50]}...")
            
        except Exception as e:
            logger.error(f"Error caching response: {e}")
    
    def clear_cache(self):
        """Clear all cached responses"""
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
            total_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f)) 
                for f in cache_files
            )
            
            return {
                'total_cached_responses': len(cache_files),
                'cache_size_mb': round(total_size / (1024 * 1024), 2),
                'cache_directory': self.cache_dir
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
