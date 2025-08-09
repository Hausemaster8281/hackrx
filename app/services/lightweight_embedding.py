import httpx
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)

class LightweightEmbeddingService:
    """Lightweight embedding service using external APIs"""
    
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        self.headers = {}  # Add HF token if needed
    
    async def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings using Hugging Face API"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    headers=self.headers,
                    json={"inputs": texts},
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    embeddings = np.array(response.json())
                    return embeddings
                else:
                    logger.error(f"API error: {response.status_code}")
                    # Fallback to simple embeddings
                    return self._create_simple_embeddings(texts)
                    
        except Exception as e:
            logger.error(f"Embedding API error: {e}")
            return self._create_simple_embeddings(texts)
    
    def _create_simple_embeddings(self, texts: List[str]) -> np.ndarray:
        """Simple hash-based embeddings as fallback"""
        embeddings = []
        for text in texts:
            # Simple hash-based embedding (384 dimensions)
            hash_val = hash(text.lower())
            embedding = np.array([
                (hash_val >> i) & 1 for i in range(384)
            ], dtype=np.float32)
            embeddings.append(embedding)
        return np.array(embeddings)
