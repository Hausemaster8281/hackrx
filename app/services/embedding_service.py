from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from typing import List, Optional, Dict, Any
import logging
import pickle
import os

from ..models.schemas import DocumentChunk, EmbeddingResult, SearchResult
from ..core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for creating and managing document embeddings"""
    
    def __init__(self):
        """Initialize the embedding service"""
        self.model_name = settings.embedding_model
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.max_search_results = settings.max_search_results
        
        # Initialize without loading the model immediately
        self.model = None
        self.dimension = 384  # Default dimension for all-MiniLM-L6-v2
        
        # Initialize FAISS index and chunk tracking
        self.index = None
        self.chunks = []
        self.chunk_map = {}  # Maps chunk indices to chunk objects
        self._initialize_index()
    
    def _load_model(self):
        """Load the sentence transformer model lazily"""
        if self.model is None:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                self.dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"Model loaded successfully with dimension: {self.dimension}")
                # Recreate index with correct dimension
                self._initialize_index()
            except Exception as e:
                logger.error(f"Failed to load embedding model {self.model_name}: {e}")
                raise
        return self.model
    
    def _old_load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise Exception(f"Failed to load embedding model: {str(e)}")
    
    def _initialize_index(self):
        """Initialize FAISS index for vector similarity search"""
        try:
            # Create a new FAISS index
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            logger.info(f"Initialized FAISS index with dimension {self.dimension}")
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {str(e)}")
            raise Exception(f"Failed to initialize FAISS index: {str(e)}")
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts"""
        try:
            # Load model if not already loaded
            model = self._load_model()
            
            logger.info(f"Creating embeddings for {len(texts)} texts")
            embeddings = model.encode(texts, normalize_embeddings=True)
            logger.info(f"Created embeddings with shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise Exception(f"Failed to create embeddings: {str(e)}")
    
    def add_chunks_to_index(self, chunks: List[DocumentChunk]) -> List[EmbeddingResult]:
        """Add document chunks to the FAISS index"""
        try:
            logger.info(f"Adding {len(chunks)} chunks to index")
            
            # Extract text content from chunks
            texts = [chunk.content for chunk in chunks]
            
            # Create embeddings
            embeddings = self.create_embeddings(texts)
            
            # Add to FAISS index
            start_id = self.index.ntotal
            self.index.add(embeddings.astype('float32'))
            
            # Update chunk map
            embedding_results = []
            for i, chunk in enumerate(chunks):
                chunk_index_id = start_id + i
                self.chunk_map[chunk_index_id] = chunk
                
                embedding_results.append(EmbeddingResult(
                    chunk_id=chunk.chunk_id,
                    embedding=embeddings[i].tolist()
                ))
            
            logger.info(f"Successfully added {len(chunks)} chunks to index. Total: {self.index.ntotal}")
            return embedding_results
            
        except Exception as e:
            logger.error(f"Error adding chunks to index: {str(e)}")
            raise Exception(f"Failed to add chunks to index: {str(e)}")
    
    def search_similar_chunks(self, query: str, top_k: int = None) -> List[SearchResult]:
        """Search for similar chunks using the query"""
        try:
            top_k = top_k or self.max_search_results
            
            if self.index.ntotal == 0:
                logger.warning("No chunks in index for search")
                return []
            
            logger.info(f"Searching for similar chunks with query: {query[:100]}...")
            
            # Create embedding for query
            query_embedding = self.create_embeddings([query])
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # Create search results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx in self.chunk_map and score >= 0.3:  # similarity threshold
                    chunk = self.chunk_map[idx]
                    results.append(SearchResult(
                        chunk=chunk,
                        score=float(score),
                        rank=i + 1
                    ))
            
            logger.info(f"Found {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            logger.error(f"Error searching for similar chunks: {str(e)}")
            raise Exception(f"Failed to search similar chunks: {str(e)}")
    
    def get_relevant_context(self, query: str, top_k: int = None) -> str:
        """Get relevant context for a query by combining top search results"""
        try:
            search_results = self.search_similar_chunks(query, top_k)
            
            if not search_results:
                return ""
            
            # Combine the content from top results
            context_parts = []
            for i, result in enumerate(search_results):
                context_parts.append(f"[Context {i+1}]: {result.chunk.content}")
            
            context = "\n\n".join(context_parts)
            logger.info(f"Retrieved context of length: {len(context)} characters")
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            return ""
    
    def clear_index(self):
        """Clear the current index and chunk map"""
        try:
            self._initialize_index()
            self.chunk_map.clear()
            logger.info("Index and chunk map cleared")
        except Exception as e:
            logger.error(f"Error clearing index: {str(e)}")
    
    def save_index(self, filepath: str):
        """Save the current index and chunk map to file"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save chunk map
            with open(f"{filepath}.chunks", 'wb') as f:
                pickle.dump(self.chunk_map, f)
            
            logger.info(f"Index saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
    
    def load_index(self, filepath: str):
        """Load index and chunk map from file"""
        try:
            # Load FAISS index
            if os.path.exists(f"{filepath}.faiss"):
                self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load chunk map
            if os.path.exists(f"{filepath}.chunks"):
                with open(f"{filepath}.chunks", 'rb') as f:
                    self.chunk_map = pickle.load(f)
            
            logger.info(f"Index loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
