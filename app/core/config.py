"""
Configuration settings for the HackRx application
"""
from pydantic_settings import BaseSettings
from pydantic import Field
import os
from typing import Optional


class Settings(BaseSettings):
    """Application configuration settings"""
    
    # API Configuration
    api_title: str = Field(default="HackRx LLM Query Retrieval System", env="API_TITLE")
    api_description: str = Field(default="LLM-Powered Intelligent Query-Retrieval System for HackRx 6.0", env="API_DESCRIPTION")
    api_version: str = Field(default="1.0.0", env="API_VERSION")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Authentication
    api_token: str = Field(default="93f02c19721b0c40d273b9395370f249d43c2d3deb2a39498fdd8a2b9e2d3ee3", env="API_TOKEN")
    
    # Gemini Configuration
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", env="GEMINI_MODEL")
    
    # Embedding Configuration
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    chunk_size: int = Field(default=400, env="CHUNK_SIZE")  # Reduced from 500
    chunk_overlap: int = Field(default=100, env="CHUNK_OVERLAP")  # Increased from 50
    
    # FAISS Configuration
    faiss_index_path: str = Field(default="./data/faiss_index", env="FAISS_INDEX_PATH")
    max_search_results: int = Field(default=10, env="MAX_SEARCH_RESULTS")  # Increased from 5
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Document Processing
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    supported_formats: str = Field(default="pdf,docx", env="SUPPORTED_FORMATS")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
