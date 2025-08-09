import httpx
import PyPDF2
import docx
from typing import List, Optional, Dict, Any
from io import BytesIO
import logging
import re
from urllib.parse import urlparse

from ..models.schemas import DocumentChunk

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Service for processing and extracting text from documents"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.doc']
        self.chunk_size = 1000
        self.chunk_overlap = 200
    
    async def download_document(self, url: str) -> bytes:
        """Download document from URL"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.content
        except Exception as e:
            logger.error(f"Error downloading document from {url}: {str(e)}")
            raise Exception(f"Failed to download document: {str(e)}")
    
    def extract_text_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            pdf_file = BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def extract_text_from_docx(self, content: bytes) -> str:
        """Extract text from DOCX content"""
        try:
            docx_file = BytesIO(content)
            doc = docx.Document(docx_file)
            
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {str(e)}")
            raise Exception(f"Failed to extract text from DOCX: {str(e)}")
    
    def detect_document_format(self, url: str, content: bytes) -> str:
        """Detect document format from URL and content"""
        parsed_url = urlparse(url)
        
        # Try to get format from URL
        if parsed_url.path.lower().endswith('.pdf'):
            return 'pdf'
        elif parsed_url.path.lower().endswith(('.docx', '.doc')):
            return 'docx'
        
        # Try to detect from content
        if content.startswith(b'%PDF'):
            return 'pdf'
        elif b'PK' in content[:4]:  # DOCX files are ZIP archives
            return 'docx'
        
        # Default to PDF if uncertain
        return 'pdf'
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
        
        # Remove page break indicators
        text = re.sub(r'--- Page \d+ ---', '', text)
        
        return text.strip()
    
    def create_chunks(self, text: str, chunk_size: int = None, overlap: int = None) -> List[DocumentChunk]:
        """Split text into chunks for processing"""
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap
        
        chunks = []
        words = text.split()
        
        if len(words) <= chunk_size:
            # If text is small enough, return as single chunk
            chunks.append(DocumentChunk(
                content=text,
                chunk_id="chunk_0",
                start_char=0,
                end_char=len(text),
                metadata={"word_count": len(words)}
            ))
            return chunks
        
        # Create overlapping chunks
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append(DocumentChunk(
                content=chunk_text,
                chunk_id=f"chunk_{len(chunks)}",
                start_char=i,
                end_char=i + len(chunk_words),
                metadata={
                    "word_count": len(chunk_words),
                    "start_word_index": i,
                    "end_word_index": i + len(chunk_words)
                }
            ))
        
        return chunks
    
    async def process_document(self, url: str) -> List[DocumentChunk]:
        """Main method to process document from URL to chunks"""
        try:
            logger.info(f"Processing document from URL: {url}")
            
            # Download document
            content = await self.download_document(url)
            
            # Detect format
            doc_format = self.detect_document_format(url, content)
            logger.info(f"Detected document format: {doc_format}")
            
            # Extract text based on format
            if doc_format == 'pdf':
                text = self.extract_text_from_pdf(content)
            elif doc_format == 'docx':
                text = self.extract_text_from_docx(content)
            else:
                raise Exception(f"Unsupported document format: {doc_format}")
            
            # Clean text
            text = self.clean_text(text)
            logger.info(f"Extracted text length: {len(text)} characters")
            
            # Create chunks
            chunks = self.create_chunks(text)
            logger.info(f"Created {len(chunks)} chunks")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise Exception(f"Document processing failed: {str(e)}")
