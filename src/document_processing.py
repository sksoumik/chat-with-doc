"""Document processing utilities."""
from pathlib import Path
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

from .config import AppConfig
from .logger import setup_logger
from .models import Document

logger = setup_logger(__name__)


class DocumentProcessor:
    """Handles document processing operations."""

    def __init__(self, config: AppConfig):
        """
        Initialize DocumentProcessor.

        Args:
            config (AppConfig): Application configuration
        """
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )

    def process_pdf(self, pdf_file: Path) -> Optional[Document]:
        """
        Process a PDF file and extract its text content.

        Args:
            pdf_file (Path): Path to the PDF file

        Returns:
            Optional[Document]: Processed document or None if processing fails
        """
        try:
            logger.info(f"Processing PDF file: {pdf_file}")
            text = ""
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text()

            return Document(content=text, source=str(pdf_file))

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_file}: {str(e)}")
            return None

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Args:
            text (str): Input text to split

        Returns:
            List[str]: List of text chunks
        """
        try:
            logger.info("Splitting text into chunks")
            return self.text_splitter.split_text(text)
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            return []
