"""Configuration settings for the application."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    """Application configuration settings."""

    page_title: str = "Chat with PDFs and URLs"
    header_text: str = "Chat with PDFs and Web pages using Gemini Pro"
    history_maxlen: int = 10
    chunk_size: int = 10000
    chunk_overlap: int = 1000
    embedding_model: str = "models/embedding-001"
    llm_model: str = "gemini-pro"
    llm_temperature: float = 0.4
    vector_store_path: str = "faiss_index"
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
