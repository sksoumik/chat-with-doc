from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader


def get_pdf_text(pdf_docs: List[str]) -> str:
    """Extracts text from PDFs."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text: str) -> List[str]:
    """Splits text into chunks of 10,000 characters with 1,000 character overlap."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks
