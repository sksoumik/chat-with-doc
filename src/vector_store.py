import logging
from pathlib import Path
from typing import List, Optional

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)

from .config import AppConfig
from .logger import setup_logger
from .models import ChatResponse, Document

logger = setup_logger(__name__)


class VectorStore:
    """Handles vector store operations."""

    def __init__(self, config: AppConfig):
        """
        Initialize VectorStore.

        Args:
            config (AppConfig): Application configuration
        """
        self.config = config
        logger.info("Initializing VectorStore")
        if not config.google_api_key:
            logger.error("Google API key not found")
            raise ValueError(
                "Google API key not found. Please set GOOGLE_API_KEY environment variable."
            )

        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=config.embedding_model, google_api_key=config.google_api_key
            )
            logger.info("Successfully initialized embeddings")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise

    def create_vector_store(self, chunks: List[str]) -> bool:
        """
        Create a vector store from text chunks.

        Args:
            chunks (List[str]): List of text chunks

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Creating vector store")
            vector_store = FAISS.from_texts(chunks, embedding=self.embeddings)
            vector_store.save_local(self.config.vector_store_path)
            return True
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            return False

    def get_response(self, question: str) -> Optional[ChatResponse]:
        """Get response for a question using the vector store."""
        try:
            logger.info(f"Processing question in VectorStore: {question}")

            vector_store_path = Path(self.config.vector_store_path)
            logger.info(f"Checking vector store at: {vector_store_path}")

            if not vector_store_path.exists():
                error_msg = f"Vector store not found at {vector_store_path}"
                logger.error(error_msg)
                return ChatResponse(output_text=error_msg)

            logger.info("Loading vector store...")
            vector_store = FAISS.load_local(
                self.config.vector_store_path, self.embeddings
            )
            logger.info("Vector store loaded successfully")

            logger.info("Performing similarity search...")
            docs = vector_store.similarity_search(question)
            logger.info(f"Found {len(docs)} relevant documents")

            if not docs:
                error_msg = "No relevant information found in the documents."
                logger.warning(error_msg)
                return ChatResponse(output_text=error_msg)

            logger.info("Creating QA chain...")
            chain = self._create_qa_chain()

            logger.info("Getting response from model...")
            response = chain(
                {"input_documents": docs, "question": question},
                return_only_outputs=True,
            )

            logger.info(f"Generated response: {response['output_text']}")
            return ChatResponse(output_text=response["output_text"])

        except Exception as e:
            error_msg = f"Error in VectorStore: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ChatResponse(output_text=error_msg)

    def _create_qa_chain(self):
        """
        Create a question-answering chain.

        Returns:
            Chain: The QA chain
        """
        prompt_template = """
        Answer the question as detailed as possible from the provided context, 
        make sure to provide all the details, if the answer is not in provided 
        context just say, "answer is not available in the context", don't 
        provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n
        Answer:
        """

        model = ChatGoogleGenerativeAI(
            model=self.config.llm_model, temperature=self.config.llm_temperature
        )

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        return load_qa_chain(model, chain_type="stuff", prompt=prompt)
