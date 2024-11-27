import logging
from typing import Optional

import streamlit as st

from .config import AppConfig
from .logger import setup_logger
from .models import ChatResponse
from .vector_store import VectorStore

logger = setup_logger(__name__)


class UserInterface:
    """Handles user interface operations."""

    def __init__(self, vector_store: VectorStore, config: AppConfig):
        """
        Initialize UserInterface.

        Args:
            vector_store (VectorStore): Vector store instance
            config (AppConfig): Application configuration
        """
        self.vector_store = vector_store
        self.config = config

    def process_question(self, question: str) -> Optional[str]:
        """Process a user question and get a response."""
        try:
            status_placeholder = st.empty()
            status_placeholder.info("Processing your question...")

            logger.info(f"Processing user question: {question}")

            if not question.strip():
                status_placeholder.warning("Empty question received")
                return "Please enter a question."

            logger.info("Calling vector store for response...")
            status_placeholder.info("Searching through documents...")

            response = self.vector_store.get_response(question)
            logger.info(f"Response received from vector store: {response}")

            if response and response.output_text:
                logger.info("Valid response received")
                status_placeholder.success("Response received!")

                with st.container():
                    st.markdown("**Response:**")
                    st.markdown(response.output_text)

                return response.output_text

            logger.warning("No valid response received")
            status_placeholder.warning("No response received")
            return "No response received. Please try again."

        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            logger.error(error_msg, exc_info=True)
            status_placeholder.error(error_msg)
            return error_msg

    @staticmethod
    def display_chat_history(history: list) -> None:
        """Display the chat history in the Streamlit interface."""
        try:
            for message in history:
                is_user = message.startswith("Q: ")

                with st.container():
                    if is_user:
                        st.info(f"👤 User: {message[3:]}")
                    else:
                        st.success(f"🤖 Assistant: {message[3:]}")
        except Exception as e:
            logger.error(f"Error displaying chat history: {str(e)}", exc_info=True)
            st.error("Error displaying chat history")
