import sys
from collections import deque
from pathlib import Path
from typing import List

import streamlit as st

from src.config import AppConfig
from src.debug_config import DEBUG_CONFIG
from src.document_processing import DocumentProcessor
from src.logger import setup_logger
from src.user_interface import UserInterface
from src.vector_store import VectorStore
from src.web_page_retrieval import WebPageRetriever

logger = setup_logger(__name__)

if DEBUG_CONFIG["development_mode"]:
    logger.debug("Running in development mode")
    logger.debug(f"Python version: {sys.version}")
    logger.debug(f"Python path: {sys.path}")


def reset_input() -> None:
    """Reset the user input field."""
    st.session_state.user_input = ""


def initialize_session_state() -> None:
    """Initialize session state variables."""
    if "history" not in st.session_state:
        st.session_state.history = deque(maxlen=10)


def main() -> None:
    """Main application entry point."""
    try:
        config = AppConfig()
        doc_processor = DocumentProcessor(config)
        vector_store = VectorStore(config)
        web_retriever = WebPageRetriever()
        ui = UserInterface(vector_store, config)

        st.set_page_config(page_title=config.page_title, layout="wide")
        st.header(config.header_text)

        initialize_session_state()

        chat_col, doc_col = st.columns([2, 1])

        with chat_col:
            st.markdown("### Chat Interface")

            if st.session_state.history:
                ui.display_chat_history(list(st.session_state.history))

            user_question = st.text_input(
                "Ask a question about your documents",
                key="user_input",
                on_change=reset_input,
            )

            if user_question:
                try:
                    logger.info(f"Processing question: {user_question}")

                    if not Path(config.vector_store_path).exists():
                        logger.warning("No vector store found")
                        st.warning("Please upload and process some documents first.")
                        return

                    question_container = st.container()
                    response_container = st.container()

                    with question_container:
                        st.info(f"👤 User: {user_question}")

                    with response_container:
                        with st.spinner("Getting response..."):
                            response = ui.process_question(user_question)

                            if response:
                                logger.info("Response received, updating history")
                                st.session_state.history.appendleft(
                                    f"Q: {user_question}"
                                )
                                st.session_state.history.appendleft(f"A: {response}")

                                st.experimental_rerun()
                            else:
                                st.error("Failed to get response")

                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}", exc_info=True)
                    st.error(f"An error occurred: {str(e)}")

        with doc_col:
            st.subheader("Document Processing")

            pdf_docs = st.file_uploader(
                "Upload your PDF files", accept_multiple_files=True, type=["pdf"]
            )

            if st.button("Process PDFs", key="pdf_button"):
                with st.spinner("Processing PDFs..."):
                    processed = False
                    for pdf in pdf_docs:
                        document = doc_processor.process_pdf(pdf)
                        if document:
                            chunks = doc_processor.chunk_text(document.content)
                            if vector_store.create_vector_store(chunks):
                                processed = True

                    if processed:
                        st.success("PDFs processed successfully!")
                    else:
                        st.warning("No PDFs were processed.")

            st.markdown("---")
            urls = st.text_area(
                "Enter URLs (one per line)",
                help="Enter web page URLs to process their content",
            )

            if st.button("Process URLs", key="url_button"):
                with st.spinner("Processing URLs..."):
                    all_chunks: List[str] = []
                    processed = False

                    for url in urls.splitlines():
                        if url.strip():
                            document = web_retriever.get_content(url)
                            if document:
                                chunks = doc_processor.chunk_text(document.content)
                                all_chunks.extend(chunks)
                                processed = True

                    if processed and vector_store.create_vector_store(all_chunks):
                        st.success("URLs processed successfully!")
                    else:
                        st.warning("No URLs were processed.")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"An error occurred: {str(e)}\nPlease check the logs for details.")


if __name__ == "__main__":
    main()
