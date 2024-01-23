import os

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

from src.document_processing import get_pdf_text, get_text_chunks
from src.user_interface import user_input
from src.vector_store import get_vector_store
from src.web_page_retrieval import get_url_content

load_dotenv()
# GOOGLE_API_KEY should be set in .env file
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def main():
    st.set_page_config("Chat with PDFs and URLs")
    st.header("Chat with PDFs and Web pages using Gemni Pro")

    user_question = st.text_input("Ask a question")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        # add a subheader in the sidebar
        st.header("Upload PDFs or enter URLs to process")
        # add a line break
        st.markdown("---")

        # Section for PDF uploads
        try:
            pdf_docs = st.file_uploader(
                "Upload your PDF files and Submit", accept_multiple_files=True
            )
            if st.button("Submit & Process PDFs"):
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDFs processed")
        except Exception as e:
            st.warning("Please upload a PDF file.")

        st.markdown("---")
        urls = st.text_area("Enter multiple URLs (one per line):")
        if st.button("Submit & Process URLs"):
            with st.spinner("Processing URLs..."):
                # Process multiple URLs
                url_contents = []
                for url in urls.splitlines():
                    url_content = get_url_content(url)
                    url_contents.append(url_content)

                # Combine text chunks from all URLs
                all_text_chunks = []
                for url_content in url_contents:
                    text_chunks = get_text_chunks(url_content)
                    all_text_chunks.extend(text_chunks)

                get_vector_store(all_text_chunks)
                st.success("URLs processed")


if __name__ == "__main__":
    main()
