import streamlit as st
from collections import deque
from src.vector_store import get_vector_store, FAISS
from src.document_processing import get_pdf_text, get_text_chunks
from src.web_page_retrieval import get_url_content
from src.user_interface import user_input

def reset_input():
    st.session_state.user_input = ""


def main():
    st.set_page_config(page_title="Chat with PDFs and URLs", layout="wide")
    st.header("Chat with PDFs and Web pages using Gemini Pro")

    if 'history' not in st.session_state:
        st.session_state.history = deque(maxlen=10)

    with st.container():
        for message in reversed(st.session_state.history):
            st.text_area("", value=message, height=80, key=message, disabled=True)

    user_question = st.text_input("Ask a question", key="user_input", on_change=reset_input)

    if user_question:
        response = user_input(user_question)
        st.session_state.history.appendleft(f"Q: {user_question}")
        st.session_state.history.appendleft(f"A: {response}")

    with st.sidebar:
        st.header("Upload PDFs or enter URLs to process")
        st.markdown("---")

        try:
            pdf_docs = st.file_uploader("Upload your PDF files and Submit", accept_multiple_files=True)
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
                url_contents = []
                for url in urls.splitlines():
                    url_content = get_url_content(url)
                    url_contents.append(url_content)

                all_text_chunks = []
                for url_content in url_contents:
                    text_chunks = get_text_chunks(url_content)
                    all_text_chunks.extend(text_chunks)

                get_vector_store(all_text_chunks)
                st.success("URLs processed")

if __name__ == "__main__":
    main()
