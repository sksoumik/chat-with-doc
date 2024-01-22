import os
import requests
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

from src.helper import get_pdf_text, get_text_chunks
from src.vector_db import get_vector_store, user_input

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# def chat_with_image(image_url, question):
#     llm = ChatGoogleGenerativeAI(model="gemini-pro-vision", temperature=0.4)
#     message = HumanMessage(
#     content=[
#         {
#             "type": "text",
#             "text": question,
#         },  # You can optionally provide text parts
#         {
#             "type": "image_url",
#             "image_url": image_url
#          },
#     ]
# )

#     context = llm.invoke([message])
#     print(context)
#     return context


def main():
    st.set_page_config("Chat with PDFs and URLs")
    st.header("Chat with PDFs and URLs using Gemni Pro")

    user_question = st.text_input("Ask a question")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        # add a subheader in the sidebar
        st.header("Upload PDFs or enter URLs to process")
        # add a line break
        st.markdown("---")

        # Section for PDF uploads
        pdf_docs = st.file_uploader(
            "Upload your PDF files and Submit", accept_multiple_files=True
        )
        if st.button("Submit & Process PDFs"):
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDFs processed")

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
        




def get_url_content(url):
    """Fetches the text content from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for error responses
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching content from URL: {url}")
        print(e)
        return None



if __name__ == "__main__":
    main()
