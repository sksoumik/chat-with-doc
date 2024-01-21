from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader


# This function takes a list of PDF documents as input and extracts the text
# from each page of each document, concatenating them into a single string.
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# This function takes a long text as input and splits it into smaller chunks of text.
# It uses a RecursiveCharacterTextSplitter with a chunk size of 10000 and an overlap of 1000.
# The chunk size is the number of characters in each chunk, and the overlap is the number of
# characters that each chunk overlaps with the next one.
# For example, if the text is 100,000 characters long, the function will return a list of 10
# chunks, each 10,000 characters long, and each chunk will overlap with the next one by 1,000
# characters.
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks
