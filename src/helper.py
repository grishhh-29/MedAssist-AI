from typing import List
# from langchain.embeddings import HuggingFaceEmbeddings
import os

# Document loaders
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# Text splitter
# from langchain.text_splitter.recursive import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Schema
# from langchain.schema import Document
from langchain_core.documents import Document

# Gemini embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time
from tenacity import retry, stop_after_attempt, wait_exponential




# Extract text from PDF files
def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    
    documents = loader.load()
    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects containing only 'source'
    in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content = doc.page_content,
                metadata = {"source": src}
            )
        )
    return minimal_docs

# Split the documents into smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk

# def download_hugging_face_embeddings():
    # """
    # Download and return the HuggingFace embeddings model.
    # """
    # model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # embeddings = HuggingFaceEmbeddings(
    #     model_name = model_name
    # )
    # return embeddings

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.environ["GEMINI_API_KEY"]
    )
    
