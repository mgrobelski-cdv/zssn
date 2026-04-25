# rag_utils.py
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Inicjalizacja modelu embeddingów (darmowy, działa lokalnie)
# Model 'all-MiniLM-L6-v2' jest mały, szybki i wystarczający do prostych zadań
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_pdf(file):
    """Wczytuje tekst z jednego pliku PDF (obsługuje pliki przeładowane przez streamlit)"""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def load_documents_from_folder(folder_path):
    """Wczytuje wszystkie pliki PDF z podanego folderu"""
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            with open(path, "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    all_text += page.extract_text() + "\n"
    return all_text

def create_vector_store(text):
    """Dzieli tekst na kawałki i tworzy bazę wektorową FAISS"""
    # 1. Dzielenie tekstu (Chunking)
    # chunk_size: długość kawałka tekstu, chunk_overlap: część wspólna między kawałkami
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    
    # 2. Tworzenie bazy wektorowej z embeddingami
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def get_relevant_context(query, vector_store):
    """Szuka w bazie wektorowej fragmentów najbardziej pasujących do pytania"""
    docs = vector_store.similarity_search(query, k=3) # Pobierz 3 najbardziej pasujące fragmenty
    context = "\n\n".join([doc.page_content for doc in docs])
    return context
