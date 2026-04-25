import os
from pypdf import PdfReader
import re

def load_pdf(file_path: str) -> str:
    """
    Wczytuje i zwraca tekst z pojedynczego pliku PDF.
    """
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    except Exception as e:
        print(f"Błąd podczas czytania pliku {file_path}: {e}")
    return text

def load_documents_from_folder(folder_path: str, chunk_size: int = 1500, overlap: int = 200) -> list[dict]:
    """
    Przeszukuje folder, wczytuje wszystkie PDFy i dzieli ich zawartość na fragmenty (chunks).
    Zwraca listę słowników z metadanymi i treścią.
    """
    documents = []
    
    # Sprawdzamy, czy folder istnieje, jeśli nie - tworzymy go
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Utworzono folder '{folder_path}'. Wrzuć do niego swoje pliki PDF.")
        return documents

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            full_text = load_pdf(file_path)
            
            # Prosty chunking (dzielenie na nachodzące na siebie fragmenty)
            # Pomaga to zachować kontekst między cięciami
            start = 0
            chunk_id = 0
            while start < len(full_text):
                end = start + chunk_size
                chunk = full_text[start:end]
                
                documents.append({
                    "filename": filename,
                    "chunk_id": chunk_id,
                    "content": chunk
                })
                
                start += (chunk_size - overlap)
                chunk_id += 1
                
    return documents

def get_relevant_context(query: str, documents: list[dict], top_k: int = 3) -> str:
    """
    Wyszukuje najbardziej pasujące fragmenty tekstu do zapytania.
    (Prosty RAG bez użycia zewnętrznych modeli embeddings - oparty na słowach kluczowych).
    """
    if not documents:
        return ""

    # Czyszczenie i podział zapytania na słowa (bazowe słowa kluczowe)
    query_words = set(re.findall(r'\w+', query.lower()))
    if not query_words:
        return ""

    scored_docs = []
    for doc in documents:
        # Podział tekstu dokumentu na słowa
        doc_words = set(re.findall(r'\w+', doc["content"].lower()))
        # Liczymy ile słów kluczowych z zapytania występuje w danym fragmencie (przecięcie zbiorów)
        score = len(query_words.intersection(doc_words))
        scored_docs.append((score, doc))
        
    # Sortujemy malejąco po wyniku
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Wybieramy najlepsze fragmenty (o ile mają wynik > 0)
    top_docs = [doc for score, doc in scored_docs[:top_k] if score > 0]
    
    # Złączamy znalezione fragmenty w jeden tekst, który trafi do modelu
    if not top_docs:
        return ""
        
    context = "Znalazłem następujące informacje w dokumentach bazy wiedzy:\n\n"
    for doc in top_docs:
        context += f"--- Fragment z pliku: {doc['filename']} ---\n{doc['content']}\n\n"
        
    return context
