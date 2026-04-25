import streamlit as st
from openai import OpenAI
import os
from rag_utils import load_documents_from_folder, get_relevant_context

st.set_page_config(layout="wide", page_title="OpenRouter chatbot app")
st.title("Gemini 3.1 Flash Lite - RAG Chatbot")

# Bezpieczniejsze pobieranie kluczy (najpierw st.secrets, potem os.environ jako fallback)
try:
    api_key = st.secrets.get("API_KEY", os.environ.get("API_KEY"))
    base_url = st.secrets.get("BASE_URL", os.environ.get("BASE_URL"))
except FileNotFoundError:
    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("BASE_URL")

selected_model = "gemini-3.1-flash-lite-preview"

# --- RAG INITIALIZATION ---
# Używamy st.cache_data, aby wczytać PDFy tylko raz (przy starcie apki)
@st.cache_data
def init_knowledge_base():
    # Zdefiniuj folder na dokumenty (możesz go zmienić)
    folder_path = "data" 
    return load_documents_from_folder(folder_path)

# Wczytanie bazy wiedzy z folderu 'data'
documents = init_knowledge_base()
if not documents:
    st.warning("Baza wiedzy jest pusta. Dodaj pliki PDF do folderu 'data'.")
else:
    st.success(f"Wczytano {len(documents)} fragmentów tekstu z bazy wiedzy.")
# --------------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not api_key:
        st.info("Invalid API key.")
        st.stop()
        
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Dodajemy wiadomość użytkownika do UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # --- RAG IN ACTION ---
    # Szukamy kontekstu w naszych dokumentach na podstawie tego co napisał użytkownik
    context = get_relevant_context(prompt, documents)
    
    # Tworzymy kopię wiadomości dla API (aby nie psuć widoku w UI, gdzie nie chcemy wyświetlać wstrzykniętego tekstu systemowego za każdym razem)
    api_messages = st.session_state.messages.copy()
    
    if context:
        # Modyfikujemy OSTATNIĄ wiadomość wysyłaną do modelu, wstrzykując kontekst
        augmented_prompt = f"Odpowiedz na poniższe pytanie bazując na dostarczonym kontekście.\n\nKONTEKST:\n{context}\n\nPYTANIE UŻYTKOWNIKA:\n{prompt}"
        api_messages[-1] = {"role": "user", "content": augmented_prompt}
    # ---------------------

    # Wywołanie modelu
    response = client.chat.completions.create(
        model=selected_model,
        messages=api_messages # wysyłamy wiadomości wzbogacone o RAG
    )
    
    msg = response.choices[0].message.content
    
    # Zapis i wyświetlenie odpowiedzi asystenta
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
