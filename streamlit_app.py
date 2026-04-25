import streamlit as st
from openai import OpenAI
import os
from rag_utils import load_pdf, load_documents_from_folder, create_vector_store, get_relevant_context

st.set_page_config(layout="wide", page_title="OpenRouter RAG chatbot")
st.title("Gemini 3.1 Flash Lite RAG app")

# API Config
api_key, base_url = st.secrets["API_KEY"], st.secrets["BASE_URL"]
selected_model = "gemini-3.1-flash-lite-preview"

# Inicjalizacja stanów sesji
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you? Upload a PDF to give me more knowledge!"}]
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

# --- SIDEBAR DLA RAG ---
with st.sidebar:
    st.header("Knowledge Base")
    
    # Opcja 1: Wgrywanie plików PDF przez interfejs
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    # Opcja 2: Wczytanie z folderu (np. jeśli masz folder 'data' w repo)
    folder_path = st.text_input("Or enter folder path:", "data/") 
    if st.button("Load from folder"):
        if os.path.exists(folder_path):
            with st.spinner("Loading folder..."):
                text = load_documents_from_folder(folder_path)
                st.session_state["vector_store"] = create_vector_store(text)
                st.success("Folder loaded!")
        else:
            st.error("Folder not found.")

    if uploaded_files:
        if st.button("Process Uploaded PDFs"):
            with st.spinner("Processing..."):
                combined_text = ""
                for file in uploaded_files:
                    combined_text += load_pdf(file) + "\n"
                st.session_state["vector_store"] = create_vector_store(combined_text)
                st.success("PDFs processed!")

    if st.button("Clear Knowledge"):
        st.session_state["vector_store"] = None
        st.rerun()

# --- CZAT ---
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not api_key:
        st.info("Invalid API key.")
        st.stop()
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # RAG LOGIC: Pobieranie kontekstu
    context = ""
    if st.session_state["vector_store"]:
        context = get_relevant_context(prompt, st.session_state["vector_store"])
    
    # Budowanie wzbogaconego promptu
    full_prompt = prompt
    if context:
        full_prompt = f"Use the following context to answer the question. If the answer is not in the context, use your general knowledge but mention that it's not in the documents.\n\nContext:\n{context}\n\nQuestion: {prompt}"

    # Wywołanie LLM (przesyłamy tylko ostatnią wersję z kontekstem, 
    # ale do historii sesji dopisujemy czysty prompt użytkownika)
    messages_for_api = st.session_state.messages.copy()
    messages_for_api[-1]["content"] = full_prompt # podmieniamy ostatni prompt na ten z kontekstem

    response = client.chat.completions.create(
        model=selected_model,
        messages=messages_for_api
    )
    
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
