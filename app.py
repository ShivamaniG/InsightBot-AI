import os
import faiss
import pickle
import pandas as pd
import fitz
from io import StringIO
from docx import Document
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient
import streamlit as st

# Constants
DATA_DIR = "data"
EMBED_DIR = "embeddings"
HF_TOKEN = "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"

# Setup
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EMBED_DIR, exist_ok=True)

# Load models once
model = SentenceTransformer("all-MiniLM-L6-v2")
client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token=HF_TOKEN)

# UI
st.set_page_config("InsightBot End-to-End", layout="wide")
st.title("📂 InsightBot Document Chat")
uploaded_file = st.file_uploader("Upload file", type=["pdf", "txt", "docx", "csv"])

query = st.text_input("Ask a question based on document:")
if st.button("Process & Ask"):
    if not uploaded_file:
        st.error("Please upload a document first.")
    else:
        # Save uploaded file
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Read content
        if uploaded_file.name.endswith(".pdf"):
            reader = PdfReader(file_path)
            content = "\n".join([page.extract_text() for page in reader.pages])
        elif uploaded_file.name.endswith(".docx"):
            doc = Document(file_path)
            content = "\n".join([p.text for p in doc.paragraphs])
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(file_path)
            content = df.to_string()
        elif uploaded_file.name.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            st.error("Unsupported file type")
            st.stop()

        # Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(content)
        metadatas = [{"source": uploaded_file.name, "text": c} for c in chunks]

        # Embeddings
        embeddings = model.encode(chunks, show_progress_bar=True)
        dim = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        # Save for future (optional)
        faiss.write_index(index, os.path.join(EMBED_DIR, "faiss_index.idx"))
        with open(os.path.join(EMBED_DIR, "metadata.pkl"), "wb") as f:
            pickle.dump(metadatas, f)

        # Retrieve
        query_embedding = model.encode([query])
        distances, indices = index.search(query_embedding, k=5)
        retrieved = [metadatas[i] for i in indices[0] if i < len(metadatas)]

        # LLM response
        context = "\n".join(chunk["text"] for chunk in retrieved)
        messages = [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"}]
        response = client.chat_completion(messages=messages, max_tokens=500, stream=False).choices[0].message.content.strip()

        # Show output
        st.subheader("🔍 Answer:")
        st.success(response)

        st.subheader("📄 Retrieved Chunks")
        for i, c in enumerate(retrieved):
            with st.expander(f"Chunk {i+1}"):
                st.text(c["text"])
