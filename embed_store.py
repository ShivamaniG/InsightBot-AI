import os
import faiss
import pickle
import hashlib
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

import pandas as pd
import fitz  # PyMuPDF

# Setup
DATA_DIR = 'data'
EMBEDDING_DIR = 'embeddings'
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Chunker
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

def get_file_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

for filename in os.listdir(DATA_DIR):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(filepath):
        continue

    file_hash = get_file_hash(filepath)
    index_path = os.path.join(EMBEDDING_DIR, f'{file_hash}_index.idx')
    meta_path = os.path.join(EMBEDDING_DIR, f'{file_hash}_meta.pkl')

    if os.path.exists(index_path) and os.path.exists(meta_path):
        print(f"Skipping {filename} — already embedded.")
        continue

    # Read content
    if filename.endswith(".txt") or filename.endswith(".md"):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    elif filename.endswith(".csv"):
        content = pd.read_csv(filepath).to_string()
    elif filename.endswith(".pdf"):
        doc = fitz.open(filepath)
        content = " ".join([page.get_text() for page in doc])
    else:
        continue

    # Split & embed
    chunks = splitter.split_text(content)
    texts = chunks
    metadatas = [{"source": filename, "text": chunk} for chunk in chunks]

    embeddings = model.encode(texts, show_progress_bar=True)
    dim = embeddings[0].shape[0]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(meta_path, 'wb') as f:
        pickle.dump(metadatas, f)

    print(f"Embedded and stored: {filename}")
