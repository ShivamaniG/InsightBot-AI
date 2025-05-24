import os
import faiss
import pickle
import pandas as pd
from docx import Document
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient

DATA_DIR = "data"
EMBED_DIR = "embeddings"
HF_TOKEN = "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"

# Initialize models once (will be reused)
model = SentenceTransformer("all-MiniLM-L6-v2")
client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token=HF_TOKEN)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

def insightbot_process(file_bytes, filename, query, save_embeddings=True):
    """
    Full pipeline to:
    - read document bytes (pdf, docx, csv, txt),
    - chunk text,
    - embed,
    - search top-k chunks,
    - ask LLM with retrieved context,
    - return answer and retrieved chunks.
    
    Args:
      file_bytes (bytes): file content bytes
      filename (str): original filename (with extension)
      query (str): question string
      save_embeddings (bool): whether to save index and metadata on disk

    Returns:
      answer (str): LLM answer
      retrieved_chunks (list of dict): retrieved metadata chunks
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(EMBED_DIR, exist_ok=True)

    # Save temp file
    file_path = os.path.join(DATA_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    # Read content
    if filename.endswith(".pdf"):
        reader = PdfReader(file_path)
        content = "\n".join([page.extract_text() or "" for page in reader.pages])
    elif filename.endswith(".docx"):
        doc = Document(file_path)
        content = "\n".join([p.text for p in doc.paragraphs])
    elif filename.endswith(".csv"):
        df = pd.read_csv(file_path)
        content = df.to_string()
    elif filename.endswith(".txt"):
        content = file_bytes.decode("utf-8")
    else:
        raise ValueError("Unsupported file type")

    # Chunk text
    chunks = splitter.split_text(content)
    metadatas = [{"source": filename, "text": chunk} for chunk in chunks]

    # Embed chunks
    embeddings = model.encode(chunks, show_progress_bar=False)
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Optionally save embeddings
    if save_embeddings:
        faiss.write_index(index, os.path.join(EMBED_DIR, "faiss_index.idx"))
        with open(os.path.join(EMBED_DIR, "metadata.pkl"), "wb") as f:
            pickle.dump(metadatas, f)

    # Embed query and search
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=5)
    retrieved = [metadatas[i] for i in indices[0] if i < len(metadatas)]

    # Prepare LLM prompt with context and question
    context = "\n".join(chunk["text"] for chunk in retrieved)
    messages = [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"}]
    response = client.chat_completion(messages=messages, max_tokens=500, stream=False).choices[0].message.content.strip()
    prompt = f"""
    You are an assistant that extracts useful insights from a document.

    Based on the following context, provide:
    1. Key Points
    2. Major Insights
    3. Action Items

    Respond in valid JSON format only. Do not include markdown or code block formatting.

    Context:
    {context}

    Format:
    {{
    "Key Points": [...],
    "Major Insights": [...],
    "Action Items": [...]
    }}
    """

    messages = [{"role": "user", "content": prompt}]
    insight_response = client.chat_completion(messages=messages, max_tokens=500, stream=False)
    content = insight_response.choices[0].message.content.strip()

    import json
    try:
        insights = json.loads(content)
    except Exception:
        insights = {
            "Key Points": [content],
            "Major Insights": [content],
            "Action Items": [content]
        }

    return response, retrieved, insights

# def generate_insight_cards(context_text: str) -> dict:

#     prompt = f"""
#     You are an assistant that extracts useful insights from a document.

#     Based on the following context, provide the following:
#     1. Key Points (short bullet points)
#     2. Major Insights (important conclusions or takeaways)
#     3. Action Items (suggested next steps or recommendations)

#     Context:
#     {context_text}

#     Format your response as JSON:
#     {{
#       "Key Points": [...],
#       "Major Insights": [...],
#       "Action Items": [...]
#     }}
#     """

#     messages = [{"role": "user", "content": prompt}]
#     response = client.chat_completion(messages=messages, max_tokens=500, stream=False)
#     content = response.choices[0].message.content.strip()

#     # Try to parse JSON response
#     import json
#     try:
#         insights = json.loads(content)
#     except Exception:
#         # fallback: return raw text in all 3 fields
#         insights = {
#             "Key Points": [content],
#             "Major Insights": [content],
#             "Action Items": [content]
#         }
#     return insights
