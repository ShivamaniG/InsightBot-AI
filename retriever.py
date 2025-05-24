import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load embedding model and FAISS index once
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('embeddings/faiss_index.idx')

# Load metadata
with open('embeddings/metadata.pkl', 'rb') as f:
    metadatas = pickle.load(f)

def retrieve(query, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(metadatas):
            results.append({
                "source": metadatas[idx].get("source", "unknown"),
                "text": metadatas[idx].get("text", "")
            })
    return results

if __name__ == "__main__":
    query = input("Enter your question: ")
    results = retrieve(query)
    print("\nTop results:")
    for res in results:
        print(f"Source: {res['source']}")
        print(f"Text snippet: {res['text']}\n")
