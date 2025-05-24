from retriever import retrieve  # your FAISS-based retriever
from summarization import summarize_texts

if __name__ == "__main__":
    query = input("Enter a topic to summarize: ")
    chunks = retrieve(query, top_k=10)
    summary = summarize_texts(chunks)
    print("\n--- Summary ---\n", summary)
