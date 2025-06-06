# 🧠 InsightBot

InsightBot is a powerful AI assistant that implements a **RAG (Retrieval-Augmented Generation) pipeline** combined with LLM (Large Language Model) response integration. It enables users to upload documents (CSV, TXT, PDF), interact via chat to get document-related answers, and extract meaningful insights.

InsightBot uses a Hugging Face Transformers API model to generate context-aware, accurate responses from document data.

## ✨ Features

- **File Upload & Chat:** Upload CSV, TXT, and PDF files and ask questions to get accurate, context-aware answers from the document.
- **RAG (Retrieval-Augmented Generation) pipeline:** Consists of retrieval and generation. First, it embeds the input query and retrieves the most relevant chunks from a document store using similarity search (like FAISS). These chunks, along with the original query, are passed to a LLM (Large Language Model) which generates a final, context-aware response. This makes answers more accurate and grounded in the
- **Insight Cards:** Extracts and displays:
  - Key Points
  - Major Insights
  - Action Items
- **Summarization Panel:** Summarizes document content or any user-provided text/topic.
- **Document Preview:** Preview uploaded documents directly in the app, including PDF page previews as images.
- **CSV Visualization:** (Work in progress) – plans to visualize CSV content for better analysis.

## 🚀 Getting Started

1. Upload your document (CSV, TXT, PDF).
2. Enter a question related to the document and ask InsightBot.
3. View the answer, related document excerpts, and insights.
4. Use the Summarization Panel to get concise summaries.
5. Preview uploaded documents within the app.
