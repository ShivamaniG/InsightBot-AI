from transformers import pipeline
from typing import List, Dict

# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Summarize multiple text chunks
def summarize_texts(text_chunks: List[Dict[str, str]], max_words=1000) -> str:
    full_text = "\n".join([chunk["text"] for chunk in text_chunks])
    summaries = []

    for i in range(0, len(full_text), max_words):
        chunk = full_text[i:i+max_words]
        summary = summarizer(chunk, max_length=100, min_length=40, do_sample=False)
        summaries.append(summary[0]["summary_text"])

    return "\n\n".join(summaries)
