from huggingface_hub import InferenceClient
from retriever import retrieve  # your retrieval function file, e.g. retrieval.py

# Initialize Hugging Face LLM client (replace token with your own)
HF_TOKEN = "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"
client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token=HF_TOKEN)

def generate_response(query, retrieved_chunks):
    """
    Generate an answer grounded in retrieved chunks plus user query.
    """
    # Combine retrieved texts as context
    context = "\n".join(chunk["text"] for chunk in retrieved_chunks)
    
    # Prepare messages for LLM chat
    messages = [
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {query}\n"
                "Answer:"
            )
        }
    ]
    
    # Get response from the LLM
    response_content = client.chat_completion(messages=messages, max_tokens=500, stream=False)
    response = response_content.choices[0].message.content.strip()
    
    # If response too short or empty, fallback to answer without context
    if not response or len(response.split()) < 10:
        messages = [{"role": "user", "content": query}]
        response_content = client.chat_completion(messages=messages, max_tokens=500, stream=False)
        response = response_content.choices[0].message.content.strip()
    
    return response

if __name__ == "__main__":
    user_query = input("Enter your question: ")
    retrieved = retrieve(user_query)
    answer = generate_response(user_query, retrieved)
    print("\nAnswer:")
    print(answer)
