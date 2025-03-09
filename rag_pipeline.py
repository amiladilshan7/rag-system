import faiss
import numpy as np
from summarizer import summarize_text

from text_processing import generate_embeddings, chunk_text




def store_embeddings(chunks):
    """
    Converts text chunks into embeddings and stores them in a FAISS index.

    Args:
        chunks (List[str]): List of text chunks.

    Returns:
        faiss.IndexFlatL2: FAISS index storing embeddings.
    """
    embeddings = generate_embeddings(chunks)
    dimension = len(embeddings[0])  # Get the embedding size
    index = faiss.IndexFlatL2(dimension)  # Create FAISS index
    index.add(np.array(embeddings))  # Store embeddings
    return index


def search_similar_text(query, chunks, index, top_k=3):
    """
    Searches for the most relevant chunks based on the query.

    Args:
        query (str): User query.
        chunks (List[str]): List of original text chunks.
        index (faiss.IndexFlatL2): FAISS index storing embeddings.
        top_k (int): Number of top results to return.

    Returns:
        List[str]: List of relevant text chunks.
    """
    query_embedding = np.array([generate_embeddings([query])[0]])  # Convert query to embedding
    distances, indices = index.search(query_embedding, top_k)  # Search in FAISS

    retrieved_texts = [chunks[i] for i in indices[0]]
    combined_text = " ".join(retrieved_texts)  # Merge retrieved chunks

    # Summarize the retrieved text using DeepSeek
    summary = summarize_text(combined_text)

    return summary

# Example usage
if __name__ == "__main__":
    sample_text = "This is a sample document. It contains useful information. This text will be chunked and stored."
    chunks = chunk_text(sample_text)
    index = store_embeddings(chunks)

    query = "What does this document contain?"
    results = search_similar_text(query, chunks, index)

    print("Relevant Chunks:", results)
