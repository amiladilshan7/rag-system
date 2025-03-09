from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into smaller overlapping chunks.

    Args:
        text (str): The full document text.
        chunk_size (int): Size of each chunk.
        overlap (int): Number of overlapping characters between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    return splitter.split_text(text)



# Load pre-trained embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(chunks):
    """
    Converts text chunks into numerical embeddings.

    Args:
        chunks (List[str]): List of text chunks.

    Returns:
        List[List[float]]: List of embeddings.
    """
    return embedding_model.encode(chunks)

# Example usage
if __name__ == "__main__":
    sample_text = "This is a long document... (your text here)"
    chunks = chunk_text(sample_text)
    embeddings = generate_embeddings(chunks)
    print("First 3 Chunks:", chunks[:3])
    print("First 3 Embeddings:", embeddings[:3])  # Preview embeddings