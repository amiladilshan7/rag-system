from document_loader import read_pdf, read_text
from text_processing import chunk_text
from rag_pipeline import store_embeddings, search_similar_text



pdf_text = read_pdf("sample.pdf")  # Replace with your PDF file
text_file_text = read_text("sample.txt")  # Replace with your text file

print(pdf_text[:500])  # Show first 500 characters


# Load document
text = read_pdf("sample.pdf")  # Replace with your file
chunks = chunk_text(text)

# Store embeddings
index = store_embeddings(chunks)

# Search and summarize
query = "Explain the document in simple terms."
summary = search_similar_text(query, chunks, index)

print("Summary:", summary)