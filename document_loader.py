import fitz  # PyMuPDF for PDFs

def read_pdf(file_path):
    """Reads a PDF file and extracts text."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def read_text(file_path):
    """Reads a plain text file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def split_text_into_chunks(text, chunk_size=1000):
    """
    Splits the text into smaller chunks of a specified size.

    Args:
        text (str): The text to be split.
        chunk_size (int): The maximum size of each chunk in characters.

    Returns:
        list: A list of text chunks.
    """
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks
