# material.py
# Responsible for loading and chunking raw text data (novels)

import os

def load_text(file_path):
    """
    Reads a text file and returns its content as a single string.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def chunk_text(text, chunk_size=5):
    """
    Splits text into chunks of N lines (default = 5 lines per chunk).
    Returns a list of text chunks.
    """
    lines = text.split("\n")
    chunks = []

    for i in range(0, len(lines), chunk_size):
        chunk = "\n".join(lines[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def load_and_chunk_novel(file_path, chunk_size=5):
    """
    Convenience function:
    Loads a novel and returns chunked passages.
    """
    text = load_text(file_path)
    chunks = chunk_text(text, chunk_size)
    return chunks


def load_all_novels(books_dir, chunk_size=5):
    """
    Loads all .txt files from the books directory and chunks them.
    Returns a dictionary with file names as keys and chunked passages as values.
    """
    novels = {}
    
    # Ensure the directory exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(f"Books directory not found: {books_dir}")
    
    # Load all text files from the directory
    for file_name in os.listdir(books_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(books_dir, file_name)
            try:
                chunks = load_and_chunk_novel(file_path, chunk_size)
                novels[file_name] = chunks
                print(f"Loaded '{file_name}': {len(chunks)} chunks")
            except Exception as e:
                print(f"Error loading '{file_name}': {e}")
    
    return novels
