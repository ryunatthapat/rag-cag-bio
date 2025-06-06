import os
from typing import List, Tuple

BIO_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'biographies.md')


def load_biographies_for_rag() -> List[Tuple[str, str]]:
    """
    Loads and chunks biographies.md by page for RAG.
    Returns a list of (page_title, page_content) tuples.
    """
    with open(BIO_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
     # Split by page delimiter (--- on its own line)
    pages = [p.strip() for p in text.split('---') if p.strip()]
    return [{"page": i+1, "text": page} for i, page in enumerate(pages)]


def load_biographies_for_cag() -> str:
    """
    Loads the entire biographies.md as a single string for CAG.
    """
    with open(BIO_PATH, 'r', encoding='utf-8') as f:
        return f.read() 