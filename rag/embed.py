import os
from openai import OpenAI
from tqdm import tqdm
from qdrant_client.models import PointStruct
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-ada-002")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


def get_embedding(text: str, model: str = EMBED_MODEL) -> list:
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def load_iphone_catalog(md_path: str) -> List[Dict]:
    """
    Parse iphone-catalog.md into a list of dicts: {"page": ..., "text": ...}
    Each page starts with '## Page N:' and ends with '---' or EOF.
    """
    pages = []
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    current_page = None
    current_text = []
    for line in lines:
        if line.startswith("## Page "):
            if current_page is not None:
                pages.append({"page": current_page, "text": "".join(current_text).strip()})
            current_page = line.strip().replace('## ', '')
            current_text = []
        elif line.strip() == '---':
            if current_page is not None:
                pages.append({"page": current_page, "text": "".join(current_text).strip()})
                current_page = None
                current_text = []
        else:
            if current_page is not None:
                current_text.append(line)
    # Catch last page if file doesn't end with ---
    if current_page is not None and current_text:
        pages.append({"page": current_page, "text": "".join(current_text).strip()})
    return pages


def embed_iphone_catalog(pages: List[Dict], client_qdrant, collection_name: str = "iphones-catalog"):
    """
    For each iPhone catalog page, get embedding and upsert into Qdrant.
    Uses an integer index as the point ID (required by Qdrant).
    """
    for idx, page in enumerate(tqdm(pages, desc="Embedding iPhone catalog")):
        page_id = idx  # integer ID
        text = page["text"]
        page_title = page["page"]
        embedding = get_embedding(text)
        point = PointStruct(
            id=page_id,
            vector=embedding,
            payload={"page": page_title, "text": text}
        )
        client_qdrant.upsert(collection_name=collection_name, points=[point])

# Example usage:
# from qdrant_client import QdrantClient
# client_qdrant = QdrantClient("localhost", port=6333)
# pages = load_iphone_catalog("data/raw/iphone-catalog.md")
# embed_iphone_catalog(pages, client_qdrant, collection_name="iphones-catalog") 