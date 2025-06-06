from rag.embed import load_iphone_catalog, embed_iphone_catalog
from rag.db import get_qdrant_client
from qdrant_client.http.models import Distance, VectorParams
import os

COLLECTION_NAME = "iphones-catalog"
VECTOR_SIZE = 1536  # OpenAI ada-002 embedding size
MD_PATH = "data/raw/iphone-catalog.md"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

def ensure_collection(client, collection_name=COLLECTION_NAME, vector_size=VECTOR_SIZE):
    if collection_name not in [c.name for c in client.get_collections().collections]:
        print(f"[INFO] Creating collection '{collection_name}' in Qdrant...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
    else:
        print(f"[INFO] Collection '{collection_name}' already exists.")

def main():
    print(f"[DEBUG] QDRANT_URL: {QDRANT_URL}")
    print(f"[DEBUG] QDRANT_API_KEY: {QDRANT_API_KEY}")
    print("[INFO] Connecting to Qdrant...")
    client = get_qdrant_client()
    ensure_collection(client)
    print(f"[INFO] Loading iPhone catalog from {MD_PATH} ...")
    pages = load_iphone_catalog(MD_PATH)
    print(f"[INFO] Loaded {len(pages)} pages. Embedding and upserting...")
    embed_iphone_catalog(pages, client, collection_name=COLLECTION_NAME)
    print("[SUCCESS] iPhone catalog upserted to Qdrant collection 'iphones-catalog'.")

if __name__ == "__main__":
    main() 