import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

QDRANT_HOST = "localhost"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_PORT = 6333
BIO_COLLECTION = "biographies"
IPHONE_COLLECTION = "iphones-catalog"
DEFAULT_VECTOR_SIZE = 1536  # OpenAI ada-002 embedding size


def get_qdrant_client():
    """
    Connect to the Qdrant instance.
    """
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        return QdrantClient(url=QDRANT_URL)


def ensure_collection(client, collection_name=IPHONE_COLLECTION, vector_size=DEFAULT_VECTOR_SIZE):
    """
    Ensure the collection exists in Qdrant.
    """
    if collection_name not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        ) 