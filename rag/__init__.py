import time
from .db import get_qdrant_client, BIO_COLLECTION, IPHONE_COLLECTION
from .retrieve import retrieve_embeddings
from .agent import generate_answer

def rag_answer(query: str):
    """
    Given a query, retrieves the most relevant page and generates an answer using OpenAI LLM.
    Returns (answer, timing_dict)
    """
    client = get_qdrant_client()
    start_retrieve = time.time()
    result = retrieve_embeddings(query, client, IPHONE_COLLECTION)
    end_retrieve = time.time()
    if not result:
        return ("No relevant data found.", {"retrieval": end_retrieve - start_retrieve, "generation": 0})
    context = result["text"]
    start_gen = time.time()
    answer = generate_answer(query, context)
    end_gen = time.time()
    return answer, {"retrieval": end_retrieve - start_retrieve, "generation": end_gen - start_gen} 