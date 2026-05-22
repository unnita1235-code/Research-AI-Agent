"""Vector store: embed extracted facts into ChromaDB for RAG retrieval."""
from __future__ import annotations
import hashlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

def rag_enabled() -> bool:
    if os.environ.get("ENABLE_RAG") != "1":
        return False
    try:
        import chromadb
        import sentence_transformers
        return True
    except ImportError:
        return False

def embed_facts(job_id: str, facts: list[str]) -> bool:
    if not rag_enabled():
        return False
    try:
        import chromadb
        from chromadb.utils import embedding_functions

        client = chromadb.EphemeralClient()
        collection_name = f"facts_{job_id[:8]}"
        
        # Use sentence-transformers "all-MiniLM-L6-v2"
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        collection = client.get_or_create_collection(name=collection_name, embedding_function=emb_fn)
        
        docs = []
        ids = []
        metadatas = []
        
        for fact in facts:
            fact_hash = hashlib.sha256(fact.encode("utf-8")).hexdigest()[:16]
            docs.append(fact)
            ids.append(fact_hash)
            metadatas.append({"job_id": job_id})
            
        if docs:
            collection.add(documents=docs, ids=ids, metadatas=metadatas)
        return True
    except Exception as e:
        logger.error(f"Failed to embed facts: {e}")
        return False

def retrieve_relevant_facts(job_id: str, query: str, n: int = 12) -> list[str]:
    if not rag_enabled():
        return []
    try:
        import chromadb
        from chromadb.utils import embedding_functions

        client = chromadb.EphemeralClient()
        collection_name = f"facts_{job_id[:8]}"
        
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        try:
            collection = client.get_collection(name=collection_name, embedding_function=emb_fn)
        except Exception:
            return []
            
        results = collection.query(query_texts=[query], n_results=n)
        if not results or not results["documents"] or not results["documents"][0]:
            return []
            
        return results["documents"][0]
    except Exception as e:
        logger.error(f"Failed to retrieve facts: {e}")
        return []
