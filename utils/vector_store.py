"""Vector store wrapper built on Chroma.

Chroma is used instead of FAISS because it stores documents + metadata +
embeddings together with a persistent on-disk store, so we don't need to
manage a separate metadata index by hand. Embeddings are computed locally
with a sentence-transformers model, so no external embedding API/key is needed.
"""
import chromadb
from chromadb.utils import embedding_functions


class VectorStore:
    def __init__(self, persist_dir="chroma_db", embedding_model="all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.collection = None

    def get_or_create_collection(self, name: str):
        # Chroma collection names have character restrictions; sanitize.
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:60] or "collection"
        self.collection = self.client.get_or_create_collection(
            name=safe_name,
            embedding_function=self.embedding_fn,
        )
        return self.collection

    def clear_collection(self):
        """Wipe existing entries so re-processing the same doc doesn't duplicate chunks."""
        existing = self.collection.get()
        ids = existing.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)

    def delete_by_ids(self, ids):
        """Remove specific chunks by ID - used by incremental sync to drop only
        the chunks belonging to a file that was modified or deleted, without
        touching any other file's chunks in the same collection."""
        if ids:
            self.collection.delete(ids=ids)

    def add_chunks(self, chunks):
        if not chunks:
            return
        self.collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{"source": c["source"], "location": c["location"]} for c in chunks],
        )

    def query(self, question: str, top_k: int = 4):
        """Return list of (document_text, metadata, distance) tuples, best match first."""
        results = self.collection.query(query_texts=[question], n_results=top_k)
        docs = results["documents"][0] if results.get("documents") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else [None] * len(docs)
        return list(zip(docs, metadatas, distances))

    def list_sources(self):
        """Distinct file names currently represented in this collection - used
        to answer 'what/how many files are loaded' questions directly and
        deterministically, instead of relying on vector search (which has no
        reliable way to retrieve a meta-question like that, since it matches
        against content chunks, not document identity)."""
        existing = self.collection.get(include=["metadatas"])
        metadatas = existing.get("metadatas") or []
        sources = sorted({m.get("source", "unknown") for m in metadatas})
        return sources