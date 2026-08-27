"""Chroma-backed semantic retrieval with source and region filters."""

from pathlib import Path
from typing import Protocol

import chromadb
from sentence_transformers import SentenceTransformer

from .models import Chunk, SearchResult


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class Retriever:
    """Persistent local vector retrieval; Chroma stores text and provenance together."""

    def __init__(self, persist_dir: str | Path = "data/chroma", collection_name: str = "investment_knowledge",
                 embedder: Embedder | None = None) -> None:
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata() for chunk in chunks],
            embeddings=self.embedder.encode([chunk.text for chunk in chunks]),
        )

    def search(self, question: str, limit: int = 4, region: str | None = None) -> list[SearchResult]:
        result = self.collection.query(
            query_embeddings=self.embedder.encode([question]),
            n_results=limit,
            where={"region": region} if region else None,
            include=["documents", "metadatas", "distances"],
        )
        return [
            SearchResult(
                Chunk(result["ids"][0][index], result["documents"][0][index], metadata["source_name"],
                      metadata["region"], metadata["url"], metadata["title"], metadata["chunk_index"]),
                result["distances"][0][index],
            )
            for index, metadata in enumerate(result["metadatas"][0])
        ]
