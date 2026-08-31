"""Chroma-backed semantic retrieval with source and region filters."""

from pathlib import Path
from typing import Protocol

import chromadb

from .models import Chunk, SearchResult


class Embedder(Protocol):
    """Turns text into normalized embedding vectors for indexing and querying."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Return one L2-normalized vector per input string."""
        ...


class OnnxMiniLmEmbedder:
    """``all-MiniLM-L6-v2`` (384-dim, L2-normalized) served through ONNX Runtime.

    This is the same model ``sentence-transformers`` exposes under that name, but
    the ONNX build that ships with Chroma has no PyTorch dependency: it cuts
    roughly 450 MB off the install and a few hundred MB of resident memory, which
    is what keeps the Streamlit Community Cloud free tier (1 GB RAM) alive. The
    ~80 MB ONNX weights download once on first use and are then cached on disk.
    """

    def __init__(self) -> None:
        """Load the cached ONNX MiniLM embedding function, downloading it if absent."""
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._embed = ONNXMiniLM_L6_V2()

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` and return plain nested lists of floats."""
        return [[float(value) for value in vector] for vector in self._embed(list(texts))]


class Retriever:
    """Persistent local vector retrieval; Chroma stores text and provenance together."""

    def __init__(self, persist_dir: str | Path = "data/index", collection_name: str = "investment_knowledge",
                 embedder: Embedder | None = None) -> None:
        """Open (or create) the Chroma store at ``persist_dir`` and its collection.

        ``embedder`` defaults to :class:`OnnxMiniLmEmbedder`; tests inject a fake
        so nothing is downloaded.
        """
        self.embedder = embedder or OnnxMiniLmEmbedder()
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    def index(self, chunks: list[Chunk]) -> None:
        """Embed ``chunks`` and upsert them with provenance metadata (idempotent by id)."""
        if not chunks:
            return
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata() for chunk in chunks],
            embeddings=self.embedder.encode([chunk.text for chunk in chunks]),
        )

    def search(self, question: str, limit: int = 4, region: str | None = None) -> list[SearchResult]:
        """Return up to ``limit`` passages ranked by cosine distance to ``question``.

        ``region`` (``"EU"`` / ``"UK"``) restricts results to that jurisdiction;
        ``None`` searches the whole knowledge base.
        """
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
