"""Split cleaned documents into overlap-preserving, attributable passages."""

import hashlib
import re

from .models import Chunk, SourceDocument


def chunk_document(document: SourceDocument, max_chars: int = 900, overlap_chars: int = 150) -> list[Chunk]:
    """Chunk on sentence boundaries while retaining source metadata on every chunk."""
    if max_chars <= overlap_chars:
        raise ValueError("max_chars must exceed overlap_chars.")
    sentences = re.split(r"(?<=[.!?])\s+", document.text.strip())
    chunks, current = [], ""
    for sentence in filter(None, sentences):
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = current[-overlap_chars:]
            current = f"{current} {sentence}".strip()
        else:
            current = candidate
    if current:
        chunks.append(current)

    document_id = hashlib.sha256(document.url.encode()).hexdigest()[:12]
    return [
        Chunk(f"{document_id}-{index}", text, document.source_name, document.region, document.url,
              document.title, index)
        for index, text in enumerate(chunks)
    ]


def chunk_documents(documents: list[SourceDocument]) -> list[Chunk]:
    return [chunk for document in documents for chunk in chunk_document(document)]
