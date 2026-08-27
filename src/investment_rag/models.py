"""Domain records shared by collection, chunking, and retrieval."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceDocument:
    source_name: str
    region: str
    url: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source_name: str
    region: str
    url: str
    title: str
    chunk_index: int

    def metadata(self) -> dict[str, str | int]:
        return {
            "source_name": self.source_name,
            "region": self.region,
            "url": self.url,
            "title": self.title,
            "chunk_index": self.chunk_index,
        }


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    distance: float
