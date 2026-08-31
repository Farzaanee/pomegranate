"""Command-line entry points for the Phase 1 RAG workflow."""

import argparse
import json
from pathlib import Path

from .chunking import chunk_documents
from .collect import CollectionError, collect_source, load_sources, save_documents
from .models import SourceDocument
from .retrieval import Retriever


def _load_documents(directory: str) -> list[SourceDocument]:
    return [SourceDocument(**json.loads(path.read_text(encoding="utf-8"))) for path in Path(directory).glob("*.json")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and query the investment education knowledge base.")
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect", help="Download and clean approved sources.")
    collect_parser.add_argument("--sources", default="sources.json")
    collect_parser.add_argument("--output", default="data/raw")
    collect_parser.add_argument("--pages-per-source", type=int, default=1)
    build_parser = commands.add_parser("build", help="Chunk clean documents and index them in Chroma.")
    build_parser.add_argument("--input", default="data/raw")
    build_parser.add_argument("--store", default="data/index")
    query_parser = commands.add_parser("query", help="Retrieve grounded passages for a question.")
    query_parser.add_argument("question")
    query_parser.add_argument("--store", default="data/index")
    query_parser.add_argument("--region", choices=["EU", "UK"])
    query_parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()

    if args.command == "collect":
        documents = []
        blocked_sources = []
        for source in load_sources(args.sources):
            print(f"Collecting {source['name']}...")
            try:
                documents.extend(collect_source(source, args.pages_per_source))
            except CollectionError as error:
                blocked_sources.append(source["name"])
                print(f"Skipped: {error}")
        print(f"Saved {len(save_documents(documents, args.output))} clean documents to {args.output}.")
        if blocked_sources:
            print("Blocked sources were skipped; update sources.json with an accessible official page before indexing "
                  f"their material: {', '.join(blocked_sources)}.")
    elif args.command == "build":
        chunks = chunk_documents(_load_documents(args.input))
        Retriever(args.store).index(chunks)
        print(f"Indexed {len(chunks)} chunks in {args.store}.")
    else:
        for item in Retriever(args.store).search(args.question, args.limit, args.region):
            print(f"[{item.chunk.source_name} | {item.chunk.region} | {item.chunk.url}]\n{item.chunk.text}\n")


if __name__ == "__main__":
    main()
