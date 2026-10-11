"""Command-line entry points for the RAG pipeline, reasoning agent, EU/UK comparison, and evaluation."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from .chunking import chunk_documents
from .collect import CollectionError, collect_source, load_sources, save_documents
from .models import SourceDocument
from .profile import GOALS, RISK_TOLERANCES, UserProfile
from .evaluation import load_cases, run_evaluation
from .reasoning import DISCLAIMER, ClaudeRecommendationLLM, ReasoningAgent, explain_region_difference
from .retrieval import Retriever


def _load_documents(directory: str) -> list[SourceDocument]:
    """Load every cleaned source document JSON file in ``directory``."""
    return [SourceDocument(**json.loads(path.read_text(encoding="utf-8"))) for path in Path(directory).glob("*.json")]


def main() -> None:
    """Parse the command line and dispatch to the selected ``investment-rag`` subcommand."""
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
    advise_parser = commands.add_parser("advise", help="Produce a grounded, cited recommendation for a user profile.")
    advise_parser.add_argument("--income", type=float, required=True, help="Monthly income, local currency.")
    advise_parser.add_argument("--amount", type=float, required=True, help="Amount available to invest.")
    advise_parser.add_argument("--goal", choices=GOALS, required=True)
    advise_parser.add_argument("--timeline", type=int, required=True, help="Investing horizon in years.")
    advise_parser.add_argument("--risk", choices=RISK_TOLERANCES, required=True, dest="risk_tolerance")
    advise_parser.add_argument("--region", choices=["EU", "UK"], required=True)
    advise_parser.add_argument("--store", default="data/index")
    advise_parser.add_argument("--model", default="claude-sonnet-5", help="Anthropic model for the reasoning step.")
    compare_parser = commands.add_parser(
        "compare", help="Explain a key EU vs UK difference on a topic, grounded in both regions' sources."
    )
    compare_parser.add_argument("topic")
    compare_parser.add_argument("--store", default="data/index")
    compare_parser.add_argument("--model", default="claude-sonnet-5", help="Anthropic model for the comparison step.")
    evaluate_parser = commands.add_parser(
        "evaluate", help="Phase 4: score groundedness, citation accuracy, and hallucination rate on an eval set."
    )
    evaluate_parser.add_argument("--cases", default="evals/cases.json")
    evaluate_parser.add_argument("--store", default="data/index")
    evaluate_parser.add_argument("--model", default="claude-sonnet-5", help="Anthropic model under evaluation.")
    evaluate_parser.add_argument("--retrieval-only", action="store_true",
                                 help="Skip the Claude calls (no API key needed); checks retrieval and region routing only.")
    evaluate_parser.add_argument("--output", help="Also write the full report as JSON to this path.")
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
    elif args.command == "query":
        for item in Retriever(args.store).search(args.question, args.limit, args.region):
            print(f"[{item.chunk.source_name} | {item.chunk.region} | {item.chunk.url}]\n{item.chunk.text}\n")
    elif args.command == "evaluate":
        llm = None if args.retrieval_only else ClaudeRecommendationLLM(model=args.model)
        # Chroma writes journal files on open; evaluate a throwaway copy so the committed index stays byte-identical.
        with tempfile.TemporaryDirectory(prefix="chroma-eval-") as scratch:
            store_copy = Path(scratch) / "index"
            shutil.copytree(args.store, store_copy)
            report = run_evaluation(load_cases(args.cases), Retriever(store_copy), llm,
                                    "" if llm is None else args.model)
        print(report.to_markdown())
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        if llm is not None and not report.meets_threshold():
            raise SystemExit(1)
    elif args.command == "compare":
        llm = ClaudeRecommendationLLM(model=args.model)
        result = explain_region_difference(Retriever(args.store), llm, args.topic)
        print(f"{DISCLAIMER}\n")
        print(f"EU: {result.eu_summary}\n")
        print(f"UK: {result.uk_summary}\n")
        print(f"Key difference: {result.key_difference}\n")
        print("Sources:")
        for citation in result.citations:
            print(f"  [{citation.label}] {citation.source_name} ({citation.region}) — {citation.url}")
            print(f'      "{citation.quote}"')
    else:
        profile = UserProfile(args.income, args.amount, args.goal, args.timeline, args.risk_tolerance, args.region)
        agent = ReasoningAgent(Retriever(args.store), ClaudeRecommendationLLM(model=args.model))
        recommendation = agent.run(profile)
        print(f"{DISCLAIMER}\n")
        print(f"{recommendation.summary}\n")
        for option in recommendation.suitable_options:
            labels = " ".join(f"[{label}]" for label in option.citation_labels)
            print(f"- {option.vehicle_type} {labels}".rstrip())
            print(f"    Why it fits: {option.why_it_fits}")
            print(f"    Tradeoffs: {option.tradeoffs}")
        print("\nReasoning:")
        for step in recommendation.reasoning:
            print(f"- {step}")
        if recommendation.risks:
            print("\nRisks:")
            for risk in recommendation.risks:
                print(f"- {risk}")
        if recommendation.caveats:
            print("\nCaveats:")
            for caveat in recommendation.caveats:
                print(f"- {caveat}")
        print("\nSources:")
        for citation in recommendation.citations:
            print(f"  [{citation.label}] {citation.source_name} ({citation.region}) — {citation.url}")
            print(f'      "{citation.quote}"')


if __name__ == "__main__":
    main()
