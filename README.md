# Grounded Investment Research Agent (EU/UK)

> **Educational only — not regulated financial advice.** This project retrieves
> information from official sources; it does not recommend specific investments or
> replace professional advice.

A retrieval-augmented pipeline that grounds investment-education answers in
trusted public sources — ESMA's Investor Corner (EU), MoneyHelper (UK), and FCA
InvestSmart (UK). Every retrieved passage keeps its source name, region, URL,
title, and chunk index so answers stay traceable.

**Status: Phase 1 (Foundations).** Local knowledge base + basic RAG retrieval.
The reasoning agent, multi-jurisdiction routing, and evaluation come in later
phases — see [project-plan.md](project-plan.md).

## Quick start

Requires **Python 3.13** (pinned in `.python-version`; 3.10+ works). Using
[uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev                                   # build .venv/, install project + pytest
uv run pytest -q                                      # offline sanity check

uv run investment-rag collect --pages-per-source 1    # download + clean sources -> data/raw/
uv run investment-rag build                           # chunk + index into Chroma -> data/chroma/
uv run investment-rag query "Why does diversification matter?" --region UK
```

Prefer plain `venv`, need to pick a different Python, or hit a `403` from a
source? See **[SETUP.md](SETUP.md)** for the full guide and troubleshooting.

## How it works

| Stage | Command | What it does |
| --- | --- | --- |
| Collect | `investment-rag collect` | Fetches the [`sources.json`](sources.json) allow-list, strips site chrome, writes reviewable JSON to `data/raw/`. Refuses to bypass access controls — blocked sources are reported and skipped. |
| Build | `investment-rag build` | Splits documents into overlap-preserving chunks and upserts them (with provenance metadata) into persistent local Chroma at `data/chroma/`. |
| Query | `investment-rag query "<question>" [--region EU\|UK]` | Semantic search over the index; prints each supporting passage with its source, region, and URL. `--region` hard-filters to prevent cross-region retrieval. |

Embeddings use `all-MiniLM-L6-v2` (downloads on first use). `data/` is gitignored
— the knowledge base is rebuilt locally, never committed.

## Repository layout

```
src/investment_rag/
  models.py      SourceDocument / Chunk / SearchResult dataclasses
  collect.py     download + HTML cleaning
  chunking.py    sentence-boundary chunking with overlap
  retrieval.py   Chroma-backed embed / index / search (+ region filter)
  cli.py         the `investment-rag` collect/build/query commands
tests/           offline tests for every stage
sources.json     allow-list of official seed pages
project-plan.md  full project brief and phase-by-phase plan
```

## Development

```bash
uv run pytest -q          # all tests run offline (no network, no model download)
```

`tests/test_integration.py` exercises the real chunking logic and a real on-disk
Chroma store; the other test modules cover cleaning, provenance, and search
wiring.
