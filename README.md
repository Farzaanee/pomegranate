# Grounded Investment Research Agent (EU/UK)

> **Educational only — not regulated financial advice.** This project retrieves
> information from official sources; it does not recommend specific investments or
> replace professional advice.

**Live demo:** the Streamlit retrieval UI in [`app.py`](app.py) is deployed on
Streamlit Community Cloud. Run it locally with `streamlit run app.py` after
`uv sync --extra demo`.

A retrieval-augmented pipeline that grounds investment-education answers in
trusted public sources — ESMA's Investor Corner (EU), MoneyHelper (UK), and FCA
InvestSmart (UK). Every retrieved passage keeps its source name, region, URL,
title, and chunk index so answers stay traceable.

**Status: Phase 2 (Reasoning Layer).** Phase 1's local knowledge base + RAG
retrieval, plus an agentic reasoning layer that combines a user profile with
retrieved evidence to produce a citation-backed recommendation. Full
multi-jurisdiction routing and evaluation come in later phases — see
[project-plan.md](src/docs/project-plan.md).

## Quick start

Requires **Python 3.12** (pinned in `.python-version`; 3.10+ works). Using
[uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev                                   # build .venv/, install project + pytest
uv run pytest -q                                      # offline sanity check

uv run investment-rag collect --pages-per-source 1    # download + clean sources -> data/raw/
uv run investment-rag build                           # chunk + index into Chroma -> data/index/
uv run investment-rag query "Why does diversification matter?" --region UK
```

Prefer plain `venv`, need to pick a different Python, or hit a `403` from a
source? See **[SETUP.md](src/docs/SETUP.md)** for the full guide and troubleshooting.

## How it works

| Stage | Command | What it does |
| --- | --- | --- |
| Collect | `investment-rag collect` | Fetches the [`sources.json`](sources.json) allow-list, strips site chrome, writes reviewable JSON to `data/raw/`. Refuses to bypass access controls — blocked sources are reported and skipped. |
| Build | `investment-rag build` | Splits documents into overlap-preserving chunks and upserts them (with provenance metadata) into persistent local Chroma at `data/index/`. |
| Query | `investment-rag query "<question>" [--region EU\|UK]` | Semantic search over the index; prints each supporting passage with its source, region, and URL. `--region` hard-filters to prevent cross-region retrieval. |
| Advise | `investment-rag advise --income ... --amount ... --goal ... --timeline ... --risk ... --region EU\|UK` | **Phase 2.** Derives evidence queries from the profile, retrieves region-scoped passages, and asks Claude for a plain-language, cited recommendation. Citations that don't match a retrieved passage are dropped, not trusted. Needs `ANTHROPIC_API_KEY` — see [SETUP.md](src/docs/SETUP.md). |

Embeddings use `all-MiniLM-L6-v2` via ONNX Runtime — the same model as
`sentence-transformers` exposes, but with no PyTorch dependency (~450 MB lighter
to install), so it fits the Streamlit Community Cloud free tier. The ~80 MB ONNX
weights download on first use and are cached.

The built index in `data/index/` **is committed** so the deployed app opens it
directly instead of re-embedding the knowledge base on every startup. `data/raw/`
is committed too; `data/chroma/` (local CLI scratch) is gitignored. Rebuild and
commit `data/index/` whenever `data/raw/` or the chunking changes.

## Repository layout

```
src/investment_rag/
  models.py      SourceDocument / Chunk / SearchResult dataclasses
  collect.py     download + HTML cleaning
  chunking.py    sentence-boundary chunking with overlap
  retrieval.py   Chroma-backed embed / index / search (+ region filter), ONNX MiniLM
  profile.py     UserProfile schema + the retrieval queries it implies (Phase 2)
  reasoning.py   evidence gathering, Claude call, citation validation (Phase 2)
  cli.py         the `investment-rag` collect/build/query/advise commands
tests/           offline tests for every stage (fakes stand in for Chroma + Claude)
sources.json     allow-list of official seed pages
src/docs/        project-plan.md, SETUP.md, and design-notes docs
```

## Development

```bash
uv run pytest -q          # all tests run offline (no network, no model download)
```

`tests/test_integration.py` exercises the real chunking logic and a real on-disk
Chroma store; the other test modules cover cleaning, provenance, search wiring,
user-profile validation, and the reasoning agent (with a fake LLM, so no API
key or network is needed to run the suite).

Agent-driven work sessions on this repo (model, effort, tokens, cost, wall
time) are tracked in [performance-log.md](src/docs/performance-log.md).
