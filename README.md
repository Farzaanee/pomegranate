# Grounded Investment Research Agent (EU/UK)

> **Educational only — not regulated financial advice.** This project retrieves
> information from official sources; it does not recommend specific investments or
> replace professional advice.

**Live demo:** the Streamlit UI in [`app.py`](app.py) (retrieval search, grounded
recommendations, EU-vs-UK comparison, and the stored evaluation results) is
deployed on Streamlit Community Cloud. Run it locally with `streamlit run app.py` after
`uv sync --extra demo`.

A retrieval-augmented pipeline that grounds investment-education answers in
trusted public sources — ESMA's Investor Corner (EU), MoneyHelper (UK), and FCA
InvestSmart (UK). Every retrieved passage keeps its source name, region, URL,
title, and chunk index so answers stay traceable.

**Status: Phase 4 (Evaluation and polish) — all planned phases complete.**
A local knowledge base with RAG retrieval (Phase 1); an agentic reasoning layer
that turns a user profile plus retrieved evidence into a citation-backed
recommendation (Phase 2); explicit EU/UK routing and a grounded cross-region
comparison (Phase 3); and an evaluation harness that scores groundedness,
citation accuracy, and hallucination rate (Phase 4). The latest live run passed
12 of 12 cases against a 90% bar — read [evaluation.md](src/docs/evaluation.md)
for what that does and does not show (it is a small, self-tuned, single-run
result, not a benchmark). The phase plan is in
[project-plan.md](src/docs/project-plan.md); what each phase built and what
was learned is in [implementation-notes.md](src/docs/implementation-notes.md).

**Architecture:** [architecture.md](src/docs/architecture.md) has the full
system diagram (how all the phases share one retrieval mechanism) and a
module-by-module reference. **Evaluation:** [evaluation.md](src/docs/evaluation.md)
covers the checks, the results, and the limitations.

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
| Compare | `investment-rag compare "<topic>"` | **Phase 3.** Retrieves the topic from each region separately and asks Claude for the EU view, the UK view, and the key difference. A citation is kept only if its `EU-`/`UK-` label matches the region of the passage it resolves to. Needs `ANTHROPIC_API_KEY`. |
| Evaluate | `investment-rag evaluate [--retrieval-only] [--output FILE]` | **Phase 4.** Runs the 12-case eval set and reports citation validity, quote support, region match, and good-advice checks; exits non-zero below 90% of cases passing. `--retrieval-only` needs no API key. |

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
  reasoning.py   evidence gathering, Claude call, citation validation (Phases 2-3)
  evaluation.py  eval cases, deterministic checks, report (Phase 4)
  cli.py         the `investment-rag` collect/build/query/advise/compare/evaluate commands
app.py           Streamlit UI (four modes)
evals/           cases.json (the eval set) and results/latest.json (the latest report)
tests/           offline tests for every stage (fakes stand in for Chroma + Claude)
sources.json     allow-list of official seed pages
src/docs/        architecture.md, evaluation.md, implementation-notes.md, project-plan.md, SETUP.md, and design-notes docs
```

## Design decisions

- **Education, not advice.** The prompts ask which *types* of vehicle generally
  fit a synthetic profile and why — never a product, an amount, a return
  forecast, or an instruction to act — and a fixed disclaimer is appended in code
  so the model cannot omit it.
- **Citations are validated in code, not trusted.** Every label must resolve to
  a passage that was actually retrieved, and comparison labels must match the
  region of the passage they resolve to. A response with no valid citation is an
  error, not an answer.
- **Region is enforced twice.** The Chroma query filters by region, and
  `gather_evidence` re-checks each result's region itself, so a store bug
  cannot leak an EU source into a UK answer.
- **Measure the raw model output.** Validation hides fabricated citations from
  the user, so the evaluation scores the payload *before* validation. The first
  live run (4 of 12 cases) found two real product bugs and two over-strict
  checks; see [evaluation.md](src/docs/evaluation.md).
- **No PyTorch.** ONNX MiniLM embeddings keep the install small enough for the
  Streamlit Community Cloud free tier, and the built index is committed so the
  app does no embedding work at startup.

## Development

```bash
uv run pytest -q          # all tests run offline (no network, no model download)
```

`tests/test_integration.py` exercises the real chunking logic and a real on-disk
Chroma store; the other test modules cover cleaning, provenance, search wiring,
user-profile validation, the reasoning agent, and the evaluation harness (with
fake LLMs, so no API key or network is needed to run the suite).

To re-run the live evaluation (about 12 Claude calls, a few minutes):
`uv run investment-rag evaluate --output evals/results/latest.json`.

Agent-driven work sessions on this repo (model, effort, tokens, cost, wall
time) are tracked in [performance-log.md](src/docs/performance-log.md).
