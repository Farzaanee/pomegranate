# Setup

Getting the knowledge-base pipeline (Phase 1), reasoning agent (Phase 2),
EU/UK comparison (Phase 3), and evaluation harness (Phase 4) running locally. For what the project is and where it's going, see
[README.md](README.md).

## Prerequisites

- **Python 3.12** — pinned in [`.python-version`](.python-version). 3.10, 3.11 and
  3.13 also work; avoid pre-release interpreters (no prebuilt wheels for
  `chromadb`, `onnxruntime`). 3.12 is the pin because it has the widest wheel
  coverage on Streamlit Community Cloud.
- **[uv](https://docs.astral.sh/uv/)** (recommended) — a `uv.lock` is committed so
  installs are reproducible. On macOS with Homebrew:
  ```bash
  brew install uv          # update later with: brew upgrade uv
  ```
  Otherwise (any OS, no Homebrew required):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  Then restart your shell (the curl installer adds `~/.local/bin` to PATH; or
  `source $HOME/.local/bin/env` for the current session).
- Roughly **150 MB free disk** and a network connection for the first run — the
  `all-MiniLM-L6-v2` ONNX weights (~80 MB) download once and are cached in
  `~/.cache/chroma/`. No PyTorch.

Check which Python uv already has:

```bash
uv python list --only-installed
uv python find '>=3.10'      # prints the interpreter uv would use here
```

If nothing suitable shows up, `uv python install 3.12` (or just run `uv sync`,
which fetches one automatically).

## Install

### With uv (recommended)

```bash
cd /path/to/pomegranate
uv sync --extra dev
```

This creates `.venv/`, installs the project in editable mode plus `pytest`, and
locks everything to `uv.lock`. Prefix commands with `uv run` (no activation
needed), or activate the venv with `source .venv/bin/activate`.

### With the standard library `venv`

```bash
cd /path/to/pomegranate
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Verify the install

```bash
uv run pytest -q
```

All tests run **offline** (no network, no model download):

- `tests/test_collect.py` — HTML cleaning, provenance in saved JSON, 403 handling
- `tests/test_chunking.py` — chunk sizing, overlap, metadata
- `tests/test_retrieval.py` — search wiring + region filter (fully mocked)
- `tests/test_integration.py` — real `chunk_document` output and real on-disk
  Chroma indexing/retrieval with a deterministic fake embedder
- `tests/test_profile.py` — user profile validation and query derivation
- `tests/test_reasoning.py` — evidence gathering, citation validation, the
  EU/UK comparison, and the reasoning agent, all with a fake LLM (no API key or
  network needed)
- `tests/test_evaluation.py` — every evaluation check's pass and fail path, quote
  matching, the pass threshold, and that `evals/cases.json` stays valid

## Run the pipeline

Each stage is a subcommand of `investment-rag`:

```bash
# 1. Download + clean the allow-listed sources into data/raw/*.json
uv run investment-rag collect --pages-per-source 1

# 2. Chunk those documents and index them into local Chroma at data/index/
#    (commit data/index/ afterwards — the deployed app opens it directly)
uv run investment-rag build

# 3. Ask a question; prints supporting passages with source, region, and URL
uv run investment-rag query "Why does diversification matter when investing?" --region UK

# 4. Phase 2: get a grounded, cited recommendation for a user profile (needs
#    ANTHROPIC_API_KEY — see Configuration below)
uv run investment-rag advise --income 2500 --amount 3000 --goal retirement \
    --timeline 15 --risk medium --region UK

# 5. Phase 3: explain an EU vs UK difference on a topic (needs ANTHROPIC_API_KEY)
uv run investment-rag compare "How can an investor check that a firm is authorised?"

# 6. Phase 4: score groundedness, citation accuracy, and hallucination rate.
#    Free, no key (retrieval and region routing only):
uv run investment-rag evaluate --retrieval-only
#    Full run: ~12 Claude calls, a few minutes, needs ANTHROPIC_API_KEY.
#    Exits 1 if fewer than 90% of cases pass.
uv run investment-rag evaluate --output evals/results/latest.json
```

Useful flags:

| Command | Flag | Default | Purpose |
| --- | --- | --- | --- |
| `collect` | `--sources` | `sources.json` | allow-list of seed pages |
| `collect` | `--output` | `data/raw` | where cleaned JSON is written |
| `collect` | `--pages-per-source` | `1` | also follow up to N-1 same-domain links per seed |
| `build` | `--input` / `--store` | `data/raw` / `data/index` | doc source and vector store paths |
| `query` | `--region` | *(none)* | `EU` or `UK`; hard-filters cross-region results |
| `query` | `--limit` | `4` | number of passages to return |
| `query` | `--store` | `data/index` | vector store to search |
| `advise` | `--income` / `--amount` | *(required)* | monthly income / amount available to invest |
| `advise` | `--goal` | *(required)* | `retirement`, `house_deposit`, `general_growth`, or `emergency_fund` |
| `advise` | `--timeline` | *(required)* | investing horizon in years |
| `advise` | `--risk` | *(required)* | `low`, `medium`, or `high` |
| `advise` | `--region` | *(required)* | `EU` or `UK`; also scopes evidence retrieval |
| `advise` | `--model` | `claude-sonnet-5` | Anthropic model used for the reasoning step |
| `compare` | `--model` / `--store` | `claude-sonnet-5` / `data/index` | model for the comparison step; vector store to search |
| `evaluate` | `--cases` | `evals/cases.json` | the eval set |
| `evaluate` | `--retrieval-only` | off | skip the Claude calls; no API key needed |
| `evaluate` | `--output` | *(none)* | also write the full report as JSON (the app's "Evaluation results" mode reads `evals/results/latest.json`) |
| `evaluate` | `--model` / `--store` | `claude-sonnet-5` / `data/index` | model under evaluation; vector store (evaluated on a temp copy, never modified) |

`data/raw/` and `data/index/` **are committed** so the deployed Streamlit app
opens a prebuilt index instead of re-embedding on startup. `data/chroma/` is a
gitignored scratch path for ad-hoc local builds. Rebuild and commit `data/index/`
whenever `data/raw/` or the chunking logic changes.

## Configuration

[`sources.json`](sources.json) is the allow-list. Each entry needs `name`,
`region` (`EU` or `UK`), and `url`:

```json
{ "name": "MoneyHelper", "region": "UK", "url": "https://www.moneyhelper.org.uk/..." }
```

Before raising `--pages-per-source`, check each publisher's terms and robots
policy and eyeball the collected JSON in `data/raw/`.

### Anthropic API key (Phases 2-4)

`investment-rag advise`, `compare`, a full `evaluate` run, and the app's
"Grounded recommendation" and "Compare EU vs UK" modes call Claude
(`investment_rag.reasoning.ClaudeRecommendationLLM`). Retrieval and search
(`collect` / `build` / `query`), `evaluate --retrieval-only`, and the app's
"Evaluation results" mode never need a key.

- **Locally:** `export ANTHROPIC_API_KEY=sk-ant-...` (or run `ant auth login`,
  which the SDK also picks up automatically).
- **On Streamlit Community Cloud:** add `ANTHROPIC_API_KEY = "sk-ant-..."` to the
  app's Secrets in the dashboard; [app.py](app.py) reads it from `st.secrets` and
  exports it into the environment before creating the agent.
- **Model/cost:** defaults to `claude-sonnet-5` for latency. Pass `--model claude-opus-5` (CLI)
  or set `model=` on `ClaudeRecommendationLLM` (app) for deeper reasoning at the
  cost of speed — that trade-off is yours to make, not the default.

## Troubleshooting

**`collect` reports a source was skipped (403 Forbidden).** Expected for some
publishers — the collector refuses to work around access controls. Replace that
entry in `sources.json` with an accessible official page and re-run `collect`.
MoneyHelper is the most reliable of the three defaults.

**First `build` or `query` downloads ~80 MB.** It's fetching the
`all-MiniLM-L6-v2` ONNX weights. Subsequent runs use the cache
(`~/.cache/chroma/onnx_models/`). No PyTorch is involved.

**`TypeError: unsupported operand type(s) for |` on import.** You're on Python
3.9. Use 3.10+ (`uv python install 3.12`).

**`evaluate` reports every case as `aborted: TypeError: Could not resolve
authentication method`.** No API key is set, so every Claude call failed. Set
`ANTHROPIC_API_KEY`, or use `--retrieval-only`.

**`advise` / `compare` raises `ReasoningError: ... cut off at the token limit`.**
The model's answer hit `max_tokens`. Retry; if it recurs, raise `max_tokens` in
`ClaudeRecommendationLLM`.

**`advise` raises `anthropic.AuthenticationError` or exits complaining about a
missing key.** Set `ANTHROPIC_API_KEY` — see *Anthropic API key* above.

**`advise` raises `ReasoningError: No evidence retrieved for this profile's
region`.** The index has no chunks for that region — run `collect` + `build`
for it, or double check `--region` matches what's indexed.

**Chroma**
An open-source vector database used for storing and searching embeddings, often 
paired with LLMs for retrieval-augmented generation (RAG). Developers use it to 
store text/document embeddings and do similarity search. Lightweight, easy to run 
locally, popular in AI/ML projects.

**Chroma tries to download an ONNX model during tests.** Shouldn't happen — the
tests pass embeddings explicitly. If a future Chroma version instantiates its
default embedding function eagerly, open an issue / ping so the collection can be
created with an explicit no-op embedding function.

**Start over.** `rm -rf data/chroma/` clears the local scratch store (rerun
`investment-rag build` to regenerate `data/index/`); `rm -rf .venv && uv sync
--extra dev` rebuilds the environment.
