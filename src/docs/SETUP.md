# Setup

Getting the Phase 1 knowledge-base pipeline running locally. For what the project
is and where it's going, see [README.md](README.md).

## Prerequisites

- **Python 3.13** — pinned in [`.python-version`](.python-version). 3.10–3.12 also
  work; avoid pre-release interpreters (no prebuilt wheels for `torch`,
  `chromadb`, `onnxruntime`).
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
- Roughly **500 MB free disk** and a network connection for the first run — the
  embedding model (`all-MiniLM-L6-v2`, ~90 MB) and PyTorch download once and are
  cached.

Check which Python uv already has:

```bash
uv python list --only-installed
uv python find '>=3.10'      # prints the interpreter uv would use here
```

If nothing suitable shows up, `uv python install 3.13` (or just run `uv sync`,
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
python3.13 -m venv .venv
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

## Run the pipeline

Three stages, each a subcommand of `investment-rag`:

```bash
# 1. Download + clean the allow-listed sources into data/raw/*.json
uv run investment-rag collect --pages-per-source 1

# 2. Chunk those documents and index them into local Chroma at data/chroma/
uv run investment-rag build

# 3. Ask a question; prints supporting passages with source, region, and URL
uv run investment-rag query "Why does diversification matter when investing?" --region UK
```

Useful flags:

| Command | Flag | Default | Purpose |
| --- | --- | --- | --- |
| `collect` | `--sources` | `sources.json` | allow-list of seed pages |
| `collect` | `--output` | `data/raw` | where cleaned JSON is written |
| `collect` | `--pages-per-source` | `1` | also follow up to N-1 same-domain links per seed |
| `build` | `--input` / `--store` | `data/raw` / `data/chroma` | doc source and vector store paths |
| `query` | `--region` | *(none)* | `EU` or `UK`; hard-filters cross-region results |
| `query` | `--limit` | `4` | number of passages to return |
| `query` | `--store` | `data/chroma` | vector store to search |

`data/raw/` and `data/chroma/` are gitignored — the knowledge base is rebuilt
locally, never committed.

## Configuration

[`sources.json`](sources.json) is the allow-list. Each entry needs `name`,
`region` (`EU` or `UK`), and `url`:

```json
{ "name": "MoneyHelper", "region": "UK", "url": "https://www.moneyhelper.org.uk/..." }
```

Before raising `--pages-per-source`, check each publisher's terms and robots
policy and eyeball the collected JSON in `data/raw/`.

## Troubleshooting

**`collect` reports a source was skipped (403 Forbidden).** Expected for some
publishers — the collector refuses to work around access controls. Replace that
entry in `sources.json` with an accessible official page and re-run `collect`.
MoneyHelper is the most reliable of the three defaults.

**First `build` or `query` hangs / downloads a lot.** It's fetching PyTorch and
the `all-MiniLM-L6-v2` weights. Subsequent runs use the cache
(`~/.cache/huggingface`, `~/.cache/torch`).

**`TypeError: unsupported operand type(s) for |` on import.** You're on Python
3.9. Use 3.10+ (`uv python install 3.13`).

**Chroma**
An open-source vector database used for storing and searching embeddings, often 
paired with LLMs for retrieval-augmented generation (RAG). Developers use it to 
store text/document embeddings and do similarity search. Lightweight, easy to run 
locally, popular in AI/ML projects.

**Chroma tries to download an ONNX model during tests.** Shouldn't happen — the
tests pass embeddings explicitly. If a future Chroma version instantiates its
default embedding function eagerly, open an issue / ping so the collection can be
created with an explicit no-op embedding function.

**Start over.** `rm -rf data/` wipes collected docs and the vector store;
`rm -rf .venv && uv sync --extra dev` rebuilds the environment.
