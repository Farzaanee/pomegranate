# RAG concepts in this repo

How the Phase 1 pipeline actually works, stage by stage, and where each stage is
weak. Code lives in [`src/investment_rag/`](src/investment_rag/); this document
explains the *why* and the *what could go wrong*.

```
sources.json ──► collect ──► data/raw/*.json ──► build ──► data/chroma/ ──► query
                (fetch +      (cleaned docs,     (chunk +   (vectors +      (embed
                 clean)        reviewable)        embed)      metadata)       + search)
```

Three stages, one shared data model ([`models.py`](src/investment_rag/models.py)):

| Record | Created by | Carries |
| --- | --- | --- |
| `SourceDocument` | collect | `source_name`, `region`, `url`, `title`, `text` |
| `Chunk` | chunking | the above **plus** `id`, `chunk_index`, chunked `text` |
| `SearchResult` | retrieval | a `Chunk` + cosine `distance` |

Provenance (source, region, URL, title) is attached at collection time and copied
onto every chunk, so any retrieved passage can be traced back to an official
page. That invariant is the backbone of the whole project.

---

## 1. Collect — [`collect.py`](src/investment_rag/collect.py)

### How it works

1. **`load_sources()`** reads [`sources.json`](sources.json) and checks every
   entry has `name`, `region`, `url`. This file is an explicit allow-list — the
   collector never follows its nose to arbitrary domains.
2. **`collect_source()`** opens a `requests.Session` with a browser-like
   `User-Agent` and `Accept-Language: en-GB`, then `GET`s the seed URL with a
   30 s timeout.
   - If the seed request fails (e.g. `403`), it raises `CollectionError` with the
     source name and URL. It does **not** retry, rotate identities, or try to
     solve a bot challenge.
   - If `--pages-per-source > 1`, **`_discover_links()`** parses the seed HTML and
     collects up to *N−1* additional URLs whose host exactly matches the seed
     host (`_same_site_link`), fragments stripped, in DOM order.
3. Each URL is fetched and passed to **`clean_html()`**:
   - Parse with BeautifulSoup (`html.parser`).
   - Read `<title>` for the document title.
   - `decompose()` every `script`, `style`, `nav`, `footer`, `header`, `aside`,
     `form`, `noscript`.
   - Pick the content root: first `<main>`, else `<article>`, else `<body>`, else
     the whole tree.
   - `get_text(" ")` and collapse all whitespace runs to single spaces.
4. Non-empty results become `SourceDocument`s. Secondary pages that fail are
   skipped silently; a failed **seed** aborts that source.
5. **`save_documents()`** writes each doc to
   `data/raw/<sha256(url)[:12]>.json` — pretty-printed, UTF-8, one file per page,
   meant to be opened and eyeballed before indexing.

The CLI ([`cli.py`](src/investment_rag/cli.py)) catches `CollectionError`,
records the source as "blocked", keeps going, and prints a reminder to swap in an
accessible page.

### Tradeoffs and caveats

| Area | Issue |
| --- | --- |
| **No JavaScript** | `requests` returns only server-rendered HTML. A client-side-rendered (CSR) SPA (single page app) hands back a nav shell with no article body, and `clean_html` then strips the shell down to nothing. (Both current live sources — ESMA, FCA — are server-rendered, so this doesn't bite *today*, but any React/Vue source would.) |
| **Bot protection** | MoneyHelper sits behind a Cloudflare challenge (`cf-mitigated: challenge`) and returns `403` to every non-browser client on every path. By design the collector reports and skips rather than bypassing — so that whole domain is simply uncollectable here. |
| **Content extraction is heuristic** | Whole tag *types* are removed. A page that puts its real content in an `<aside>`, or its nav inside `<main>`, gets mangled. The `<main>`→`<article>`→`<body>` fallback can grab far too much (entire body) or, as with FCA InvestSmart, land on an `<article>` that only holds ~1,600 characters of blurbs. |
| **Structure is destroyed** | Whitespace collapse flattens headings, lists, and paragraphs into one run-on string ("Risk Diversification spreads risk."). Downstream chunking has no section boundaries to respect. |
| **Boilerplate leaks** | CMS layout tokens (`primary_grey_background`, `white_background`, `grey_3_background` from ESMA's Drupal) survive as "text". Cookie/consent strings, "skip to content", breadcrumb labels can too. |
| **Titles are raw** | `<title>` often includes site furniture — `"InvestSmart | FCA"`, `"Investor Corner"` — with no way to override from `sources.json`. |
| **Link discovery is crude** | `_discover_links` takes the *first N* `<a href>`s in DOM order, which on most sites are the top nav / menu, not article links. No sitemap use, no URL-pattern allow-list, no relevance ranking. Raising `--pages-per-source` mostly collects menu pages. |
| **No politeness controls** | No `robots.txt` check, no crawl delay, no rate limiting, no conditional GET (`ETag`/`If-Modified-Since`), no on-disk HTTP cache. Every run re-downloads everything. |
| **Spoofed User-Agent** | The collector sends a Chrome UA string while being a script — a mild inconsistency with the project's "don't disguise access" stance. |
| **Weak fetch hygiene** | No `Content-Type` check (a PDF or image URL would be parsed as HTML), no response-size cap, no retry/backoff, fixed 30 s timeout, encoding left to `requests`' guess. |
| **Re-collect leaves orphans** | Filenames are keyed on the URL. Re-collecting the same URL overwrites its file, but if a source *drops* a URL, its stale JSON stays in `data/raw/` and gets re-indexed. |
| **No dedup** | Two seed pages that share a section produce two near-identical documents. |

### Room for improvement

- **Opt-in headless rendering** (Playwright/Selenium) for sources flagged as JS
  in `sources.json`, feeding rendered HTML into the same `clean_html`.
- **Better main-content extraction** with `trafilatura` / `readability-lxml` /
  `boilerpy3` instead of the tag-type heuristic.
- **Preserve structure**: emit Markdown-ish text (newline between blocks, keep
  `#` headings) so chunking can split on real boundaries.
- **Politeness layer**: `urllib.robotparser`, per-domain crawl delay, a
  `requests-cache` backend, conditional requests.
- **Sitemap-driven discovery** plus per-source URL-pattern allow-lists, instead
  of scraping anchors.
- **Richer provenance**: store `fetched_at`, HTTP status, canonical URL, content
  hash; let `sources.json` supply an explicit title.
- **Change detection**: hash content, and on re-collect delete files for URLs no
  longer produced by any source.
- **Near-duplicate detection** (SimHash/MinHash) before saving.
- A **local-file ingest path** (`collect --from-file page.html …`) for sources
  that can only be saved manually from a browser.

---

## 2. Chunking — [`chunking.py`](src/investment_rag/chunking.py)

### How it works

`chunk_document(document, max_chars=900, overlap_chars=150)`:

1. Reject `max_chars <= overlap_chars`.
2. Split `document.text` on sentence boundaries with the regex
   `(?<=[.!?])\s+` — "a period/!/? followed by whitespace".
3. Greedily pack sentences into a buffer. When adding the next sentence would
   push the buffer past `max_chars` (and the buffer isn't empty):
   - flush the buffer as a chunk,
   - reseed the buffer with the **last `overlap_chars` characters** of the chunk
     just flushed,
   - append the current sentence to that seed.
4. Flush whatever remains.
5. Give each chunk an id `<sha256(url)[:12]>-<index>` and copy all document
   metadata onto it.

`chunk_documents()` flattens this over a list of documents.

The overlap means consecutive chunks literally share ~150 characters of text (and
therefore overlapping embeddings), so a fact sitting on a chunk boundary still
appears whole in at least one chunk.

### Tradeoffs and caveats

| Area | Issue |
| --- | --- |
| **Characters, not tokens** | `max_chars=900` is unrelated to what the embedding model can encode. `all-MiniLM-L6-v2` truncates at **256 word-piece tokens** (~1,000–1,200 chars of typical English, less for dense/technical text). Some 900-char chunks are silently truncated at encode time — the tail is embedded as if it didn't exist. |
| **Overlap is a raw slice** | `current[-150:]` cuts mid-word and mid-sentence. Chunks then *start* with a fragment like `"...tion spreads risk. Your time horizon..."`, which is grammatically broken and slightly degrades the embedding of that chunk. |
| **Naive sentence splitting** | The regex splits on any `. `, so `"e.g. "`, `"U.K. "`, `"Fig. 1 "`, `"3.5% "`, `"No. "` all fracture sentences. Languages without a space after the period aren't split at all. |
| **Oversized first / lone sentences** | The flush condition requires a non-empty buffer, so the first sentence is always admitted regardless of length. A single sentence longer than `max_chars` becomes its own chunk, unsplit and over-limit. |
| **Ragged tail** | The last chunk can be a tiny dangling fragment or, if the final sentence is long, much bigger than the rest. No minimum-size merge. |
| **No structural awareness** | A chunk can span the end of one section and the start of an unrelated one (headings were flattened away upstream), mixing topics in a single vector. |
| **Thin position metadata** | Only `chunk_index`. No character offsets, no section heading — so you can't do sentence-window or parent-document retrieval, and a citation points at "chunk 7", not a location on the page. |
| **Cross-page duplicates** | Identical text on two pages yields two chunks with different ids (different URL prefix). Both get indexed; both can be retrieved for the same query. |
| **Stale chunks on rebuild** | Chunk ids are `url + index`. If a page's content shrinks, `build` upserts the new lower-index chunks but the old higher-index ids are never deleted (`upsert` doesn't remove). The index keeps serving text the page no longer contains. |
| **Id tied to exact URL** | Tracking params or `http`/`https` differences change the digest and produce a parallel set of chunks. |

### Room for improvement

- **Token-aware splitting** using the embedding model's own tokenizer (e.g. 200
  tokens, 40-token overlap), or move to a longer-context embedding model so
  900-char chunks fit comfortably.
- **Recursive / structure-aware splitting**: split on headings → paragraphs →
  sentences → characters, only descending when a piece is still too big.
- **Real sentence segmentation** (`syntok`, `blingfire`, spaCy, NLTK Punkt) to
  handle abbreviations and decimals.
- **Sentence-level overlap** — carry whole trailing sentences, not a byte slice.
- **Minimum chunk size** with a merge for the trailing fragment.
- **Store `char_start` / `char_end` and the nearest heading** in metadata; enable
  sentence-window and parent-document retrieval.
- **Track the doc→chunk-id set**; on rebuild, delete ids that disappeared.
- **Global dedup** of identical / near-identical chunks before indexing.
- Optionally **semantic chunking** (embedding-similarity boundary detection).

---

## 3. Retrieval — [`retrieval.py`](src/investment_rag/retrieval.py)

### How it works

- **`SentenceTransformerEmbedder`** wraps `all-MiniLM-L6-v2` (384-dim,
  English, general-purpose) and calls `model.encode(..., normalize_embeddings=True)`
  so every vector is unit length. The model (~90 MB) plus PyTorch download from
  Hugging Face on first use and are then cached.
- **`Retriever`** holds a `chromadb.PersistentClient` at `data/chroma` and a
  collection `investment_knowledge` created with `hnsw:space = "cosine"`.
- **`index(chunks)`** computes embeddings client-side and `upsert`s
  `ids`, `documents` (chunk text), `metadatas` (`Chunk.metadata()`), and
  `embeddings` together. Because embeddings are passed explicitly, Chroma's own
  embedding function is never invoked.
- **`search(question, limit=4, region=None)`** embeds the question, calls
  `collection.query(n_results=limit, where={"region": region} if region else None)`,
  and rebuilds `SearchResult(Chunk(...), distance)` from the returned
  `ids/documents/metadatas/distances`.
- The CLI prints each hit as `[source_name | region | url]` followed by the
  passage text.

The `where={"region": ...}` filter is applied by Chroma *during* the search, so a
`--region UK` query can never return an EU chunk — regional isolation is a hard
guarantee, not a post-filter.

### Tradeoffs and caveats

| Area | Issue |
| --- | --- |
| **Dense-only, one small model** | A single 384-dim bi-encoder vector per chunk. No lexical / BM25 channel, so exact terms the model doesn't represent well — `ISA`, `MiFID II`, `KIID`, fund names, percentages — can be missed entirely even when the phrase is right there in a chunk. |
| **Generic embedding model** | `all-MiniLM-L6-v2` is trained on general web/QA pairs, not finance or regulatory text, and caps at 256 tokens. Domain terms and long passages are its weak spots. |
| **No relevance threshold** | `search` always returns `limit` results. Ask something the corpus can't answer and you still get 4 passages at cosine distance ≈ 1 (i.e. unrelated). A downstream reasoning agent could ground a confident answer on noise. There's no `--min-score` / "no good match" path. |
| **No re-ranking** | Top-k order is raw approximate-nearest-neighbour similarity. A cross-encoder over the top ~20 would reorder these meaningfully. |
| **No diversity** | Overlapping chunks and repeated boilerplate can occupy several of the `k` slots. No MMR / dedup on results. |
| **Approximate search** | HNSW recall is < 100 % and parameters (`M`, `ef`) are untuned defaults. Fine at the current corpus size (tens of chunks); matters at scale. |
| **Region filter is exact-match and fragile** | `{"region": "UK"}` must match the stored value byte-for-byte. A typo in `sources.json` (`"uk"`, `"U.K."`) silently removes a source from all regional queries. Only `EU` / `UK` exist; there's no "either" mode (though `None` = unfiltered works). |
| **Metadata assumed present** | Result construction does `metadata["region"]`, `metadata["title"]`, etc. with no `.get`. A row missing a key would `KeyError`. Collection currently guarantees them, so this is latent. |
| **Cold start every call** | Each CLI invocation loads PyTorch + the model and re-embeds the query. No warm service, no query-embedding cache. |
| **No query processing** | The question is embedded as-is. No lowercasing, expansion, HyDE, or multi-query. User phrasing has to resemble the source phrasing. |
| **Unpinned model** | `all-MiniLM-L6-v2` is fetched by name, revision unpinned — the upstream artifact could change. Also a network + supply-chain dependency on first run. |
| **Single mutable collection** | `build` always upserts into the same collection. No snapshot/versioning, so you can't compare two chunking or embedding strategies side by side, and a bad rebuild pollutes the only copy. |
| **Limited filters** | You can filter by `region` but not `source_name`, language, or recency; no pagination. |
| **Distances aren't scores** | Chroma returns cosine *distance* (`0`–`2`); it's printed by tests but not surfaced or calibrated for the user. |

### Room for improvement

- **Hybrid retrieval**: add BM25 (or SPLADE) and fuse with the dense results via
  reciprocal-rank fusion.
- **Stronger / domain embeddings**: `bge-small-en-v1.5`, `e5-small-v2`,
  `gte-small`, or a finance-tuned model; a long-context model
  (`nomic-embed-text`, `jina-embeddings-v3`) removes the truncation problem.
- **Cross-encoder re-ranker** (`bge-reranker-base`, `mxbai-rerank`) over the top
  ~20 → return the best 4.
- **MMR / result dedup** so `k` slots hold `k` distinct ideas.
- **Score threshold + explicit "no confident answer"** return, and print the
  score (`1 - distance`) so callers can reason about confidence.
- **Persist the embedder** as a lazy singleton or a small local service; cache
  query embeddings; consider ONNX / int8 for latency.
- **Pin the model revision** and verify its hash (or vendor it).
- **Validate `region`** at `load_sources` time (enum), and add `source_name` /
  language / `fetched_at` filters plus a multi-region OR mode.
- **Tune HNSW** (or use exact search while the corpus is tiny).
- **Evaluation harness** (Phase 4): a labelled question → expected-passage set,
  scored with recall@k, MRR, and a groundedness check, run in CI.

---

## Cross-cutting notes

- **Provenance** is structurally enforced (frozen dataclasses, `metadata()`), which
  is the strongest part of the design. The weak links are *title quality* and
  *URL canonicalisation*, not whether metadata is present.
- **Idempotency vs staleness**: deterministic ids make re-runs safe to repeat but
  also mean removed content is never evicted. A doc→chunk-id manifest per source
  would fix both collect and build.
- **Testing**: [`tests/`](tests/) covers cleaning, chunking, provenance, and
  search wiring offline; `tests/test_integration.py` exercises real chunking and
  a real on-disk Chroma store with a deterministic fake embedder. Nothing tests
  the *real* embedding model or real network fetches — deliberately, to keep the
  suite fast and offline.
- **Determinism**: chunk ids and cleaning are deterministic; HNSW search is
  approximate, so retrieval results can shift slightly across Chroma versions or
  index rebuilds.
- **Security surface**: HTML from untrusted pages is parsed (BeautifulSoup, no
  code execution); discovered links are restricted to the seed's exact host, so
  there's no obvious SSRF pivot; JSON loaded is the project's own output.
