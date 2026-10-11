# Implementation notes

> **Educational only — not regulated financial advice.** This project retrieves information from official sources; it does not recommend specific investments or replace professional advice.

What was actually built in each phase, how it works, and what was learned along
the way. The *plan* — goals, scope, risks, and each phase's objectives and
acceptance criteria — lives in [project-plan.md](project-plan.md); the system
diagram and module reference are in [architecture.md](architecture.md); the
evaluation method and results are in [evaluation.md](evaluation.md).

Each phase has three parts: **what was built**, the **acceptance criterion** as
it was met, and **learnings**. A closing section collects the lessons that cut
across phases.

> **Provenance of the learnings.** Phase 4's come from the evaluation runs
> themselves. Phases 1–3 are reconstructed from the commit history and the
> project's own docs ([data-sources.md](data-sources.md),
> [web-concepts.md](web-concepts.md)), not from a diary kept at the time —
> they record what the repository shows was fixed or decided, and are worth
> amending with your own account of what you learned about investing and about
> the tooling.

---

## Phase 1: local knowledge base

The included Python pipeline collects trusted public pages, removes site chrome, chunks readable content, and stores embeddings plus provenance in local Chroma storage. Every retrieved passage retains its source name, region, original URL, title, and chunk index.

### Setup

Requires Python 3.10 or later; the repo pins 3.12 via `.python-version`, which is the
version tested against and the one with the widest prebuilt-wheel coverage on
Streamlit Community Cloud. Avoid pre-release interpreters — `chromadb` and
`onnxruntime` ship no wheels for them.

Using [uv](https://docs.astral.sh/uv/) (recommended; a `uv.lock` is committed):

```bash
uv sync --extra dev   # downloads Python 3.12 if needed, builds .venv/, installs the project
uv run investment-rag collect --pages-per-source 1
uv run investment-rag build
uv run investment-rag query "Why does diversification matter when investing?" --region UK
uv run pytest
```

Or with the standard library `venv`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
investment-rag collect --pages-per-source 1
investment-rag build
investment-rag query "Why does diversification matter when investing?" --region UK
pytest
```

The `all-MiniLM-L6-v2` ONNX weights (~80 MB, no PyTorch) download on first use and are cached. `data/raw/` and the built index `data/index/` are committed so the deployed app opens a prebuilt index; `data/chroma/` is gitignored local scratch. Increase `--pages-per-source` only after checking each publisher's terms, robots policy, and the collected JSON documents in `data/raw/`. A publisher may return `403 Forbidden`; the collector will report and skip that source rather than attempting to bypass access controls. Replace its entry in `sources.json` with an accessible official page, then collect again.

### Pipeline

1. `sources.json` is the allow-list of 51 curated seed pages from ESMA Investor Corner (EU), FCA InvestSmart (UK), and FCA Consumer Warnings (UK); MoneyHelper was dropped because it returns `403` to the collector (see [data-sources.md](data-sources.md)).
2. `investment-rag collect` downloads seed pages and optionally a limited number of same-domain links, then saves cleaned documents as reviewable JSON.
3. `investment-rag build` creates overlap-preserving chunks and upserts them into persistent Chroma storage.
4. `investment-rag query` performs semantic retrieval and prints the supporting text with source and URL. Use `--region EU` or `--region UK` to prevent cross-region retrieval.
5. `investment-rag advise` (Phase 2) derives evidence queries from a user profile, retrieves region-scoped passages via the same retriever, and asks Claude to produce a plain-language, cited recommendation — see the Phase 2 section below.

The test suite validates cleaning, chunking, and required provenance metadata offline. A successful sample query that returns a relevant passage carrying its official source, region, and URL meets the Phase 1 retrieval criterion.

### Learnings

- **The regulators are supervisors, not educators.** ESMA's Investor Corner is
  about eight HTML pages, and FCA's sitemap lists 43,142 URLs of which only a few
  dozen are written for retail investors (the rest are news and firm-facing
  regulation). Crawling whole domains would have flooded the index with noise, so
  the knowledge base is a curated allow-list, not a crawl
  ([data-sources.md](data-sources.md)). The first index, built from a two-seed
  one-hop crawl, was roughly half off-topic (board governance, careers,
  `/privacy`, RSS feeds) until it was replaced by the 51-page curated set.
- **Respect access controls; don't route around them.** MoneyHelper, the richest
  UK consumer-investing source, returns `403 Forbidden` to the collector. The
  collector reports and skips such sources rather than bypassing them, and the
  source was replaced with accessible official pages. The cost is a thinner UK
  corpus than hoped for.
- **`requests` is not a browser.** A client-side-rendered page arrives as a nav
  shell with an empty content div; after chrome removal almost nothing is left
  ([web-concepts.md](web-concepts.md)). Candidate pages were therefore checked
  for HTTP 200, an HTML content type, and at least ~400 characters of body text
  before being admitted.
- **Build the index offline, not at startup.** The first Streamlit version
  chunked and embedded the whole corpus on every cold start, which made the app
  slow to open. Committing `data/index/` and only embedding the user's query at
  search time fixed the load time. The index is copied to a temp directory
  because Chroma writes journal files on open and the deployment checkout is
  read-only.
- **No PyTorch.** ONNX Runtime's MiniLM (the same `all-MiniLM-L6-v2` model)
  removes roughly 450 MB from the install, which is what makes the Streamlit
  Community Cloud free tier (1 GB RAM) viable.
- **Known gap.** ESMA publishes ~50 investor factsheets and warnings as PDFs, and
  the collector only reads HTML; a PDF loader would roughly double the EU corpus.

---

## Phase 2: reasoning layer

See [architecture.md](architecture.md) for the system diagram and request-flow
walkthrough. Implements the Phase 2 plan below on top of the Phase 1 pipeline:

- `src/investment_rag/profile.py` — the `UserProfile` schema (income, investable amount, goal, timeline, risk tolerance, region) and `retrieval_queries()`, which turns a profile into the sub-queries used to gather evidence (horizon, risk tolerance, fees, goal, debt and emergency savings as of Phase 4, and a region-specific query — ISA vs. general account for UK, MiFID II protections for EU).
- `src/investment_rag/reasoning.py` — `gather_evidence` runs those queries through the Phase 1 `Retriever` with `region` locked to the profile's region, dedupes by chunk id, labels passages `"1"`, `"2"`, … for citation, and stops once `MAX_EVIDENCE_PASSAGES` (10) unique passages are collected (taking results round-robin across queries as of Phase 4, so no query is starved by the cap). Each passage is truncated to `MAX_PASSAGE_CHARS` in the prompt to keep prefill time down. `ClaudeRecommendationLLM` calls Claude (`claude-sonnet-5` by default, favoring latency over `claude-opus-5`) with `output_config.format` (a JSON schema), so the response is a validated `{summary, suitable_options, reasoning, risks, caveats, citations}` object rather than free text. The system prompt caps each array (4 suitable options, 5 reasoning steps, 5 risks, 5 caveats, 8 citations); since Anthropic's `json_schema` format rejects `maxItems`, `parse_recommendation` re-enforces the same caps by slicing. The prompts frame the task as educational suitability analysis over a synthetic profile — which vehicle *types* fit and why, never a product, an amount, or an instruction to act — which is what stops Claude from softening or partially refusing the answer. `parse_recommendation` then resolves each citation's label against the actual retrieved passages and **drops any that don't match** — the model can invent a label, but a fabricated citation cannot survive into the returned `Recommendation`. If none survive, it raises `ReasoningError` rather than returning an ungrounded answer.
- `ReasoningAgent.run(profile)` ties it together: gather evidence → prompt → validate → return.
- Exposed via `investment-rag advise` (CLI) and the "Grounded recommendation" mode in `app.py` (Streamlit), gated on an `ANTHROPIC_API_KEY` being configured.
- `tests/test_profile.py` and `tests/test_reasoning.py` cover validation, query derivation, evidence dedupe/region-scoping, and citation validation — all against fakes, so the suite stays offline and free.

**Acceptance criterion met:** given a sample profile (e.g. UK, medium risk, 15-year horizon, retirement goal), `advise` returns a coherent, plain-language recommendation whose citations resolve to real passages from region-matched official sources — verified end-to-end against the committed Phase 1 index.

### Learnings

- **Framing decides whether Claude answers at all.** The first system prompt
  (an "educational investment-literacy assistant" with a disclaimer up front)
  did not reliably produce a usable answer; the fix was reframing the task as
  suitability *education* over *synthetic* profiles — categories of vehicle and
  general principles, never a product, an amount, a return forecast, or an order
  to act. The reframing also changed the output schema from generic
  `reasoning_steps` / `considerations` to per-option `suitable_options`, each
  carrying its own citation labels.
- **Latency is a design constraint, not a polish item.** A first working version
  was too slow, and the fix was several small caps rather than one big change:
  default model `claude-sonnet-5` instead of `claude-opus-5`, `max_tokens` cut
  from 8,000 to 4,000, at most 10 evidence passages truncated to 600 characters
  each, and per-array caps on the response (4 options, 5 reasoning steps, 5
  risks, 5 caveats, 8 citations).
- **Schema limits shape the validation code.** Anthropic's `json_schema` output
  format rejects `maxItems`, so the array caps are stated in the prompt and
  enforced again by slicing in `parse_recommendation`.
- **Never trust a model's citation labels.** The model can invent a label, so
  every label is resolved against the passages actually retrieved, invented ones
  are dropped, and a response left with none raises `ReasoningError`. The
  standing disclaimer is appended in code, not left to the model.
- **Fakes make the suite fast but can't see model behaviour.** All Phase 2 tests
  run against a fake retriever and fake LLM, so they pass offline and free. They
  also could not catch the defects Phase 4 later found in real calls (below).

---

## Phase 3: multi-jurisdiction logic

Adds explicit multi-jurisdiction routing on top of the region locking Phase 2 already
applied at the retriever level:

- `gather_evidence` (`src/investment_rag/reasoning.py`) now re-checks every retrieved chunk's
  `region` field against `profile.region` before keeping it, instead of trusting the Chroma
  `where` filter alone. This is the explicit routing/classification step: it holds even if the
  retriever's own filter is ever misconfigured or bypassed, and is covered by
  `test_gather_evidence_drops_out_of_region_results_even_if_the_retriever_leaks_them`, which
  exercises a deliberately leaky fake retriever.
- `explain_region_difference` (`reasoning.py`) is new agent functionality for the second Phase 3
  acceptance criterion — explaining a key EU/UK difference when asked directly. Given a topic, it
  runs `gather_region_comparison_evidence`, which queries each region separately and labels
  passages `"EU-1"`, `"UK-1"`, … by the region they were retrieved under (rather than a shared
  counter), so the region a citation came from is legible in the prompt itself. `ClaudeRecommendationLLM.compare` asks Claude (same model choice as Phase 2) for a
  `{eu_summary, uk_summary, key_difference, citations}` object. `parse_region_difference` then
  validates citations the same way Phase 2's `parse_recommendation` does, plus one more check
  specific to the comparison: a citation is only kept if its label's region prefix (`"EU-"` /
  `"UK-"`) actually matches the region of the chunk it resolved to, so the model can't launder an
  EU claim under a UK-looking label or vice versa.
- Exposed via `investment-rag compare <topic>` (CLI) and the "Compare EU vs UK" mode in `app.py`
  (Streamlit), both gated on `ANTHROPIC_API_KEY` like the Phase 2 recommendation mode.
- `tests/test_reasoning.py` adds the region-leak test above, a same-profile-under-both-regions
  comparison scenario (`test_same_profile_evaluated_under_both_regions_never_mixes_sources`), and
  coverage for the comparison pipeline's evidence gathering, citation validation, and empty-evidence
  error path.

**Acceptance criterion met:** the agent only cites region-appropriate sources for a given user —
enforced in code, not just by the retriever's filter — and `investment-rag compare` explains a
concrete EU/UK difference (e.g. ISA wrappers vs. MiFID II protections) with citations traceable to
region-matched official sources.

### Learnings

- **Defence in depth beats trusting one filter.** Phase 2 already passed
  `region` to Chroma's `where` filter. Phase 3 re-checks every result's region in
  `gather_evidence` anyway, and tests it with a deliberately leaky retriever, so
  a store or filter bug cannot put an EU source into a UK answer.
- **Make provenance legible in the prompt.** Labelling comparison passages
  `EU-1`, `UK-1` (by region) rather than `1`, `2` (a shared counter) lets the
  model, the validator, and a human reading the prompt see which jurisdiction a
  citation belongs to.
- **Some safeguards are insurance, not behaviour.** Because each label maps to
  one fixed chunk, the `EU-`/`UK-` prefix check in `parse_region_difference`
  cannot fail with the current labelling; it protects against a future change to
  how labels are built. For the same reason the Phase 4 evaluation does not score
  it.
- **The corpus is lopsided, which limits comparisons.** The index holds 43 UK
  pages (FCA InvestSmart and Consumer Warnings) against 8 EU pages (ESMA). A
  comparison can only be as good as the thinner side's evidence, so the EU half
  of an answer is more likely to rest on a single generic passage. (This is an
  inference from the page counts, not something the evaluation measured.)
- **Same profile, both regions, is a good test shape.** Running one profile under
  each region and asserting the evidence sets never mix catches routing bugs that
  a single-region test cannot.

---

## Phase 4: evaluation and polish

Adds the evaluation harness, fixes the defects it surfaced, and finishes the documentation.
See [evaluation.md](evaluation.md) for the full method, results, and limitations.

- `src/investment_rag/evaluation.py` runs an eval set through the product's own evidence-gathering,
  prompt, and parsing functions but keeps the **raw model payload**, so it measures what the model
  *tried* to cite rather than what validation cleaned up. Deterministic checks cover groundedness
  (every cited label exists, every cited quote is a faithful excerpt of its passage), region purity,
  the standing disclaimer, and good-advice hygiene (no product names, hype, return forecasts, or
  amounts to invest; soft rates for fee and debt awareness). `evals/cases.json` holds 12 cases:
  8 EU/UK profiles and 4 EU-vs-UK comparison topics.
- `investment-rag evaluate` (CLI) runs it; `--retrieval-only` needs no API key; `--output` writes a
  JSON report, and a full run exits non-zero below the 90% bar. The Streamlit app gains an
  "Evaluation results" mode that renders the stored `evals/results/latest.json`.
- Defects the first live run exposed, now fixed: thinking tokens could truncate the JSON answer
  (thinking disabled, truncation raises a readable `ReasoningError`); `gather_evidence` starved its
  later queries, including the region-specific one, under the 10-passage cap (now round-robin);
  and debt guidance was neither retrieved nor reliably requested (new retrieval query plus an
  explicit caveat instruction). Two checker rules that wrongly failed faithful output were relaxed.
- `tests/test_evaluation.py` (offline, fakes only) covers each check's pass and fail path, the
  quote-matching edge cases, the threshold, and that the committed eval set stays valid.

**Acceptance criteria:** *Evaluation set passed* — 12 of 12 cases (100%, bar 90%) on the final live
run, after 4 of 12 on the first; the final figure is not held out, because checks and product were
adjusted after the first run, and it is one run of a non-deterministic model. *Demoable interface* —
the Streamlit app runs end-to-end in four modes. *Documentation* — README, architecture,
evaluation, and setup docs are current, and the README and evaluation doc state the not-financial-advice disclaimer.

### Learnings

Full detail and numbers are in [evaluation.md](evaluation.md); in short:

- **Measure the raw model output, not the validated result.** Validation drops
  invented citations before the user sees them, so scoring only its output would
  show a 0% hallucination rate by construction. The harness keeps the raw
  payload and counts what the model *attempted* to cite. (On the live runs it
  never invented a label.)
- **An evaluation finds what unit tests with fakes cannot.** 32 offline tests
  passed while the real product had a truncation bug (thinking tokens share the
  `max_tokens` budget with the JSON, so a long answer could be cut off
  mid-string) and a retrieval bug (the 10-passage cap was filled one query at a
  time, so the goal and region-specific queries contributed nothing). Both only
  appeared once real model calls and the real index were involved.
- **Read each failure before fixing it.** The first run passed 4 of 12 cases. The
  failures were five different findings — two product bugs, a quality gap (debt
  mentioned in 2 of 6 recommendations), and two over-strict checks (faithful
  quotes rejected because the model dropped scraped CSS tokens or completed a
  word a chunk boundary had cut; "no guarantee" and a description of scammers'
  promises flagged as hype). Treating all of them as "the model is wrong" would
  have meant tuning the wrong thing.
- **Adjusting after you look contaminates the result.** The final 12/12 comes
  after changing both the product and the checks in response to the first run,
  on the same 12 cases, in a single run of a non-deterministic model. It is a
  smoke test, not a benchmark, and the docs say so.
- **Deterministic checks are cheap and traceable but shallow.** A verbatim quote
  shows the model pointed at a real passage, not that its summary is faithful,
  and regex language checks have false negatives. A claim-faithfulness check (LLM
  judge or human spot-check) is the next step.
- **Opening a store can modify it.** Running the evaluation against the
  committed `data/index/` made Chroma rewrite its files; the command now works on
  a temp copy.
- **Data quality problems surface late.** Scraper residue such as
  `primary_grey_background` in the cleaned ESMA text only became visible when
  the model's quotes disagreed with the passages. The cleaning step should be
  fixed and the index rebuilt; this is recorded but not done.
- **Wall-clock time is mostly waiting on the model.** A 12-case live run takes
  3–5 minutes (~18–23 seconds per call), which is why the evaluation has a free
  `--retrieval-only` mode for quick checks.

---

## Learnings across phases

- **Validate in code; never trust the model or a single filter.** Citation
  labels, region labels, the disclaimer, and array sizes are all enforced after
  the model responds. The recurring pattern: assume the layer below can be wrong,
  and check at the boundary.
- **Scope the source set before scaling the pipeline.** Source selection (what
  the regulators actually publish, what blocks the collector, what the HTML
  contains) shaped the project more than model choice did.
- **Tune for latency early.** Model choice, token caps, and evidence caps were
  all latency decisions made once a real call was slow.
- **Test against the real thing at least once.** Fakes are right for the daily
  suite; the evaluation is how the real model and real index get exercised.
