# Evaluation

How Phase 4 measures whether the agent's answers are grounded, what the
numbers do and do not establish, and what the first runs found. The code is
[`evaluation.py`](../investment_rag/evaluation.py); the eval set is
[`evals/cases.json`](../../evals/cases.json); the latest stored report is
[`evals/results/latest.json`](../../evals/results/latest.json).

> Educational only, not regulated financial advice. Passing this evaluation
> shows the agent *cites real passages and avoids a list of known bad habits*.
> It does not show the output is good investment guidance.

## Running it

```bash
# Full run: calls Claude once per case (~12 calls, a few minutes, small cost). Needs ANTHROPIC_API_KEY.
uv run investment-rag evaluate --output evals/results/latest.json

# Free run: retrieval and region routing only; no API key, no model calls.
uv run investment-rag evaluate --retrieval-only
```

A full run exits non-zero if fewer than 90% of cases pass (`PASS_THRESHOLD`),
so it can gate CI. The command evaluates a throwaway copy of `data/index/`,
because Chroma rewrites its journal files on open and the committed index must
stay byte-identical. The Streamlit app's **Evaluation results** mode renders
the stored report without needing a key.

## How a case is run

Each case goes through the same functions the product uses (`gather_evidence`,
the prompt builders, `parse_recommendation` / `parse_region_difference`), but
the harness keeps the **raw model payload** that parsing would otherwise clean
up. That matters: `parse_recommendation` silently drops a citation whose label
was never retrieved, so measuring only its output would always show a 0%
hallucination rate. The harness scores what the model *attempted to cite*.

The eval set has 12 cases: 8 synthetic profiles (4 UK, 4 EU; retirement,
house deposit, general growth, emergency fund; low to high risk; 1 to 20 year
horizons; low to higher income) and 4 EU-vs-UK comparison topics (investor
protections, checking a firm is authorised, costs and fees, scams).

## What is checked

A case **passes** when it did not abort and every *hard* check passed. *Soft*
checks are reported as rates but never gate a case.

| Check | Kind | What it means |
| --- | --- | --- |
| `evidence_retrieved` | hard | Retrieval found something to ground an answer in. |
| `region_pure_evidence` | hard | Every retrieved passage belongs to the profile's region (comparisons: its label's region). |
| `both_regions_retrieved` | hard | Comparison only: both EU and UK evidence was found. |
| `no_fabricated_citations` | hard | Every citation label the model emitted exists in the retrieved evidence. **This is the hallucination rate.** |
| `quotes_supported` | hard | Every cited quote is a faithful excerpt of the passage it cites (see below). **This is citation accuracy.** |
| `has_citations` / `parses_with_valid_citations` | hard | At least one citation survives validation. |
| `citations_in_profile_region` | hard | Recommendation citations all come from the profile's region. |
| `options_cited` | hard | Every suitable option carries at least one valid citation label. |
| `cites_both_regions` | hard | A comparison cites at least one EU and one UK source. |
| `disclaimer_present` | hard | The standing disclaimer is the final caveat. |
| `no_product_names` | hard | No provider or ticker from a deny-list (the prompt forbids naming products). |
| `no_hype` | hard | No asserted "guaranteed", "risk-free", "get rich"-style language. |
| `no_return_projection` | hard | No specific forecast such as "will grow 7%". |
| `no_amount_to_invest` | hard | No "invest £500" or "put 60% in" instruction. |
| `retrieval_on_topic` | soft | Retrieved text contains at least one of the case's expected terms. |
| `mentions_fees` | soft | The caveats, risks, or reasoning mention fees/costs/charges. |
| `mentions_debt` | soft | They mention debt, borrowing, loans, or credit cards. |

The last two soft checks exist because the project brief defines good advice as
including *fee awareness* and *checking debt before investing*. They are soft
because a short emergency-fund answer can reasonably omit one.

**Quote matching.** A quote is supported when at least 90% of its words appear,
in order, in the cited passage, after lowercasing, stripping punctuation, and
discarding scraper residue such as `white_background`. An ellipsis splits a
quote into fragments that must each match. A passage that was cut mid-word at a
chunk boundary ("learly") is completed from the quote before comparing. An
invented sentence fails; a faithful excerpt with a dropped formatting token
passes.

**Language checks are heuristics.** A hype word is ignored when a negation
precedes it ("no guarantee") or a scam/claim/warning word is near it ("promises
like guaranteed profits"), because the model is then reporting speech, not
making it. Regexes will have false negatives, for example hype phrased in words
the lists don't contain.

## What this does not establish

- **Faithfulness of the claim.** A verbatim quote proves the model pointed at a
  real passage, not that its summary of that passage is accurate. There is no
  LLM judge and no human review of the prose.
- **Quality of the advice.** Nothing here checks that the vehicle types suit the
  profile, only that the answer is cited, regional, and free of listed
  problems.
- **Statistical confidence.** 12 cases and a single run of a non-deterministic
  model. Treat the rates as a smoke test, not a benchmark; one run's pass rate
  can move by several cases between runs.
- **Out-of-sample performance.** The checks and the product were adjusted after
  the first run (below), so the final 12/12 is not a held-out result.

## Results

Both runs: `claude-sonnet-5`, 12 cases, the committed index of 51 pages.

| Metric | Run 1 (baseline) | Run 2 (after fixes) |
| --- | ---: | ---: |
| Cases passed | 4 / 12 | **12 / 12** |
| Groundedness rate (bar: 90%) | 33% | **100%** |
| Citation label validity (hallucinated citations) | 100% (0%) | 100% (0%) |
| Quote support | 92.9% | 100% |
| Region match | 100% | 100% |
| Retrieval hit rate | 100% | 100% |
| Fee awareness (soft) | 100% | 100% |
| Debt awareness (soft) | 33% | 87.5% |
| Cases aborted by an error | 2 | 0 |
| Wall time for the run | 4 min 38 s | 3 min 37 s |

### What the first run found

The baseline failed 8 of 12 cases. Reading each failure separately showed they
were three different kinds of problem:

| Finding | Cause | Fix |
| --- | --- | --- |
| 2 cases aborted with `JSONDecodeError` | **Product bug.** The model was thinking before answering, and thinking tokens share the 4,000-token `max_tokens` budget with the JSON; replaying one of the aborted cases used 3,989 of the 4,000, so any slightly longer answer is cut off mid-string. | Extended thinking disabled on both Claude calls (the task is schema-constrained extraction over supplied passages), and a `max_tokens` stop now raises a readable `ReasoningError`. |
| Evidence never included the goal or region-specific queries | **Product bug, found while preparing the debt fix.** `gather_evidence` filled its 10-passage cap one query at a time (3 + 3 + 3 + 1), so the later queries, including "ISA vs general account" and "MiFID II protections", contributed nothing. | Results are now taken round-robin: every query's best hit before any query's second hit. |
| Debt mentioned in only 2 of 6 recommendations | **Quality gap.** The prompt asked for a debt caveat only loosely and no query retrieved debt guidance. | New retrieval query ("paying off debt and building emergency savings before investing") and an explicit caveat instruction in the system prompt. |
| 5 cases failed `quotes_supported` | **Checker too strict.** The flagged quotes were faithful: the model had dropped CSS residue (`white_background`) from the ESMA text, or completed a word the chunk boundary had cut ("learly" to "clearly"). | Quote matching made tolerant of exactly those two things, with tests that an invented sentence still fails. |
| 2 cases failed `no_hype` | **Checker too strict.** One negated the word ("no guarantee"); the other was describing what scammers promise ("guaranteed AI trading profits"). | Negation and scam-context exemptions, with a test that a plain assertion is still caught. |

No run, before or after, produced a fabricated citation label: the model never
cited a passage number it was not given.

## Known limitations

- **Scraper residue in the knowledge base.** Cleaned ESMA pages still contain
  tokens like `primary_grey_background`. They pollute chunk text and embeddings.
  The fix belongs in `collect.clean_html` followed by a rebuild and re-commit of
  `data/index/`; it was out of scope for Phase 4 and is not done.
- **Heuristic language checks.** See above; they are a floor, not a guarantee.
- **Debt awareness is 87.5%, not 100%.** One UK retirement case did not mention
  debt. It is soft by design and is left visible in the report.
- **The eval set is small and hand-written.** Growing it, adding adversarial
  profiles, and an LLM-judge or human spot-check of claim faithfulness are the
  obvious next steps.
