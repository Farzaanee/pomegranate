# Performance log

Cost and time tracking for agent-driven work sessions on this repo, so effort,
token spend, and wall-clock time are visible over time rather than anecdotal.

## Methodology

Claude Code writes one JSONL transcript per session to
`~/.claude/projects/<project>/<session-id>.jsonl`; each assistant turn logs its
`model`, `effort`, `requestId`, and a `usage` object (`input_tokens`,
`output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens` split
by TTL). A run's entry below is computed by filtering that file to the turns
between the triggering message and the last tool call before the summary, then:

- **Requests** — unique `requestId` values in that window (duplicate JSONL
  lines share a `requestId` and are counted once).
- **Tokens** — summed straight from `usage` per request.
- **Cost** — token counts × Anthropic's published per-model rate (input,
  output; cache write at 1.25× input for a 5-minute TTL or 2× for 1-hour;
  cache read at 0.1× input). Rates used for Claude Sonnet 5 and 5.5 (same price): $2.00 / $10.00 per
  1M input/output tokens.
- **Wall time** — the timestamp of the tool call taken as the run's start
  point vs. the last one before the final summary.

**Known undercount:** the measurement is taken from inside the run, one or two
tool calls before the final summary, so it excludes that summary's own output
tokens (typically a few hundred) and the seconds spent generating it. Treat
each entry as a slight underestimate, not an exact total.

## Log

| Date | Task | Model | Effort | Requests | Input (fresh) | Output | Cache read | Cache write | Est. cost | Wall time |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-09-05 | Phase 2 implementation (`profile.py`, `reasoning.py`, `advise` CLI command, Streamlit integration, tests, docs) + this logging setup + the architecture diagram artifact | claude-sonnet-5 | xhigh | 70 | 140 | 101,664 | 9,782,165 | 207,917 (all 1h TTL) | ~$3.81 | ~19.5 min |
| 2026-09-28 | Phase 3 implementation (region-leak defense in `gather_evidence`, `explain_region_difference` EU/UK comparison agent, `compare` CLI command, Streamlit "Compare EU vs UK" mode, tests, docs) | claude-sonnet-5 | default | 45 | 90 | 31,771 | 4,282,866 | 86,987 (all 1h TTL) | ~$1.52 | ~5.2 min |
| 2026-10-11 | Phase 4 implementation (`evaluation.py`, 12-case eval set, `evaluate` CLI command, live eval runs, fixes for the defects they found in `reasoning.py` / `profile.py`, Streamlit "Evaluation results" mode, tests, `evaluation.md` + doc updates) | claude-sonnet-5-5 | high | 37 | 74 | 63,904 | 4,965,066 | 189,326 (all 1h TTL) | ~$2.39 | ~17.7 min |

### Reading the first entry

- **Output-heavy relative to fresh input** (101.7k vs. 140 tokens) is expected
  for a coding task: most "input" was already-cached system prompt, skill
  content, and file reads rather than new tokens each turn.
- **Cache read dwarfs everything else** (9.78M tokens) because this session
  loaded several large skill documents (`claude-api`, `artifact-design`,
  `artifact-diagramming`) and re-sent the accumulating conversation prefix on
  every one of 70 requests; at 0.1× input price this is the cheapest token
  category by far ($1.96 of the $3.81 total) despite being the largest count.
- **70 requests** for one task reflects the harness's per-tool-call request
  pattern (a request per tool use / turn), not 70 separate user prompts.

### Reading the second entry

- **Much cheaper and faster than Phase 2** (~$1.52 vs. ~$3.81, ~5.2 vs. ~19.5
  min) for a comparably scoped change: this session ran no long-running
  collection or embedding work and touched fewer, already-familiar files, so
  fewer requests (45 vs. 70) were needed to reach the same "implemented,
  tested, documented" bar.
- **Cache read is again the dominant token count** (4.28M) for the same reason
  as the first entry — the accumulating conversation prefix is re-sent every
  request — and again the cheapest category by far at 0.1× input price
  (~$0.86 of the ~$1.52 total).
- Wall time is measured from the triggering message to the last tool call
  before this log entry was written, so — per the methodology note above — it
  slightly undercounts the time actually spent on the task.

### Reading the third entry

- **Cost split:** ~$0.99 cache reads, ~$0.76 cache writes, ~$0.64 output
  (63.9k tokens) of the ~$2.39 total. Cache reads are 4.97M tokens for the same
  reason as before — the growing conversation prefix is re-sent on each of the
  37 requests — and this session also loaded two large skill documents
  (`claude-api`, `developing-with-streamlit`).
- **Cheaper than Phase 2 (~$3.81), dearer than Phase 3 (~$1.52)**, with the most
  wall time of the three after Phase 2. About 8 of the 17.7 minutes were not
  Claude Code working but waiting on the project's own live evaluation runs
  (4 min 38 s for the first, 3 min 37 s for the second), which is why wall time
  is high relative to the request count.
- **Not included — the project's own API spend.** The two full `evaluate` runs
  (12 Claude calls each) and one 4-case replay made roughly 28 calls to
  `claude-sonnet-5` using the project's own key. That spend is billed to that
  key, not recorded in this transcript, and was not measured. An upper bound,
  assuming every call used ~5k input and ~4k output tokens at $2 / $10 per 1M,
  is about $1.4; the second run (thinking disabled) used far fewer output
  tokens, so the real figure is likely well below that.
- **Effort was `high`** (the harness setting for this session), compared with
  `xhigh` for Phase 2 and `default` for Phase 3, so the three rows are not a
  like-for-like effort comparison.
