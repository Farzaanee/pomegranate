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
  cache read at 0.1× input). Rates used for Claude Sonnet 5: $2.00 / $10.00 per
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
