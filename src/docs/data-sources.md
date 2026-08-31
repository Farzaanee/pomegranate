# Data sources

What the knowledge base is built from, how much investor-education material each
publisher actually has, and which pages [`sources.json`](sources.json) pulls.

Scope: **retail investor education** in the EU and UK — what investing is, risk and
diversification, costs, checking that a firm is authorised, and the scam patterns
aimed at investors. Not firm-facing regulation, enforcement, or market data.

## The publishers

| Publisher | Region | Role | Investor-education footprint |
| --- | --- | --- | --- |
| **ESMA** — European Securities and Markets Authority (`esma.europa.eu`) | EU | EU markets regulator | **Small.** One "Investor Corner" hub (~8 HTML pages) plus a library of ~50+ investor factsheets/warnings as PDFs. The rest of the site is technical material for firms and national regulators. |
| **FCA** — Financial Conduct Authority (`fca.org.uk`) | UK | UK conduct regulator | **Small–medium.** The "InvestSmart" campaign (~27 HTML pages) and an investment-relevant slice of `/consumers/` (~25 pages of scam warnings and how-to-check guidance). |

Neither regulator publishes a large teaching corpus — they are supervisors, not
educators. FCA's sitemap lists **43,142 URLs**, but ~22k are news and ~16k are
regulatory publications for firms; only a few dozen pages are written for retail
investors.

MoneyHelper (`moneyhelper.org.uk`, UK) has a much larger consumer-investing
section but returns `403 Forbidden` to the collector, so it is not used. Older
docs still mention it as a source — it is not in `sources.json`.

## How the site inventories were taken

- **FCA:** `robots.txt` → `sitemap.xml` → `sitemap-main.xml?page=1..2` (43,142
  URLs). Filtered by path prefix (`/investsmart/`, `/consumers/`).
- **ESMA:** no `sitemap.xml` is published. The Investor Corner section was
  crawled from `https://www.esma.europa.eu/investor-corner` one hop deep.
- Each candidate URL was then fetched and run through `collect.clean_html` to
  confirm HTTP 200, an HTML content type, and enough body text to be worth
  indexing (≥ ~400 characters after chrome removal).

## Curated set (what `sources.json` collects)

**51 pages**, one `sources.json` entry each, fetched at `--pages-per-source 1`
(no link-following — the list is explicit).

| Group | `source_name` | Pages | Contents |
| --- | --- | --- | --- |
| ESMA Investor Corner | `ESMA Investor Corner` | 8 | Getting ready to invest, cost of investment products, checking a firm is regulated, frauds & scams, product intervention, making a complaint, AI-for-investing warning |
| FCA InvestSmart | `FCA InvestSmart` | 22 | Should you invest, the five questions / five checks, picking the right investment, golden rules, risk vs return, diversification, mainstream vs high-risk investments, price graphs, crowdfunding, unregulated collective investment schemes, crypto (basics, investing in it, using AI research tools), FOMO / hype / calm-rational-informed behaviour guides, pump-and-dump, the GameStop episode |
| FCA Consumer Warnings | `FCA Consumer Warnings` | 21 | Your rights, glossary of financial terms, how to check a firm/individual is authorised, protecting yourself from scams, clone firms, misleading promotions, dealing with EEA firms/funds, and scam-type pages: crypto, online-trading, forex, binary options, boiler room, recovery room, Ponzi/pyramid, land banking, carbon credits, mini-bonds, loan-fee fraud, pension scams, side pockets, greenwashing |

Rough size once collected: **~295 KB** of cleaned text → `data/raw/` ≈ **~400 KB**
(51 JSON files) → ~420 chunks → `data/index/` ≈ **~6 MB**. Comparable to the
current index, but on-topic (see below).

## Deliberately excluded

| Excluded | Why |
| --- | --- |
| FCA `/investsmart/` interactive shells — `hype-type-revealer`, `welcome-htr`, `resources`, `*-skills-into-action`, `about-campaign` | Quiz/widget pages; little or no standalone text after chrome removal |
| FCA `/consumers/warning-list-unauthorised-firms` | Returns `403` to the collector |
| FCA `/consumers/` non-investment pages — car finance, PPI, funeral plans, buy-now-pay-later, mortgages, interest-rate hedging, Welsh-language duplicates | Out of scope |
| ESMA `/esmas-activities/investors-and-issuers/*` (fund management, crowdfunding rules, credit-rating agencies, benchmark administrators) | Substantial text, but written for firms and markets — the regulatory framework, not investor guidance. Reintroduces the noise this curation removes. |
| Whole-domain crawls of either site | Tens of thousands of pages of news and firm-facing regulation |

## Available but not yet ingestible

- **ESMA investor publications** — `https://www.esma.europa.eu/investor-corner/publications-investors`
  links to ~50+ factsheets and warnings (finfluencers, crypto frauds, online
  scams in an AI world, joint ESAs crypto warning, …) as **PDFs**. The collector
  only handles HTML; a PDF loader would roughly double the EU corpus.
- **FCA warning list** of unauthorised firms — useful but `403` to the collector
  and better consumed as data than as prose.

## The current `data/index/` is not this set

The committed index was built before this curation, from the old two-seed list
crawled one hop deep. It contains noise — ESMA board-governance and careers
pages, FCA `/privacy`, `/accessibility`, `/news/rss.xml`. Of the 56 indexed
pages, roughly half are off-topic.

## Refreshing the knowledge base

```bash
uv run investment-rag collect          # fetch the 51 curated URLs -> data/raw/
uv run investment-rag build            # chunk + embed -> data/index/
git add data/raw data/index && git commit -m "refresh knowledge base"
```

`collect` reports and skips any URL that starts returning `403` rather than
trying to bypass it; replace such an entry with an accessible official page.
