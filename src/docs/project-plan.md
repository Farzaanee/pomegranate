# Grounded Investment Research Agent (EU/UK)

> **Educational only — not regulated financial advice.** This project retrieves information from official sources; it does not recommend specific investments or replace professional advice.

## Phase 1: local knowledge base

The included Python pipeline collects trusted public pages, removes site chrome, chunks readable content, and stores embeddings plus provenance in local Chroma storage. Every retrieved passage retains its source name, region, original URL, title, and chunk index.

### Setup

Requires Python 3.10 or later; the repo pins 3.12 via `.python-version`, which is the
version tested against and the one with the widest prebuilt-wheel coverage on
Streamlit Community Cloud. Avoid pre-release interpreters — `chromadb` and
`onnxruntime` ship no wheels for them.

Using [uv](https://docs.astral.sh/uv/) (recommended; a `uv.lock` is committed):

```bash
uv sync --extra dev   # downloads Python 3.13 if needed, builds .venv/, installs the project
uv run investment-rag collect --pages-per-source 1
uv run investment-rag build
uv run investment-rag query "Why does diversification matter when investing?" --region UK
uv run pytest
```

Or with the standard library `venv`:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
investment-rag collect --pages-per-source 1
investment-rag build
investment-rag query "Why does diversification matter when investing?" --region UK
pytest
```

The `all-MiniLM-L6-v2` ONNX weights (~80 MB, no PyTorch) download on first use and are cached. `data/raw/` and the built index `data/index/` are committed so the deployed app opens a prebuilt index; `data/chroma/` is gitignored local scratch. Increase `--pages-per-source` only after checking each publisher's terms, robots policy, and the collected JSON documents in `data/raw/`. A publisher may return `403 Forbidden`; the collector will report and skip that source rather than attempting to bypass access controls. Replace its entry in `sources.json` with an accessible official page, then collect again.

### Pipeline

1. `sources.json` is the allow-list of ESMA Investor Corner (EU), MoneyHelper (UK), and FCA InvestSmart (UK) seed pages.
2. `investment-rag collect` downloads seed pages and optionally a limited number of same-domain links, then saves cleaned documents as reviewable JSON.
3. `investment-rag build` creates overlap-preserving chunks and upserts them into persistent Chroma storage.
4. `investment-rag query` performs semantic retrieval and prints the supporting text with source and URL. Use `--region EU` or `--region UK` to prevent cross-region retrieval.

The test suite validates cleaning, chunking, and required provenance metadata offline. A successful sample query that returns a relevant passage carrying its official source, region, and URL meets the Phase 1 retrieval criterion.

---

# Project Brief: Grounded Investment Research Agent (EU/UK)

## 1. Project Summary
This project is a portfolio piece designed to demonstrate applied AI engineering skills, specifically retrieval-augmented generation (RAG) and agentic AI, through a realistic, socially useful fintech use case: an agent that helps everyday people, not just high-net-worth individuals, figure out what kinds of investments might suit them.

Rather than generating advice from general knowledge alone, the agent grounds its reasoning in trusted, official sources: ESMA's (European Securities and Markets Authority) Investor Corner for the EU, and FCA (Financial Conduct Authority) InvestSmart / MoneyHelper for the UK. It takes in a simple user profile (income, goals, timeline, risk tolerance, and region) and produces a recommendation with citations back to the source material, plus a plain-language explanation of the reasoning.

The project serves two goals simultaneously: building a credible, demoable AI engineering portfolio piece for fintech job applications, and building the creator's own working knowledge of personal finance and investing principles along the way.

## 2. Target User
- Primary: the project creator, as both a learning tool and a portfolio demonstration
- Secondary (as a design target): everyday individuals, including those with modest income such as students or early-career professionals, who want grounded, jargon-free investment guidance without needing a wealth manager
- Audience for the portfolio itself: hiring managers and recruiters evaluating AI engineering skill for fintech roles

## 3. Goals and Success Metrics
- Technical: a working RAG pipeline plus an agentic reasoning layer that produces grounded, citation-backed recommendations
- Differentiator: correct handling of multi-jurisdiction logic (EU vs. UK) based on user region
- Learning: creator can explain, unprompted, the core investing principles used in the project, such as diversification, risk tolerance, time horizon, and fees
- Portfolio quality: a clean public repository with documentation, a short demo (video or live), and a written explanation of design decisions
- Trustworthiness: low hallucination rate, meaning recommendations should be traceable to real source passages, verified through the evaluation step in Phase 4

## 4. Risks and Constraints
- Finance is not an exact science. There is no single ground truth, so the system must be framed as educational and principle-based, not as regulated financial advice
- Regional regulatory differences between the EU and UK add complexity and must be handled carefully to avoid misleading users
- Time constraint: the creator is building this part-time, roughly 10 hours per week, alongside other commitments
- Legal and ethical note: the project should include a clear disclaimer stating this is not licensed financial advice

## 5. Out of Scope (for now)
- Real-time market data or live trading integration
- Country-specific tax optimization advice
- Support for jurisdictions beyond the EU and UK
- Regulatory compliance or certification as an actual financial advisory tool

---

## Time Assumption
Based on 10 hours per week, this project runs across roughly 5 weeks, for about 50 hours total.

---

## Phase 1: Foundations (Week 1) — approximately 10 hours
**Objective:** Establish the knowledge base and a basic RAG pipeline.

**Key activities**
- Learn investing basics: risk, diversification, asset classes, time horizon
- Collect and clean source documents from ESMA Investor Corner, MoneyHelper, and FCA InvestSmart
- Chunk documents and build a basic retrieval pipeline

**Recommended stack**
- Python for the core pipeline
- An embedding model such as OpenAI embeddings or an open-source alternative like sentence-transformers
- A vector store such as Chroma or FAISS for local development

**Resources and skills needed**
- Basic Python and familiarity with APIs
- Understanding of how embeddings and vector similarity search work
- Time to read foundational investing content from the chosen sources

**Acceptance criteria to move to Phase 2**
- Given a sample finance question, the retriever returns relevant, correctly sourced passages
- Source documents are cleanly chunked and stored with metadata (source name, region, URL)

---

## Phase 2: Reasoning Layer (Week 2) — approximately 10 hours
**Objective:** Add the agentic layer that reasons over retrieved content.

**Key activities**
- Learn about investment vehicles such as index funds, bonds, and ISAs versus general accounts, plus suitability concepts like risk tolerance and timeline
- Design a user profile schema (income, goals, timeline, risk tolerance, region)
- Build a reasoning agent that combines the user profile with retrieved evidence to produce a recommendation with citations

**Recommended stack**
- An agent framework such as LangChain, LlamaIndex, or a lightweight custom orchestration using function calling
- A capable language model via API, such as Claude or GPT, for the reasoning step
- Simple structured prompts or a state machine to keep reasoning steps traceable

**Resources and skills needed**
- Understanding of prompt design and structured output
- Basic grasp of investment suitability concepts, ideally reinforced by the recommended YouTube channels like Plain Bagel or PensionCraft
- Familiarity with function calling or tool use patterns in LLM frameworks

**Acceptance criteria to move to Phase 3**
- Given a sample user profile, the agent produces a coherent recommendation that cites specific retrieved passages
- Recommendations are explainable in plain language, not just a raw citation dump

---

## Phase 3: Multi-Jurisdiction Logic (Week 3) — approximately 10 hours
**Objective:** Handle EU versus UK differences correctly.

**Key activities**
- Learn key regulatory and product differences between the EU and UK, such as ISAs being UK-specific, and MiFID II versus FCA rules
- Add a routing or classification step so the agent selects the correct source set based on user region
- Test comparison scenarios where the same user profile is evaluated under both regions

**Recommended stack**
- Simple rule-based routing logic, or a lightweight classifier if you want to demonstrate additional ML skill
- Extended metadata tagging in the vector store to separate EU and UK sources cleanly

**Resources and skills needed**
- Comparative understanding of EU and UK investment regulation basics, at a plain-language level, not legal expertise
- Testing discipline to verify the agent doesn't mix up sources between regions

**Acceptance criteria to move to Phase 4**
- The agent correctly selects and cites only region-appropriate sources for a given user
- The agent can explain a key difference between EU and UK options when asked directly

---

## Phase 4: Evaluation and Polish (Weeks 4 to 5) — approximately 20 hours
**Objective:** Make the project portfolio-ready and trustworthy.

**Key activities**
- Deepen understanding of what good advice looks like, including fee awareness, avoiding hype, and checking debt before investing
- Build evaluation criteria covering groundedness, hallucination checks, and citation accuracy
- Add a simple user interface or chat interface
- Write full documentation, including a README and short design write-up

**Recommended stack**
- A lightweight front end such as Streamlit or a simple web chat interface
- A small evaluation script or notebook comparing agent outputs against source documents
- GitHub for version control and hosting the final repository

**Resources and skills needed**
- Basic evaluation or QA mindset for LLM outputs
- Enough front-end skill to build a simple, clean demo interface
- Technical writing skill for the documentation

**Acceptance criteria for project completion**
- The agent passes a small evaluation set with an acceptable groundedness rate, for example correctly citing sources in the large majority of test cases
- A demoable interface exists and functions end-to-end
- Documentation clearly explains the architecture, design decisions, and disclaimers about the tool not being licensed financial advice

---

## Stretch Goals (optional, if time allows)
- Add a third jurisdiction for comparison, such as Germany or another specific EU member state
- Add an "explain your reasoning" trace view showing which sources were used
- Add basic personalization memory across sessions

---

## Suggested Weekly Rhythm
At 10 hours per week, pair one finance topic with one matching technical milestone each week, learned and built in parallel, so concepts map directly onto what is being implemented. Total: approximately 50 hours across 5 weeks.
