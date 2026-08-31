# NAmazon — Devpost Submission Draft

> Replace every bracketed field before submission.

- **Public GitHub repository:** https://github.com/Levi3110/NAmazon
- **Demo video or hosted demo:** [DEMO URL]
- **Team:** Levi3110 (Team Leader); [ADD OTHER MEMBERS, IF ANY]

## Inspiration and Problem

Product search often assumes that a customer already knows the right keywords.
Real customers do not always begin that way: some have hard constraints and are
ready to buy, while others want to explore styles or categories. A useful
shopping assistant must distinguish these modes, ask targeted questions, retain
context when preferences change, and ground every recommendation in catalog
evidence instead of inventing products.

NAmazon addresses this problem with a voice-first conversational shopping agent.
It supports separate Buying and Browsing strategies, hybrid retrieval over a
50,000-product catalog, semantic conversation memory, evidence-gated
clarification, optional LLM reranking, and a speaking AI avatar. It preserves the
competition's exact `Agent.reset(...)` and `Agent.respond(...)` interface while
also providing an end-to-end interactive demonstration.

## What It Does

At the beginning of the demo, the customer selects Buying or Browsing. NAmazon
then extracts product constraints and retrieves candidates through multiple
routes:

1. BM25 for exact keywords and attributes.
2. Multilingual Sentence Transformer embeddings with FAISS for semantic search.
3. Metadata/category filtering for structured constraints and budget.
4. Weighted reciprocal-rank fusion to create a candidate pool.

Before generation, a scope and evidence gate checks whether the request is
actually about shopping and whether the retrieved catalog contains enough
support. Unrelated requests receive a fixed shopping-only response. Vague
shopping requests receive one focused clarification question, with no LLM call
and no speculative recommendation. When evidence is sufficient, Qwen reranks
only retrieved IDs and writes a short grounded response. A circuit breaker falls
back to deterministic weighted RRF when Qwen is unavailable.

The demo accepts typed input or browser microphone audio. LiveKit transports the
audio over WebRTC and local Faster-Whisper converts English or Vietnamese speech
to text. Responses are sent to LiveTalking, which combines Edge TTS and Wav2Lip
to produce assistant audio and an audio-reactive avatar. Customers can select a
recommended product, choose a size, confirm it, and see a cached product preview.
Confirmed choices become anonymous preference signals for future sessions in
the same browser.

## How It Addresses the Challenge

- **Buying sessions:** prioritize hard constraints, metadata filters, and exact
  requirement satisfaction before semantic similarity.
- **Browsing sessions:** prioritize semantic relevance, category diversity, and
  topic suggestions.
- **Intent overrides:** use a single-writer reducer and session checkpoint so a
  new preference atomically replaces conflicting state.
- **Boundary behavior:** track asked attributes and explicit no-preference
  responses to avoid repetitive questioning.
- **Safe personalization:** combine the anonymized profile with semantically
  relevant session memories and confirmed demo behavior.
- **Grounding:** recommend only `parent_asin` values from the frozen catalog and
  call the LLM only after retrieval passes the evidence gate.

## Architecture

```text
Typed input or microphone
  -> Shopping scope / intent classifier
  -> Checkpointed constraint + semantic memory
  -> BM25 + FAISS + metadata/category routes
  -> Weighted RRF candidate pool
  -> Evidence gate
       weak: deterministic clarification, no LLM
       strong: optional Qwen reranking, offline RRF fallback
  -> Ranked catalog IDs + grounded response
  -> LiveTalking voice/video
```

The interactive stack is supervised by one command. It starts LiveKit, waits for
readiness, starts LiveTalking, then starts the combined frontend/backend and
opens the browser.

The repository includes a detailed, implementation-aligned workflow for the
Buying, Browsing, voice, memory, retrieval, ranking, and confirmation paths in
[`docs/WORKFLOW.md`](docs/WORKFLOW.md).

## Development Tools

- Visual Studio Code for Python, frontend, and configuration development.
- macOS Terminal for server operation and integration debugging.
- Python virtual environments and pip for dependency isolation.
- pytest for regression and interface tests.
- The deterministic local evaluator for repeatable ranking measurements.

## APIs and Services Used

- A private, self-hosted **Qwen `qwen3.6-27b`** model exposed through an
  OpenAI-compatible vLLM API. Its endpoint and optional key are stored only in a
  gitignored `.env` file.
- **LiveKit** local server and SDK for browser microphone WebRTC transport.
- **LiveTalking** HTTP/WebRTC endpoints (`/offer` and `/human`) for digital-human
  video and assistant speech.
- **Edge TTS** through LiveTalking for English speech synthesis.
- **DuckDuckGo image results** for optional confirmed-product previews. This is
  a demo-only feature with an offline placeholder and does not affect scoring.

The `openai` package is used as a protocol-compatible client for the self-hosted
Qwen service; no OpenAI-hosted model is required.

## Libraries and Frameworks

- `rank-bm25`
- Hugging Face Sentence Transformers
- FAISS CPU
- NumPy and PyTorch
- OpenAI Python client
- LiveKit Python/API SDK and vendored browser client
- Faster-Whisper
- LiveTalking, Wav2Lip, `aiohttp`, and `aiortc`
- `python-dotenv`, `truststore`, and pytest

## Datasets and Assets

NAmazon uses Amazon Reviews 2023 from McAuley Lab at UCSD, specifically the
`Clothing_Shoes_and_Jewelry` category. The challenge package contains a frozen
50,000-product text/metadata catalog keyed by `parent_asin` and 200 labeled
public development sessions. The organizer keeps 800 evaluation sessions
private. Raw user IDs, free-text reviews, purchase timestamps, and raw purchase
histories are not exposed to the agent.

Runtime assets include the multilingual Sentence Transformers model,
Faster-Whisper `base`, a Wav2Lip checkpoint, the LiveKit browser client, and a
preprocessed still avatar for the demo. Data attribution and usage notes are in
`DATA_ATTRIBUTION.md`. Large datasets and model weights are not committed to the
public repository.

LiveTalking is based on the upstream Apache-2.0 project at
https://github.com/lipku/LiveTalking, tested from commit
`c4f8c16a86bdc4d217782cac52fb431dc5bca7b0`. Third-party code and model assets
remain subject to their own licenses and terms.

## Results and Reproduction

On all 200 public sessions, with Qwen disabled to isolate retrieval quality,
NAmazon achieved:

- Hit Rate@10: **0.675**
- MRR: **0.412635**
- MTTC: **4.815 turns**
- Technical Score: **0.58499**

With Qwen reranking enabled on the same 200 sessions, NAmazon achieved:

- Hit Rate@10: **0.69**
- MRR: **0.481871**
- MTTC: **4.865 turns**
- Technical Score: **0.612261**
- Prompt tokens: **2,494,783**
- Completion tokens: **76,797**
- Total tokens: **2,571,580**

The Qwen-enabled evaluator ran sequentially in **5,542.40 seconds** (92.373
minutes), averaging **27.712 wall-clock seconds per session**. This is an
end-to-end full-suite average rather than a per-turn p50/p95 measurement. The
self-hosted endpoint had no per-token vendor API charge. The run used local
FAISS, no paid vector database, and existing development hardware, so the
observed incremental direct monetary cost was **$0**. Electricity and underlying
infrastructure opportunity cost were not separately estimated.

The Qwen-disabled run reports zero LLM tokens and demonstrates the required
network-independent fallback. Full Qwen measurements are stored in
[`docs/qwen_results.json`](docs/qwen_results.json).

Reproduce the evaluation after setup:

```bash
QWEN_RERANK_ENABLED=false .venv/bin/python -m evaluator.local_evaluator
```

Reproduce the Qwen-enabled measurement with:

```bash
QWEN_RERANK_ENABLED=true .venv/bin/python -m evaluator.local_evaluator --output results-qwen.json
```

Route-level recall can be reproduced independently with:

```bash
QWEN_RERANK_ENABLED=false .venv/bin/python scripts/retrieval_diagnostics.py
```

Run the automated tests:

```bash
TECHJAM_ENABLE_SBERT=false QWEN_RERANK_ENABLED=false .venv/bin/python -m pytest -q
```

Run the complete voice/avatar demo:

```bash
.venv/bin/python run_namazon.py
```

## Challenges

The main engineering challenges were combining sparse and dense signals without
losing hard constraints, preventing race conditions between transcript and
intent updates, maintaining the exact competition API while adding demo-only
features, avoiding LLM hallucination under weak evidence, and coordinating three
real-time services for microphone, avatar, and shopping inference.

## Limitations and What We Would Improve

NAmazon searches a frozen clothing/shoes/jewelry catalog rather than live
inventory. It does not place orders, validate stock, or verify that a chosen size
is currently available. Optional Qwen, image search, Edge TTS, and the avatar can
depend on network or hardware availability, although core retrieval has an
offline fallback. Image results may not show the exact catalog variant. The
scope/evidence classifier remains heuristic, and local JSON behavior memory is
not appropriate for production accounts without encryption, consent, retention,
and deletion controls. LiveKit development credentials and localhost HTTP are
not production security settings.

With more time, we would add live retailer inventory and checkout APIs,
calibrated multilingual intent/evidence models, learned rank-fusion weights,
streaming ASR and interruption handling, verified first-party product imagery,
privacy-preserving account memory, GPU performance profiling, accessibility
testing, and a TLS-secured deployment with monitoring and rate limits.

## Team Contributions

- **Levi3110 — Team Leader:** system architecture and technical direction;
  hybrid retrieval/RAG, backend integration, voice/avatar coordination, UI,
  evaluation, release preparation, and documentation.

- **[ADD OTHER TEAM MEMBERS, IF ANY]** — [specific responsibilities].

Remove the second line if this is a solo submission.

## Public Repository

Source, setup instructions, evaluator reproduction, tests, and architecture:
**https://github.com/Levi3110/NAmazon**

First-party NAmazon source is released under Apache License 2.0. Amazon Reviews
2023 data, Wav2Lip/model checkpoints, LiveTalking, LiveKit, the avatar, and all
other third-party assets retain their own licenses and terms.
