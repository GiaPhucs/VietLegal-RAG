# VietLegal-RAG

**Evidence-grounded Vietnamese legal question answering with hybrid retrieval, neural reranking, structured legal evidence, grounded generation, and citation validation.**

VietLegal-RAG is an independent retrieval-augmented generation system for Vietnamese legal question answering. The runtime retrieves legal passages from a structured corpus, reranks chunk-level evidence, generates an answer, then independently validates legal citations and factual grounding before the answer is allowed through the final shipping gate.

## Architecture

```text
Question
  ↓
Query normalization
  ↓
Dense Parent + VietLegal-Harrier Chunk + BM25 Parent + BM25 Chunk
  ↓
Weighted Reciprocal Rank Fusion
  ↓
Parent-local expansion + direct chunk candidates
  ↓
V3 chunk-only neural reranker
  ↓
Top evidence chunks
  ↓
Structured legal evidence aggregation
  ↓
Qwen3.5-2B + Stage2-v2 grounded generation
  ↓
Citation + grounding validation
  ↓
Deterministic citation repair when uniquely supported
  ↓
Mandatory revalidation
  ↓
Final grounded answer + trusted [E#] evidence
```

The reranker operates on **chunks**, not whole legal Articles. Article/provision aggregation happens only after reranking.

## Main runtime components

- Dense Parent retrieval: `AITeamVN/Vietnamese_Embedding_v2`
- Direct Chunk retrieval: `mainguyen9/vietlegal-harrier-0.6b`
- Lexical retrieval: persistent SQLite FTS5 BM25 Parent + Chunk indexes
- Neural reranker: `AITeamVN/Vietnamese_Reranker` + VietLegal-RAG V3 LoRA adapter
- Generator: `Qwen/Qwen3.5-2B` + Stage2-v2 LoRA adapter
- Structured evidence aggregator
- Legal-aware citation and grounding validator
- Deterministic evidence-backed citation repair

## Verified end-to-end smoke test

A clean portable runtime test was executed with project artifacts staged on local SSD.

```text
Pipeline status: PASS
Final shipping gate: PASS
Citation mismatches: 0
Unsupported citations: 0
Unsupported substantive claims: 0
Pipeline runtime after model initialization: 68.38s
Full process wall time including model startup: 154.04s
```

The timing above is a **single-query runtime smoke measurement**, not a benchmark claim.

## Runtime performance note

Runtime indexes and databases should be stored on local SSD. In the verified smoke environment, moving runtime artifacts from mounted Google Drive to local SSD reduced the pipeline time substantially, especially hybrid retrieval and candidate hydration.

## Installation

Verified runtime dependencies are pinned in `requirements-runtime.txt`.

```bash
pip install -r requirements-runtime.txt
pip install -e . --no-deps
```

CUDA is required by the canonical neural runtime.

## Runtime artifacts

The current runtime artifact bundle contains 151 files totaling 4.592 GB.

Integrity metadata is stored in `artifact_manifest.json`.

For an existing local artifact source:

```bash
python scripts/download_artifacts.py \
  --source-root /path/to/Task2 \
  --artifact-root ./artifacts
```

Then verify the runtime layout:

```bash
python scripts/run_pipeline.py \
  --artifact-root ./artifacts \
  --check-only
```

Run a legal question:

```bash
python scripts/run_pipeline.py \
  --artifact-root ./artifacts \
  --question "Thời điểm nào sẽ thông báo phạm nhân hết hạn chấp hành án phạt tù?"
```

## Public artifact download status

**Project-specific runtime artifacts are hosted publicly on Hugging Face at `gphs09/VietLegal-RAG-artifacts` and are pinned by repository commit in `artifact_manifest.json`.**

Therefore this command:

```bash
python scripts/download_artifacts.py
```

downloads the project-specific runtime bundle using the commit-pinned URLs recorded in `artifact_manifest.json`.

Base neural models are fetched from Hugging Face; project-specific LoRA adapters, indexes, metadata, and SQLite stores are described by the artifact manifest.

## Validation policy

Generated legal citations are not trusted automatically. A generated answer can contain a correct substantive rule but an incorrect legal instrument identity. The validator therefore evaluates citation fidelity independently from factual grounding.

An answer is returned as final only when:

- explicit legal-document mismatches are zero;
- unsupported citations are zero;
- unsupported substantive claims are zero;
- the final validation gate is `PASS`.

When an explicit citation mismatch has exactly one trusted evidence candidate, the runtime may repair it deterministically and then re-run validation from scratch. Ambiguous or unsupported citations are not guessed.

## Evaluation

The independent evaluation framework is designed around retrieval, evidence, citation, grounding, and system metrics, including:

- Recall@K, MRR@K, nDCG@K
- Evidence Recall / Evidence Precision
- Citation Precision / Citation Recall / Citation Coverage
- Supported Claim Rate / Unsupported Claim Rate
- Evidence Coverage
- latency, throughput, RAM and VRAM

A dedicated independent benchmark is separate from the runtime smoke tests described above.

## Source provenance

See `SOURCE_PROVENANCE.md` for implementation provenance and reconstruction notes where available.

## Resumable artifact downloads

The public artifact downloader supports interrupted-download recovery.

Incomplete files are retained as `.part` files and resumed with HTTP Range requests when supported. Transient network failures are retried automatically, and files are moved into their final path only after the expected byte size is reached.

All completed artifacts are then verified against the SHA256 values in `artifact_manifest.json`.

