# VietLegal-RAG

**Evidence-grounded Vietnamese legal question answering with hybrid retrieval, neural reranking, structured legal evidence, grounded generation, and citation validation.**

VietLegal-RAG is an independent retrieval-augmented generation system for Vietnamese legal question answering. The runtime retrieves legal passages from a structured corpus, reranks chunk-level evidence, generates an answer, then independently validates legal citations and factual grounding before the answer is allowed through the final shipping gate.

<!-- PROJECT_CONTEXT_START -->

## Project Context

VietLegal-RAG originated from **Task 2 — LegalQA of the UIT Data Science Challenge 2026**.

The competition workflow was developed primarily in **Google Colab and Google Drive** during the competition period. After the competition, the verified system was further consolidated, extended, independently benchmarked, and packaged as the reproducible `VietLegal-RAG v2.1.0` portfolio release presented in this repository.

The original competition submission was developed individually.

**Development history note:** the project was developed in Google Colab/Drive during the competition; this GitHub repository was created later as the packaged, documented, and reproducible public release. The short Git history therefore reflects the release-packaging phase, not the full experimental timeline.

The current `finalholdout200` benchmark is a post-competition independent evaluation and should not be confused with competition leaderboard metrics.

<!-- PROJECT_CONTEXT_END -->

<!-- PERSONAL_CONTRIBUTIONS_START -->

## My Contributions

The original competition work and the subsequent public portfolio release were developed individually. My main engineering contributions include:

- Designed and integrated the hybrid retrieval pipeline combining Dense Parent, VietLegal-Harrier Chunk, BM25 Parent, and BM25 Chunk retrieval.
- Implemented and evaluated weighted Reciprocal Rank Fusion, parent-local expansion, and candidate-pool construction.
- Developed and evaluated the V3 chunk-level neural reranking pipeline.
- Integrated the Qwen3.5-2B + Stage2-v2 LoRA grounded-generation runtime.
- Designed the structured legal Evidence Aggregator and the citation/grounding validation pipeline, including deterministic citation repair.
- Built leakage-checked evaluation protocols for retrieval, evidence, citation, grounding, and runtime performance.
- Profiled and optimized runtime I/O, including moving indexes and databases from mounted Google Drive to local SSD.
- Packaged the public v2.1.0 release with resumable artifact downloads, SHA256 verification, frozen artifact revisions, GitHub release documentation, and Hugging Face artifact hosting.

<!-- PERSONAL_CONTRIBUTIONS_END -->

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

<!-- V21_OFFICIAL_BENCHMARK_START -->

## Official v2.1.0 benchmark

The current primary evaluation is a frozen **200-query final holdout**
(`finalholdout200`) with zero query-ID overlap with both the reranker
training set and the historical Dev600 set.

| Metric | Result |
|---|---:|
| Hybrid Parent AnyHit@5 | **91.00%** |
| V3 Exact AnyHit@5 | **80.00%** |
| V3 Parent AnyHit@5 | **97.00%** |
| V3 Parent MRR@20 | **0.8404** |
| Candidate Pool Any Exact Gold | **98.00%** |
| Candidate Pool Any Parent Gold | **99.50%** |
| Trusted Parent Citation Precision | **72.73%** |
| Supported Claim Rate | **88.02%** |
| Unsupported Claim Rate | **3.49%** |
| Final Shipping Gate PASS | **42.50%** |

The deterministic citation repair layer reduced explicit citation
mismatches from **93 to 6 (93.55% reduction)**.

On a Tesla T4:

- Retrieval + V3 latency: **14.15 s p50 / 16.21 s p95**
- Generation latency: **59.54 s p50 / 61.81 s p95**
- Generation throughput: **8.36 tokens/s**

Full methodology and results:

[`docs/benchmarks/v2.1.0/BENCHMARK_v2.1.0_finalholdout200.md`](docs/benchmarks/v2.1.0/BENCHMARK_v2.1.0_finalholdout200.md)

> Evidence and citation evaluation uses answer-aware **Silver Oracle**
> labels rather than official human-annotated citation gold.
> Grounding values are deterministic legal-aware validator proxy metrics
> and must not be interpreted as definitive hallucination rates.
> Shipping-gate PASS measures conservative release-gate compliance,
> not answer accuracy.

<!-- V21_OFFICIAL_BENCHMARK_END -->

## Evaluation

The independent evaluation framework is designed around retrieval, evidence, citation, grounding, and system metrics, including:

- Recall@K, MRR@K, nDCG@K
- Evidence Recall / Evidence Precision
- Citation Precision / Citation Recall / Citation Coverage
- Supported Claim Rate / Unsupported Claim Rate
- Evidence Coverage
- latency, throughput, RAM and VRAM

A dedicated independent benchmark is separate from the runtime smoke tests described above.

<!-- DEVELOPMENT_HISTORY_START -->

## Development History

A milestone-based engineering history is available in [`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md).

It summarizes the progression from the competition-era LegalQA workflow through hybrid retrieval, V3 reranking, structured evidence aggregation, grounded generation, citation validation, deterministic repair, independent benchmarking, runtime profiling, and the reproducible public release.

The development log is an engineering milestone record derived from retained experiment artifacts and verified benchmark outputs. It is not presented as a substitute for a historical Git commit timeline.

<!-- DEVELOPMENT_HISTORY_END -->

## Source provenance

See [`SOURCE_PROVENANCE.md`](SOURCE_PROVENANCE.md) for the development history, Colab/Drive-to-package migration process, source provenance, and the distinction between historical competition experiments and the packaged public v2.1.0 runtime.

## Resumable artifact downloads

The public artifact downloader supports interrupted-download recovery.

Incomplete files are retained as `.part` files and resumed with HTTP Range requests when supported. Transient network failures are retried automatically, and files are moved into their final path only after the expected byte size is reached.

All completed artifacts are then verified against the SHA256 values in `artifact_manifest.json`.

