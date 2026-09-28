# Development and Source Provenance

## Project origin

VietLegal-RAG originated from the LegalQA work developed for **Task 2 of the UIT Data Science Challenge 2026**.

The original research and competition workflow was developed interactively in **Google Colab and Google Drive**.

Training runs, retrieval experiments, evaluation outputs, model adapters, indexes, and runtime artifacts were produced across multiple notebook and Colab sessions rather than maintained from the beginning as a conventional Git-based software project.

## Historical source preservation

A source audit performed during the v2.1 portfolio-packaging phase found that a single canonical historical source tree containing the complete competition-era retrieval → reranking → generation pipeline had not been preserved.

The retained materials primarily included:

- trained model adapters and checkpoint metadata;
- retrieval indexes and metadata;
- SQLite stores;
- benchmark and experiment outputs;
- runtime configuration information;
- verified architecture constraints.

For this reason, the current repository does **not** claim to be a byte-for-byte snapshot of every historical Colab notebook or every intermediate competition experiment.

## Public v2.1.0 package

After the competition, the verified components and runtime behavior were consolidated into the public `vietlegal_rag` Python package.

The packaging phase added:

- explicit runtime interfaces;
- stable artifact paths;
- dependency pinning;
- command-line execution;
- resumable artifact download;
- exact-size and SHA256 verification;
- benchmark provenance;
- public GitHub and Hugging Face release artifacts.

The public package therefore represents the **cleaned, packaged, and reproducible v2.1 system** rather than a raw dump of the original Colab environment.

## Why the Git history is short

Most experimentation and competition development happened before this GitHub repository became the canonical public home of the project.

GitHub was introduced primarily during the packaging, documentation, reproducibility, and release phase.

Therefore the small number of Git commits should be interpreted as the history of the **public release-packaging process**, not as the complete development timeline of the research project.

For a milestone-oriented development history, see:

[`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md)

## Canonical v2.1 architecture

```text
Question
  ↓
Dense Parent + VietLegal-Harrier Chunk + BM25 Parent + BM25 Chunk
  ↓
Weighted Reciprocal Rank Fusion
  ↓
Parent-local expansion + direct chunk candidates
  ↓
V3 chunk-level neural reranker
  ↓
Top chunks
  ↓
Map to Article / legal provision
  ↓
Evidence Aggregator
  ↓
Qwen3.5-2B + Stage2-v2 grounded generation
  ↓
Citation + grounding validation
  ↓
Deterministic citation repair when uniquely supported
  ↓
Mandatory revalidation
  ↓
Final shipping gate
```

A key frozen architecture constraint is:

```text
chunks
→ V3 reranker
→ top chunks
→ legal provision mapping
→ Evidence Aggregator
```

Whole legal Articles are not reconstructed and fed back into V3.

## Evaluation provenance

The primary public v2.1 benchmark is the independent `finalholdout200` evaluation.

Key leakage controls:

- 200 unique holdout queries;
- Train5600 overlap: 0;
- historical Dev600 overlap: 0;
- Silver Oracle coverage: 200/200;
- Silver Oracle is not used during inference.

Evidence and citation metrics use answer-aware **Silver Oracle** evidence for evaluation only. These labels are not official human-annotated citation gold.

Grounding metrics are frozen deterministic legal-aware validator proxies and must not be interpreted as definitive hallucination rates.

Historical competition-era metrics are retained only as historical context where explicitly identified.

## Historical notebooks

Historical notebooks should only be published when a real notebook can be verified, cleaned of credentials/private data, and clearly labeled as a historical experiment.

This repository does not create synthetic notebooks or artificial commit history merely to make the development timeline appear longer.

If historical notebooks are later added, they should be placed under `notebooks/` or `experiments/` and labeled as non-canonical research artifacts.

The canonical public runtime remains the packaged implementation under `src/vietlegal_rag/`.
