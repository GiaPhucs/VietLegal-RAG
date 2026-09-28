# VietLegal-RAG Development Log

This document summarizes the major research and engineering milestones that led to the public `VietLegal-RAG v2.1.0` release.

It is a **milestone-oriented engineering record**, not a reconstructed Git commit history.

The original competition workflow was developed primarily in Google Colab and Google Drive. The public repository was created later during the packaging and reproducibility phase.

Historical competition-era metrics and the current independent benchmark use different protocols and should not be compared directly unless explicitly stated.

---

## Phase 1 — Competition-era LegalQA workflow

### Goal

Develop a retrieval-augmented system for Vietnamese legal question answering.

### Development environment

- Google Colab
- Google Drive
- GPU-backed notebook experiments
- persisted model adapters, indexes, SQLite stores, and evaluation outputs

### Research direction

The system evolved as a multi-stage pipeline rather than relying on a single retriever or a single end-to-end generator.

Key research areas included:

- lexical retrieval;
- dense semantic retrieval;
- legal passage retrieval;
- rank fusion;
- neural reranking;
- grounded answer generation.

---

## Phase 2 — Hybrid retrieval

### Problem

A single retriever did not consistently handle both semantic queries and exact legal terminology.

### Change

The retrieval layer evolved into four complementary channels:

- Dense Parent retrieval;
- VietLegal-Harrier direct Chunk retrieval;
- BM25 Parent retrieval;
- BM25 Chunk retrieval.

The channels are combined with weighted Reciprocal Rank Fusion.

### Design rationale

- Dense retrieval improves semantic matching.
- BM25 preserves exact lexical/legal identifiers.
- Harrier adds a fine-grained legal-passage neural signal.
- Parent retrieval helps maintain document-level recall.

---

## Phase 3 — Candidate Pool and parent-local expansion

### Problem

Relevant legal information can cross chunk boundaries, while neural reranking over the entire corpus is computationally impractical.

### Change

Candidate Pool combines:

- top fused parents;
- parent-local neighboring chunks;
- direct Harrier chunks;
- direct BM25 chunks.

### Current independent benchmark

| Metric | Result |
|---|---:|
| Exact gold coverage | 82.00% |
| Parent gold coverage | 92.76% |
| Any exact gold | 98.00% |
| Any parent gold | 99.50% |

---

## Phase 4 — V3 chunk-level neural reranker

### Problem

Candidate recall was strong, but relevant passages were not always ranked early enough.

### Change

V3 uses `AITeamVN/Vietnamese_Reranker` with the project-specific V3 LoRA adapter.

### Frozen architecture

```text
chunk candidates
→ V3 reranker
→ top chunks
→ Article / provision mapping
→ Evidence Aggregator
```

### Independent benchmark impact

| Metric | Before V3 | V3 |
|---|---:|---:|
| Exact AnyHit@5 | Harrier 63.00% | 80.00% |
| Exact MRR@20 | 0.5100 | 0.6282 |
| Parent AnyHit@5 | Hybrid 91.00% | 97.00% |
| Parent MRR@20 | 0.7469 | 0.8404 |

---

## Phase 5 — Structured legal Evidence Aggregator

### Problem

Fine-grained chunks are effective for retrieval but are not always the best context format for legal answer generation.

### Change

Top-ranked chunks are mapped back to legal provisions and grouped using document identity plus legal structure.

### Design principle

```text
fine-grained relevance first
→ structured legal reconstruction later
```

---

## Phase 6 — Grounded generator

Generator:

`Qwen/Qwen3.5-2B`

Adaptation:

Stage2-v2 final LoRA adapter.

Canonical runtime:

- 4-bit NF4;
- FP16 compute;
- 3072-token input cap;
- greedy decoding;
- thinking disabled;
- public default: 512 new tokens.

---

## Phase 7 — Citation validation

### Observed failure

The generator could produce a substantively correct rule while citing the wrong legal instrument.

Verified smoke example:

```text
Correct evidence:
Nghị định 49/2020/NĐ-CP
Điều 8

Generated citation:
Thông tư 49/2020/TT-BCA
```

### Change

An independent legal-aware Citation Validator was added so generated citations are checked against trusted evidence rather than accepted automatically.

---

## Phase 8 — Grounding validation

Answer claims are evaluated against evidence using a frozen deterministic legal-aware proxy.

| Metric | Result |
|---|---:|
| Supported Claim Rate | 88.02% |
| Partial Claim Rate | 8.49% |
| Unsupported Claim Rate | 3.49% |
| Evidence Coverage | 40.60% |

These values are validator proxy metrics and are not human-annotated hallucination rates.

---

## Phase 9 — Deterministic citation repair

### Problem

Calling the generator again to repair a citation could introduce another hallucination.

### Change

Citation mismatches are repaired only when exactly one trusted replacement candidate exists.

| Repair metric | Result |
|---|---:|
| Initial citation mismatches | 93 |
| Final citation mismatches | 6 |
| Mismatch reduction | 93.55% |
| Initial unsupported citations | 66 |
| Final unsupported citations | 66 |

Unsupported citations remain unchanged by design because the repair layer refuses to invent a source.

---

## Phase 10 — Conservative shipping gate

| Status | Count |
|---|---:|
| PASS | 85 |
| BLOCKED_CITATION_REPAIR | 60 |
| BLOCKED_REQUIRES_CITATION_REPAIR | 3 |
| BLOCKED_REQUIRES_GROUNDING_REPAIR | 4 |
| BLOCKED_REVIEW | 48 |

**Final Shipping Gate PASS: 85 / 200 = 42.50%**

Shipping PASS is a conservative release-gate metric, not answer accuracy.

---

## Phase 11 — Independent finalholdout200 benchmark

Benchmark identity:

- Queries: 200
- Train5600 overlap: 0
- historical Dev600 overlap: 0
- Silver Oracle coverage: 200/200
- Oracle used during inference: No
- Runtime errors: 0

Benchmark SHA256:

`66167e8b5845be10115fe46dbbb8178e0e461ad1c6c8a4e9fba5b221f2c069be`

Retrieval protocol:

`afe7b10870b5855d`

Generation / grounding protocol:

`8edf37148b32a119`

### Headline results

| Metric | Result |
|---|---:|
| Hybrid Parent AnyHit@5 | 91.00% |
| V3 Exact AnyHit@5 | 80.00% |
| V3 Parent AnyHit@5 | 97.00% |
| V3 Parent MRR@20 | 0.8404 |
| Candidate Any Exact Gold | 98.00% |
| Candidate Any Parent Gold | 99.50% |
| Trusted Parent Citation Precision | 72.73% |
| Trusted Parent Citation Recall | 12.81% |
| Citation Coverage | 41.03% |
| Supported Claim Rate | 88.02% |
| Unsupported Claim Rate | 3.49% |
| Shipping PASS | 42.50% |

### Main finding

Document discovery is no longer the dominant bottleneck. The major remaining constraints are citation recall, citation coverage, claim-to-evidence alignment, unsupported citation handling, and generation latency.

---

## Phase 12 — Runtime profiling and local-SSD optimization

Initial Drive-based runtime:

```text
Full pipeline:        522.90 s
Hybrid retrieval:     399.44 s
Candidate hydration:   71.82 s
```

After staging runtime artifacts to local SSD:

```text
Full pipeline:         68.38 s
Hybrid retrieval:      14.92 s
Candidate hydration:    0.34 s
```

**Overall speedup: 7.65×**

The primary bottleneck was storage I/O rather than neural inference.

---

## Phase 13 — Reproducible public release

### Source

Tag: `v2.1.0`

Frozen source commit:

`a0c3d8e3498bf3c7a42e03431eba23a9d4cf562f`

### Runtime artifacts

- 151 runtime artifacts
- exact total bytes: 4,930,429,361
- per-file size validation
- per-file SHA256 validation
- commit-pinned public URLs

Frozen Hugging Face artifact revision:

`17876627834fa9982b5c57cad3fb54394591ee32`

### Public release features

- Python package structure;
- CLI runner;
- resumable downloads;
- HTTP Range recovery;
- retry/backoff;
- exact-size validation;
- SHA256 verification;
- GitHub Release;
- Hugging Face artifact repository;
- independent benchmark documentation;
- anonymous public reproducibility audit.

---

## Current v2.1 conclusion

VietLegal-RAG v2.1 established a reproducible evidence-grounded legal QA pipeline with strong retrieval and reranking performance.

The benchmark exposed the next engineering priority clearly:

```text
retrieval is no longer the primary bottleneck
→ citation coverage and claim/evidence alignment are
```

Future engineering changes belong to v2.2 rather than rewriting the frozen v2.1.0 release.

## Historical notebooks

Historical notebooks should only be published when they are real, verifiable, and cleaned of credentials or private information.

No synthetic notebooks or artificial backdated Git history should be created merely to make the development timeline appear longer.
