# VietLegal-RAG v2.1 — Evaluation

<!-- V21_FINALHOLDOUT200_START -->

## Primary v2.1.0 benchmark — finalholdout200

The primary v2.1.0 benchmark is now `finalholdout200`.

Benchmark identity:

- Queries: **200**
- Train5600 overlap: **0**
- Historical Dev600 overlap: **0**
- Silver Oracle coverage: **200/200**
- Oracle used during inference: **No**
- Runtime errors: **0**
- Benchmark SHA256:
  `66167e8b5845be10115fe46dbbb8178e0e461ad1c6c8a4e9fba5b221f2c069be`
- STEP 9B protocol: `afe7b10870b5855d`
- STEP 9C protocol: `8edf37148b32a119`

### Headline metrics

| Category | Metric | Result |
|---|---|---:|
| Retrieval | Hybrid Parent AnyHit@5 | **91.00%** |
| Retrieval | Hybrid Parent MRR@20 | **0.7469** |
| Reranking | V3 Exact AnyHit@5 | **80.00%** |
| Reranking | V3 Parent AnyHit@5 | **97.00%** |
| Reranking | V3 Parent MRR@20 | **0.8404** |
| Candidate Pool | Any Exact Gold | **98.00%** |
| Candidate Pool | Any Parent Gold | **99.50%** |
| Evidence | Silver Parent Precision | **51.59%** |
| Evidence | Silver Parent Recall | **41.78%** |
| Trusted Citation | Parent Precision | **72.73%** |
| Trusted Citation | Parent Recall | **12.81%** |
| Trusted Citation | Coverage | **41.03%** |
| Grounding Proxy | Supported Claim Rate | **88.02%** |
| Grounding Proxy | Unsupported Claim Rate | **3.49%** |
| Shipping Gate | PASS | **42.50%** |

Full report:

`docs/benchmarks/v2.1.0/BENCHMARK_v2.1.0_finalholdout200.md`

Historical Dev600 and competition-era metrics below are retained only as
historical context and are **not** the primary v2.1.0 benchmark.

<!-- V21_FINALHOLDOUT200_END -->

## Portfolio benchmark

| Category | Metric | Result |
|---|---|---:|
| Retrieval | Exact Recall@5 | 0.2430 |
| Retrieval | Parent Recall@5 | 0.4142 |
| Retrieval | Any Exact@5 | 81.83% |
| Retrieval | Any Parent@5 | 93.33% |
| Retrieval | MRR@20 | 0.7006 |
| Retrieval | Exact Recall@20 | 0.3843 |
| Retrieval | Parent Recall@20 | 0.5844 |
| Retrieval | Exact Recall@50 | 0.4977 |
| Retrieval | Parent Recall@50 | 0.7095 |
| Citation | Silver Exact Citation Precision | 43.63% |
| Citation | Silver Exact Citation Recall | 18.60% |
| Citation | Silver Parent Citation Precision | 63.70% |
| Citation | Silver Parent Citation Recall | 30.55% |
| Citation | Any Exact Citation Hit / Query | 72.33% |
| Citation | Any Parent Citation Hit / Query | 85.17% |
| Grounding Proxy | Supported Sentence Rate | 89.28% |
| Grounding Proxy | Partial Sentence Rate | 7.24% |
| Grounding Proxy | Unsupported Sentence Rate | 3.48% |
| Grounding Proxy | Citation Coverage | 96.52% |
| Grounding Proxy | Weighted Answer-Evidence Token Coverage | 0.9422 |
| Evidence | Evidence Units / Query | 4.545 |
| Evidence | Queries With Aggregation | 34.33% |
| Evidence | Queries With Article-Level Evidence | 98.17% |

## Historical competition-era generation metrics

| Metric | Dev600 |
|---|---:|
| METEOR | 0.5335 |
| ROUGE-L | 0.5591 |

These generation metrics are retained as historical competition-era results and are not recomputed as the primary V2.1 portfolio metrics.

## Key findings

- Any Exact@5: **81.83%**.
- Any Parent@5: **93.33%**.
- MRR@20: **0.7006**.
- Silver Parent Citation Precision: **63.70%**.
- Silver Exact Citation Precision: **43.63%**.
- At least one Silver Oracle document is cited in **85.17%** of Dev600 queries.
- Unsupported Sentence Rate proxy: **3.48%**.

## Interpretation

The retrieval stack is substantially stronger at locating the relevant legal document than at selecting the exact Silver Oracle passage. This is visible in the gap between parent-level and exact-chunk citation metrics.

The citation layer therefore exposes a remaining passage-level grounding bottleneck rather than hiding it behind generation-only metrics.

## Methodology notes

- The dataset does not provide official gold citation labels.
- Citation evaluation uses answer-aware Silver Oracle evidence.
- Silver Oracle evidence is never used during inference.
- Parent citation metrics measure document-level alignment.
- Exact citation metrics require overlap with Silver Oracle chunks.
- Grounding metrics are deterministic legal-aware lexical proxies.
- Unsupported Sentence Rate is not a definitive hallucination rate.
- V2.1 inference remains question-only.

## Architecture evaluated

```text
Question
   ↓
Dense Parent Retrieval
+ VietLegal-Harrier Direct Chunk Retrieval
+ BM25 Parent
+ BM25 Chunk
   ↓
4-way Weighted RRF
   ↓
Parent-local Expansion + Direct Chunk Union
   ↓
V3 Oracle-Supervised Neural Reranker
   ↓
Top5 Evidence
   ↓
Evidence Aggregator
   ↓
Qwen3.5-2B Stage1 / Stage2 Hybrid
   ↓
Legal-Aware Claim-Evidence Alignment
   ↓
Explicit Citations + Grounding Status
```
