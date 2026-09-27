# Architecture

## VietLegal-RAG v2.1

```text
QUESTION
  ↓
Query Normalization
  ↓
Hybrid Retrieval
  ├─ Dense Parent Top200
  ├─ VietLegal-Harrier Direct Chunk Top500
  ├─ BM25 Parent Top200
  └─ BM25 Chunk Top200
  ↓
4-way Weighted RRF
  ↓
Top Parents + Parent-local Expansion
+ Direct Chunk Candidates
  ↓
Deduplicated H3 Candidate Pool (~680 chunks/query)
  ↓
V3 Oracle-Supervised Neural Reranker
  ↓
Top-5 Chunks
  ↓
Structure-Aware Evidence Aggregator
  ↓
Qwen3.5-2B Stage1 / Stage2 Hybrid
  ↓
Legal-Aware Claim–Evidence Alignment
  ↓
Explicit Citations + Grounding Status
```

## Design constraint

The reranker operates on chunks, not reconstructed full Articles.

```text
chunks → V3 reranker → top chunks → map to legal provision → aggregate
```

This preserves the reranker's training/inference distribution.

## Evidence aggregation

Retrieved chunks are grouped using legal structure when available:

- Article
- standard section
- numeric section
- appendix
- table
- chapter
- mục
- roman section
- fallback chunk

The canonical corpus does not contain a reliable `clause` metadata field, so the system does not invent clause-level structural metadata.