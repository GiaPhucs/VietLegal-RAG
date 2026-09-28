# VietLegal-RAG v2.1.0

## Release status

This release is the reproducibility-certified runtime baseline for VietLegal-RAG.

## Architecture

- Four-way hybrid retrieval: Dense Parent, VietLegal-Harrier Chunk, BM25 Parent, BM25 Chunk
- Weighted Reciprocal Rank Fusion
- Parent-local and direct-chunk candidate assembly
- V3 chunk-only neural reranking
- Structured legal evidence aggregation
- Qwen3.5-2B + Stage2-v2 grounded generation
- Legal citation and grounding validation
- Deterministic evidence-backed citation repair
- Mandatory post-repair validation
- Trusted [E#] evidence rendering

## Reproducibility certification

The release workflow was verified using a source-only clean clone and public runtime artifacts:

```text
source-only clean clone
  -> pip install -e .
  -> download 151 artifacts from public commit-pinned URLs
  -> verify exact file count and byte count
  -> verify 151/151 SHA256 hashes
  -> portable CLI check
  -> full end-to-end inference
  -> final shipping gate PASS
```

## Public runtime artifacts

- Provider: Hugging Face Hub
- Repository: `gphs09/VietLegal-RAG-artifacts`
- Immutable revision: `17876627834fa9982b5c57cad3fb54394591ee32`
- Artifact files: 151
- Artifact bundle: 4.592 GB

## Canonical smoke result

- Pipeline status: `PASS`
- Final shipping gate: `PASS`
- Citation mismatches: `0`
- Unsupported citations: `0`
- Unsupported substantive claims: `0`
- Pipeline runtime after model initialization: `68.76978921890259 s`
- Full wall time including model startup: `162.59432816505432 s`

The timing above is a single-query release smoke measurement, not a benchmark claim.

## Artifact downloader

- Public commit-pinned URLs
- `.part` interrupted-download recovery
- HTTP Range resume when supported
- transient-failure retry support
- atomic completion
- SHA256 integrity verification

## Important runtime policy

Generated legal citations are not trusted automatically. Final answers are emitted only after the downstream citation/grounding shipping gate reaches `PASS`.

