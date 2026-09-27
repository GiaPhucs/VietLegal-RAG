# Source provenance

## Important

The original research workflow was developed interactively in Colab.

A repository audit found no persisted canonical `.py` or `.ipynb`
implementation containing the complete historical retrieval → reranking →
generation pipeline.

Files discovered during the audit were primarily:

- benchmark Markdown;
- Hugging Face checkpoint / adapter README metadata;
- generated research artifacts.

Therefore this public repository does **not** claim that the files under
`src/` are byte-for-byte historical competition source code.

## What `src/` represents

The public package is a clean reference implementation reconstructed from
components that were explicitly verified during the V2.1 portfolio audit.

Implemented directly:

- structure-aware Evidence Aggregator;
- zero-loss post-rerank evidence grouping;
- legal-document-aware citation linker;
- grounding proxy;
- Silver citation evaluator;
- dependency-injected pipeline orchestration.

## External backends

The following components are represented by interfaces because the
canonical historical implementation was not persisted as source code:

- Dense Parent retrieval;
- VietLegal-Harrier retrieval;
- BM25 Parent / Chunk retrieval;
- weighted RRF / parent-local expansion;
- V3 neural reranker runtime;
- Qwen3.5-2B + QLoRA generation runtime.

Model/index implementations can be attached through the interfaces in
`src/vietlegal_rag/backends.py`.

## Frozen architecture constraint

The final architecture preserves:

```text
chunks
  → V3 reranker
  → top chunks
  → map to Article / legal provision
  → Evidence Aggregator
```

Full Articles are not reconstructed and fed back into the reranker.

## Evaluation

Public benchmark summaries are derived from the frozen Dev600 audit.

Citation metrics use answer-aware Silver Oracle evidence for evaluation
only. Silver Oracle evidence is not used during inference.
