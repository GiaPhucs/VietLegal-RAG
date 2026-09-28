# Third-Party Notices

The Apache-2.0 license in the root of this repository applies to the original VietLegal-RAG source code distributed in this repository. It does not relicense third-party models, model weights, datasets, or legal-text sources.

## Upstream neural models

The following upstream Hugging Face repositories displayed `apache-2.0` as their license when this notice was prepared.

| Component | Repository | Upstream license |
|---|---|---|
| Generator | `Qwen/Qwen3.5-2B` | Apache-2.0 |
| Dense Parent encoder | `AITeamVN/Vietnamese_Embedding_v2` | Apache-2.0 |
| Neural reranker base | `AITeamVN/Vietnamese_Reranker` | Apache-2.0 |
| Legal chunk retriever | `mainguyen9/vietlegal-harrier-0.6b` | Apache-2.0 |

Upstream repositories:

- https://huggingface.co/Qwen/Qwen3.5-2B
- https://huggingface.co/AITeamVN/Vietnamese_Embedding_v2
- https://huggingface.co/AITeamVN/Vietnamese_Reranker
- https://huggingface.co/mainguyen9/vietlegal-harrier-0.6b

Users remain responsible for reviewing the current upstream licenses, model cards, base-model terms, and any other applicable restrictions before redistribution or deployment.

## Project-specific adapters and runtime artifacts

The public artifact repository contains project-specific LoRA adapters, retrieval indexes, metadata, and SQLite runtime stores used to reproduce VietLegal-RAG v2.1.0.

The repository-level Apache-2.0 license does not override licenses or terms inherited from upstream base models or source data.

## Legal-text corpus and data

The canonical VietLegal-RAG runtime uses a structured Vietnamese legal-text corpus and historical competition-era data.

**This repository does not currently claim a single blanket license for the underlying legal-text corpus.**

Legal documents, competition-provided data, and any derived corpus materials remain subject to their original source, dataset, competition, or redistribution terms where applicable.

Before redistributing the corpus or using it commercially, verify the licensing and provenance of the original data sources.

## No legal-advice warranty

VietLegal-RAG is a research and engineering project. The presence of legal text or generated legal answers does not constitute professional legal advice.
