from __future__ import annotations

from typing import Any, Dict, List, Protocol


Chunk = Dict[str, Any]


class RetrieverBackend(Protocol):
    """
    Retrieval contract.

    Intended production implementation:
      Dense Parent
      + VietLegal-Harrier direct Chunk
      + BM25 Parent
      + BM25 Chunk
      + weighted RRF
      + parent-local expansion
    """

    def retrieve(
        self,
        question: str,
    ) -> List[Chunk]:
        ...


class RerankerBackend(Protocol):
    """
    Neural reranker contract.

    Portfolio system:
      AITeamVN/Vietnamese_Reranker
      + V3 Oracle-supervised adapter.

    Public repository does not bundle private/local checkpoints.
    """

    def rerank(
        self,
        question: str,
        candidates: List[Chunk],
    ) -> List[Chunk]:
        ...


class GeneratorBackend(Protocol):
    """
    Grounded answer generator contract.

    Portfolio system:
      Qwen/Qwen3.5-2B
      + project QLoRA adapter.

    Generator receives evidence only after reranking.
    """

    def generate(
        self,
        question: str,
        evidence_units: List[Dict[str, Any]],
    ) -> str:
        ...
