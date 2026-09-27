from __future__ import annotations

from typing import Any, Dict, List, Optional

from .backends import (
    GeneratorBackend,
    RerankerBackend,
    RetrieverBackend,
)

from .evidence import (
    LegalAwareCitationLinker,
    aggregate_top_chunks,
)


class VietLegalRAG:
    """
    Dependency-injected VietLegal-RAG v2.1 orchestrator.

    The public package intentionally does not bundle:
      - corpus indexes
      - model checkpoints
      - LoRA adapters
      - competition data

    Backends must be provided by the caller.
    """

    def __init__(
        self,
        retriever: RetrieverBackend,
        reranker: RerankerBackend,
        generator: GeneratorBackend,
        citation_linker: Optional[
            LegalAwareCitationLinker
        ] = None,
        top_k: int = 5,
    ):

        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.citation_linker = (
            citation_linker
        )

        self.top_k = int(
            top_k
        )


    def answer(
        self,
        question: str,
    ) -> Dict[str, Any]:

        question = str(
            question
        ).strip()


        if not question:

            raise ValueError(
                "question must not be empty"
            )


        candidates = (
            self.retriever.retrieve(
                question
            )
        )


        reranked = (
            self.reranker.rerank(
                question,
                candidates,
            )
        )


        top_chunks = (
            reranked[
                :self.top_k
            ]
        )


        # Important frozen constraint:
        # rerank chunks first, aggregate second.
        evidence_units = (
            aggregate_top_chunks(
                top_chunks
            )
        )


        answer = (
            self.generator.generate(
                question,
                evidence_units,
            )
        )


        output = {
            "question":
                question,

            "answer":
                answer,

            "evidence_units":
                evidence_units,

            "citations":
                [],

            "grounding":
                None,
        }


        if self.citation_linker is None:

            return output


        aligned = (
            self.citation_linker.align_answer(
                answer,
                evidence_units,
            )
        )


        citations = {}

        for claim in aligned:

            evidence_id = claim.get(
                "evidence_id"
            )

            if not evidence_id:

                continue

            if evidence_id not in citations:

                citations[
                    evidence_id
                ] = {
                    "evidence_id":
                        evidence_id,

                    "document_id":
                        claim.get(
                            "document_id"
                        ),

                    "citation_label":
                        claim.get(
                            "citation_label"
                        ),
                }


        supported = sum(
            x[
                "status"
            ]
            ==
            "SUPPORTED"

            for x in aligned
        )


        partial = sum(
            x[
                "status"
            ]
            ==
            "PARTIAL"

            for x in aligned
        )


        unsupported = sum(
            x[
                "status"
            ]
            ==
            "UNSUPPORTED"

            for x in aligned
        )


        n = max(
            1,
            len(
                aligned
            )
        )


        output[
            "citations"
        ] = list(
            citations.values()
        )


        output[
            "claim_alignment"
        ] = aligned


        output[
            "grounding"
        ] = {
            "supported_units":
                supported,

            "partial_units":
                partial,

            "unsupported_units":
                unsupported,

            "supported_rate":
                supported / n,

            "unsupported_rate":
                unsupported / n,

            "method":
                "legal-aware lexical proxy",
        }


        return output
