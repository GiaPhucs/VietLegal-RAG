from __future__ import annotations

from typing import Any, Dict, List
import inspect
import time


class PipelineRuntime:
    """
    Canonical VietLegal-RAG end-to-end orchestration runtime v1.

    Pipeline
    --------
    Question
      -> hybrid retrieval
      -> candidate pool
      -> V3 chunk reranking
      -> Top-K evidence chunks
      -> structured legal evidence
      -> grounded generator
      -> citation + grounding validator
      -> deterministic citation repair when safe
      -> revalidation
      -> trusted final answer renderer

    Shipping invariant
    ------------------
    A model-generated answer is NEVER returned as final unless
    downstream validation reaches final_gate == PASS.
    """

    VERSION = "1.0"

    DEFAULT_EVIDENCE_TOP_K = 5

    DEFAULT_RERANK_BATCH_SIZE = 16


    def __init__(
        self,
        *,
        hybrid_retriever,
        dense_query_encoder,
        harrier_query_encoder,
        candidate_assembler,
        reranker,
        evidence_aggregator,
        generator,
        validator,
        citation_repairer,
        final_renderer,
    ):

        self.hybrid_retriever = (
            hybrid_retriever
        )

        self.dense_query_encoder = (
            dense_query_encoder
        )

        self.harrier_query_encoder = (
            harrier_query_encoder
        )

        self.candidate_assembler = (
            candidate_assembler
        )

        self.reranker = (
            reranker
        )

        self.evidence_aggregator = (
            evidence_aggregator
        )

        self.generator = (
            generator
        )

        self.validator = (
            validator
        )

        self.citation_repairer = (
            citation_repairer
        )

        self.final_renderer = (
            final_renderer
        )


    # =============================================================================================
    # Hybrid API adapter
    # =============================================================================================

    def _hybrid_search(
        self,
        question: str,
    ):

        """
        Bind to the installed verified HybridRetriever.search API.

        The retrieval implementation remains authoritative; this
        helper only handles public parameter naming.
        """

        method = (
            self.hybrid_retriever.search
        )


        sig = inspect.signature(
            method
        )


        params = sig.parameters


        kwargs = {}


        # Query argument.
        if "query" in params:

            kwargs[
                "query"
            ] = question

        elif "question" in params:

            kwargs[
                "question"
            ] = question


        # Dense encoder argument aliases.
        dense_names = [
            "dense_encoder",
            "dense_query_encoder",
            "dense_parent_encoder",
            "encoder",
        ]


        for name in dense_names:

            if name in params:

                kwargs[
                    name
                ] = self.dense_query_encoder

                break


        # Harrier encoder aliases.
        harrier_names = [
            "harrier_encoder",
            "harrier_query_encoder",
            "chunk_encoder",
        ]


        for name in harrier_names:

            if name in params:

                kwargs[
                    name
                ] = self.harrier_query_encoder

                break


        # Preferred keyword invocation.
        try:

            return method(
                **kwargs
            )

        except TypeError as keyword_error:

            # Conservative positional fallback for the canonical
            # query + two encoder contract.
            try:

                return method(
                    question,
                    self.dense_query_encoder,
                    self.harrier_query_encoder,
                )

            except Exception:

                raise RuntimeError(
                    "Unable to invoke HybridRetriever.search with "
                    f"signature {sig}. Keyword error: {keyword_error}"
                )


    # =============================================================================================
    # Evidence conversion
    # =============================================================================================

    @staticmethod
    def _generator_evidence(
        evidence_units: List[
            Dict[str, Any]
        ],
    ) -> List[
        Dict[str, Any]
    ]:

        result = []


        for unit in evidence_units:

            parts = []


            if unit.get(
                "document_name"
            ):

                parts.append(
                    str(
                        unit[
                            "document_name"
                        ]
                    )
                )


            if unit.get(
                "unit_label"
            ):

                parts.append(
                    str(
                        unit[
                            "unit_label"
                        ]
                    )
                )


            if unit.get(
                "unit_title"
            ):

                parts.append(
                    str(
                        unit[
                            "unit_title"
                        ]
                    )
                )


            result.append(
                {
                    "evidence_rank":
                        unit.get(
                            "evidence_rank"
                        ),

                    "evidence_id":
                        unit.get(
                            "evidence_id"
                        ),

                    "citation_label":
                        (
                            f'[E{unit.get("evidence_rank")}]'
                        ),

                    "heading":
                        " — ".join(
                            parts
                        ),

                    "document_id":
                        unit.get(
                            "document_id"
                        ),

                    "document_link":
                        unit.get(
                            "document_link"
                        ),

                    "unit_type":
                        unit.get(
                            "unit_type"
                        ),

                    "unit_label":
                        unit.get(
                            "unit_label"
                        ),

                    "unit_title":
                        unit.get(
                            "unit_title"
                        ),

                    "chunk_ids":
                        unit.get(
                            "chunk_ids",
                            []
                        ),

                    "text":
                        unit.get(
                            "text",
                            ""
                        ),
                }
            )


        return result


    # =============================================================================================
    # Main pipeline
    # =============================================================================================

    def run(
        self,
        question: str,
        *,
        evidence_top_k: int = 5,
        rerank_batch_size: int = 16,
        generator_max_new_tokens: int | None = None,
    ) -> Dict[str, Any]:

        question = str(
            question
        ).strip()


        if not question:

            raise ValueError(
                "question must not be empty"
            )


        timings = {}


        total_start = time.time()


        # -----------------------------------------------------------------------------------------
        # 1. Hybrid retrieval
        # -----------------------------------------------------------------------------------------

        start = time.time()


        hybrid_result = (
            self._hybrid_search(
                question
            )
        )


        timings[
            "hybrid_retrieval"
        ] = (
            time.time()
            -
            start
        )


        # -----------------------------------------------------------------------------------------
        # 2. Candidate pool
        # -----------------------------------------------------------------------------------------

        start = time.time()


        pool_result = (
            self.candidate_assembler
            .assemble(
                query=question,
                hybrid_result=hybrid_result,
            )
        )


        candidates = pool_result[
            "candidates"
        ]


        timings[
            "candidate_pool"
        ] = (
            time.time()
            -
            start
        )


        if not candidates:

            return {
                "status":
                    "NO_CANDIDATES",

                "question":
                    question,

                "final_answer":
                    None,

                "timings":
                    timings,
            }


        # -----------------------------------------------------------------------------------------
        # 3. V3 reranker
        # -----------------------------------------------------------------------------------------

        start = time.time()


        reranked = (
            self.reranker.rerank(
                question,
                candidates,
                batch_size=
                    int(
                        rerank_batch_size
                    ),
                top_k=None,
            )
        )


        timings[
            "v3_rerank"
        ] = (
            time.time()
            -
            start
        )


        if not reranked:

            return {
                "status":
                    "NO_RERANKED_EVIDENCE",

                "question":
                    question,

                "final_answer":
                    None,

                "timings":
                    timings,
            }


        top_chunks = reranked[
            :int(
                evidence_top_k
            )
        ]


        # -----------------------------------------------------------------------------------------
        # 4. Structured evidence
        # -----------------------------------------------------------------------------------------

        start = time.time()


        evidence_result = (
            self.evidence_aggregator
            .aggregate(
                question=question,
                ranked_chunks=top_chunks,
            )
        )


        evidence_units = (
            evidence_result[
                "evidence_units"
            ]
        )


        generator_evidence = (
            self._generator_evidence(
                evidence_units
            )
        )


        timings[
            "evidence_aggregation"
        ] = (
            time.time()
            -
            start
        )


        # -----------------------------------------------------------------------------------------
        # 5. Generator
        # -----------------------------------------------------------------------------------------

        start = time.time()


        generation = (
            self.generator.generate(
                question=question,
                evidence=generator_evidence,
                max_new_tokens=
                    generator_max_new_tokens,
            )
        )


        raw_answer = generation[
            "answer"
        ]


        timings[
            "generation"
        ] = (
            time.time()
            -
            start
        )


        # -----------------------------------------------------------------------------------------
        # 6. First validation
        # -----------------------------------------------------------------------------------------

        start = time.time()


        initial_validation = (
            self.validator.validate(
                answer=raw_answer,
                evidence_units=
                    evidence_units,
            )
        )


        timings[
            "initial_validation"
        ] = (
            time.time()
            -
            start
        )


        working_answer = (
            raw_answer
        )

        citation_repair = None

        final_validation = (
            initial_validation
        )


        # -----------------------------------------------------------------------------------------
        # 7. Safe deterministic citation repair
        # -----------------------------------------------------------------------------------------

        if (
            initial_validation[
                "final_gate"
            ]
            ==
            "REQUIRES_CITATION_REPAIR"
        ):

            start = time.time()


            citation_repair = (
                self.citation_repairer
                .repair(
                    answer=
                        raw_answer,

                    citation_validation=
                        initial_validation[
                            "citation_validation"
                        ],
                )
            )


            timings[
                "citation_repair"
            ] = (
                time.time()
                -
                start
            )


            # Do not guess.
            if not citation_repair[
                "all_repairable"
            ]:

                timings[
                    "total"
                ] = (
                    time.time()
                    -
                    total_start
                )


                return {
                    "status":
                        "BLOCKED_CITATION_REPAIR",

                    "question":
                        question,

                    "raw_answer":
                        raw_answer,

                    "final_answer":
                        None,

                    "hybrid_result":
                        hybrid_result,

                    "candidate_pool":
                        pool_result,

                    "top_chunks":
                        top_chunks,

                    "evidence":
                        evidence_result,

                    "generation":
                        generation,

                    "initial_validation":
                        initial_validation,

                    "citation_repair":
                        citation_repair,

                    "timings":
                        timings,
                }


            working_answer = (
                citation_repair[
                    "answer"
                ]
            )


            # Mandatory post-repair validation.
            start = time.time()


            final_validation = (
                self.validator.validate(
                    answer=
                        working_answer,

                    evidence_units=
                        evidence_units,
                )
            )


            timings[
                "post_repair_validation"
            ] = (
                time.time()
                -
                start
            )


        # -----------------------------------------------------------------------------------------
        # 8. Shipping gate
        # -----------------------------------------------------------------------------------------

        if (
            final_validation[
                "final_gate"
            ]
            !=
            "PASS"
        ):

            timings[
                "total"
            ] = (
                time.time()
                -
                total_start
            )


            return {
                "status":
                    (
                        "BLOCKED_"
                        +
                        str(
                            final_validation[
                                "final_gate"
                            ]
                        )
                ),

                "question":
                    question,

                "raw_answer":
                    raw_answer,

                "working_answer":
                    working_answer,

                "final_answer":
                    None,

                "hybrid_result":
                    hybrid_result,

                "candidate_pool":
                    pool_result,

                "top_chunks":
                    top_chunks,

                "evidence":
                    evidence_result,

                "generation":
                    generation,

                "initial_validation":
                    initial_validation,

                "citation_repair":
                    citation_repair,

                "final_validation":
                    final_validation,

                "timings":
                    timings,
            }


        # -----------------------------------------------------------------------------------------
        # 9. Final renderer
        # -----------------------------------------------------------------------------------------

        start = time.time()


        rendered = (
            self.final_renderer.render(
                answer=
                    working_answer,

                evidence_units=
                    evidence_units,

                grounding_validation=
                    final_validation[
                        "grounding_validation"
                    ],
            )
        )


        timings[
            "final_render"
        ] = (
            time.time()
            -
            start
        )


        timings[
            "total"
        ] = (
            time.time()
            -
            total_start
        )


        return {
            "status":
                "PASS",

            "question":
                question,

            "final_answer":
                rendered[
                    "final_answer"
                ],

            "raw_answer":
                raw_answer,

            "working_answer":
                working_answer,

            "hybrid_result":
                hybrid_result,

            "candidate_pool":
                pool_result,

            "top_chunks":
                top_chunks,

            "evidence":
                evidence_result,

            "generation":
                generation,

            "initial_validation":
                initial_validation,

            "citation_repair":
                citation_repair,

            "final_validation":
                final_validation,

            "rendering":
                rendered,

            "timings":
                timings,
        }
