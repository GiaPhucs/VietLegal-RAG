from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
from collections import defaultdict
import re
import unicodedata


_TOKEN_RE = re.compile(
    r"\w+",
    flags=re.UNICODE
)


def _tokenize(
    text: str,
) -> List[str]:

    text = unicodedata.normalize(
        "NFKC",
        str(
            text or ""
        )
    ).lower()

    return _TOKEN_RE.findall(
        text
    )


@dataclass
class CandidatePoolConfig:
    """
    Canonical VietLegal-RAG Candidate Pool v1.

    local_radius=1 means:
        previous chunk + seed + next chunk
        => maximum local3.
    """

    fused_parent_k: int = 50

    local_radius: int = 1

    harrier_direct_k: int = 500

    bm25_direct_k: int = 200

    max_candidates: int = 900


class CandidatePoolAssembler:
    """
    Build bounded chunk-level reranker candidates.

    Sources:
        1. parent-local expansion from fused parent documents
        2. direct Harrier chunks
        3. direct BM25 chunks

    Whole documents/articles are never emitted.
    """

    def __init__(
        self,
        corpus_store,
        config: CandidatePoolConfig | None = None,
    ):

        self.corpus_store = (
            corpus_store
        )

        self.config = (
            config
            or
            CandidatePoolConfig()
        )


    @staticmethod
    def _direct_maps(
        harrier_chunks: List[Dict[str, Any]],
        bm25_chunks: List[Dict[str, Any]],
    ):

        harrier_by_doc = defaultdict(
            list
        )

        bm25_by_doc = defaultdict(
            list
        )


        for row in harrier_chunks:

            document_id = str(
                row.get(
                    "document_id",
                    ""
                )
            )

            if document_id:

                harrier_by_doc[
                    document_id
                ].append(
                    row
                )


        for row in bm25_chunks:

            document_id = str(
                row.get(
                    "document_id",
                    ""
                )
            )

            if document_id:

                bm25_by_doc[
                    document_id
                ].append(
                    row
                )


        return (
            harrier_by_doc,
            bm25_by_doc,
        )


    @staticmethod
    def _best_direct_seed(
        document_id: str,
        harrier_by_doc,
        bm25_by_doc,
    ) -> Dict[str, Any] | None:
        """
        Pick the strongest direct-retrieval chunk seed.

        We compare normalized reciprocal ranks rather than raw scores,
        because Harrier and BM25 scores are on different scales.
        """

        candidates = []


        for row in harrier_by_doc.get(
            document_id,
            []
        ):

            rank = int(
                row.get(
                    "rank",
                    10**9
                )
            )

            candidates.append(
                (
                    1.0
                    /
                    (
                        60
                        +
                        rank
                    ),
                    "harrier",
                    row,
                )
            )


        for row in bm25_by_doc.get(
            document_id,
            []
        ):

            rank = int(
                row.get(
                    "rank",
                    10**9
                )
            )

            candidates.append(
                (
                    1.0
                    /
                    (
                        60
                        +
                        rank
                    ),
                    "bm25",
                    row,
                )
            )


        if not candidates:

            return None


        candidates.sort(
            key=lambda x:
                (
                    -x[0],
                    x[1],
                )
        )


        source = candidates[
            0
        ][
            1
        ]

        row = dict(
            candidates[
                0
            ][
                2
            ]
        )


        row[
            "_seed_source"
        ] = source


        return row


    def _lexical_seed(
        self,
        query: str,
        document_id: str,
    ) -> Dict[str, Any] | None:
        """
        Fallback only when no direct Harrier/BM25 seed exists.

        Select one chunk inside the parent by distinct query-token
        overlap. This is bounded parent-local selection, not whole-doc
        reranking.
        """

        chunks = (
            self.corpus_store
            .get_document_chunks(
                document_id
            )
        )


        if not chunks:

            return None


        query_tokens = set(
            _tokenize(
                query
            )
        )


        best = None
        best_key = None


        for chunk in chunks:

            tokens = set(
                _tokenize(
                    chunk.get(
                        "retrieval_text",
                        ""
                    )
                )
            )


            overlap = len(
                query_tokens
                &
                tokens
            )


            coverage = (
                overlap
                /
                max(
                    1,
                    len(
                        query_tokens
                    )
                )
            )


            # Deterministic tie breaker:
            # higher overlap -> higher coverage -> earlier chunk
            key = (
                overlap,
                coverage,
                -int(
                    chunk.get(
                        "chunk_index",
                        0
                    )
                ),
            )


            if (
                best is None
                or
                key > best_key
            ):

                best = chunk
                best_key = key


        if best is None:
            return None


        return {
            "chunk_id":
                best[
                    "chunk_id"
                ],

            "document_id":
                best[
                    "document_id"
                ],

            "chunk_index":
                best[
                    "chunk_index"
                ],

            "_seed_source":
                "lexical_fallback",

            "_lexical_overlap":
                best_key[
                    0
                ],
        }


    def _local_chunks(
        self,
        query: str,
        parent_row: Dict[str, Any],
        harrier_by_doc,
        bm25_by_doc,
    ) -> List[Dict[str, Any]]:

        document_id = str(
            parent_row[
                "document_id"
            ]
        )


        seed = self._best_direct_seed(
            document_id,
            harrier_by_doc,
            bm25_by_doc,
        )


        if seed is None:

            seed = self._lexical_seed(
                query,
                document_id,
            )


        if seed is None:

            return []


        seed_chunk_id = str(
            seed[
                "chunk_id"
            ]
        )


        neighbors = (
            self.corpus_store
            .get_neighbors(
                seed_chunk_id,
                radius=self.config.local_radius,
            )
        )


        output = []


        for chunk in neighbors:

            row = dict(
                chunk
            )

            row[
                "_local_seed_chunk_id"
            ] = seed_chunk_id

            row[
                "_local_seed_source"
            ] = seed.get(
                "_seed_source"
            )

            row[
                "_fused_parent_rank"
            ] = parent_row.get(
                "rank"
            )

            output.append(
                row
            )


        return output


    def assemble(
        self,
        *,
        query: str,
        hybrid_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        query = str(
            query
        ).strip()


        if not query:

            raise ValueError(
                "query must not be empty"
            )


        cfg = self.config


        fused_parents = (
            hybrid_result.get(
                "candidates",
                []
            )[
                :cfg.fused_parent_k
            ]
        )


        direct = hybrid_result.get(
            "direct_chunks",
            {}
        )


        harrier_chunks = (
            direct.get(
                "harrier",
                []
            )[
                :cfg.harrier_direct_k
            ]
        )


        bm25_chunks = (
            direct.get(
                "bm25",
                []
            )[
                :cfg.bm25_direct_k
            ]
        )


        (
            harrier_by_doc,
            bm25_by_doc,
        ) = self._direct_maps(
            harrier_chunks,
            bm25_chunks,
        )


        # -----------------------------------------------------------------------------------------
        # Registry
        # -----------------------------------------------------------------------------------------

        registry: Dict[
            str,
            Dict[str, Any]
        ] = {}


        insertion_order = []


        def ensure_candidate(
            chunk_id: str,
        ):

            chunk_id = str(
                chunk_id
            )


            if chunk_id in registry:

                return registry[
                    chunk_id
                ]


            hydrated = (
                self.corpus_store
                .get_chunk(
                    chunk_id
                )
            )


            if hydrated is None:

                return None


            candidate = dict(
                hydrated
            )


            candidate.update(
                {
                    "sources":
                        [],

                    "harrier_rank":
                        None,

                    "bm25_rank":
                        None,

                    "fused_parent_rank":
                        None,

                    "local_seed_chunk_id":
                        None,

                    "local_seed_source":
                        None,
                }
            )


            registry[
                chunk_id
            ] = candidate

            insertion_order.append(
                chunk_id
            )


            return candidate


        def add_source(
            chunk_id: str,
            source: str,
            **kwargs,
        ):

            candidate = ensure_candidate(
                chunk_id
            )


            if candidate is None:

                return


            if source not in candidate[
                "sources"
            ]:

                candidate[
                    "sources"
                ].append(
                    source
                )


            for key, value in kwargs.items():

                if value is None:
                    continue


                old = candidate.get(
                    key
                )


                # Ranks: keep the best/smallest.
                if key.endswith(
                    "_rank"
                ):

                    if (
                        old is None
                        or
                        int(
                            value
                        )
                        <
                        int(
                            old
                        )
                    ):

                        candidate[
                            key
                        ] = int(
                            value
                        )

                elif old is None:

                    candidate[
                        key
                    ] = value


        # -----------------------------------------------------------------------------------------
        # 1. Parent-local3
        # -----------------------------------------------------------------------------------------

        local_seed_sources = {
            "harrier":
                0,

            "bm25":
                0,

            "lexical_fallback":
                0,
        }


        local_parent_count = 0


        for parent in fused_parents:

            local = self._local_chunks(
                query,
                parent,
                harrier_by_doc,
                bm25_by_doc,
            )


            if not local:
                continue


            local_parent_count += 1


            seed_source = local[
                0
            ].get(
                "_local_seed_source"
            )


            if seed_source in local_seed_sources:

                local_seed_sources[
                    seed_source
                ] += 1


            for chunk in local:

                add_source(
                    chunk[
                        "chunk_id"
                    ],
                    "parent_local",

                    fused_parent_rank=
                        parent.get(
                            "rank"
                        ),

                    local_seed_chunk_id=
                        chunk.get(
                            "_local_seed_chunk_id"
                        ),

                    local_seed_source=
                        chunk.get(
                            "_local_seed_source"
                        ),
                )


        # -----------------------------------------------------------------------------------------
        # 2. Direct Harrier
        # -----------------------------------------------------------------------------------------

        for row in harrier_chunks:

            add_source(
                row[
                    "chunk_id"
                ],
                "harrier_direct",

                harrier_rank=
                    row.get(
                        "rank"
                    ),
            )


        # -----------------------------------------------------------------------------------------
        # 3. Direct BM25
        # -----------------------------------------------------------------------------------------

        for row in bm25_chunks:

            add_source(
                row[
                    "chunk_id"
                ],
                "bm25_direct",

                bm25_rank=
                    row.get(
                        "rank"
                    ),
            )


        # -----------------------------------------------------------------------------------------
        # 4. Bounded final pool
        #
        # Dedupe already happened through registry.
        # Preserve insertion order:
        #     local3 -> Harrier -> BM25
        # -----------------------------------------------------------------------------------------

        selected_ids = insertion_order[
            :cfg.max_candidates
        ]


        candidates = [
            registry[
                chunk_id
            ]

            for chunk_id in selected_ids
        ]


        for candidate_order, row in enumerate(
            candidates,
            start=1
        ):

            row[
                "candidate_order"
            ] = candidate_order


        source_counts = defaultdict(
            int
        )


        for row in candidates:

            for source in row[
                "sources"
            ]:

                source_counts[
                    source
                ] += 1


        return {
            "query":
                query,

            "config": {
                "fused_parent_k":
                    cfg.fused_parent_k,

                "local_radius":
                    cfg.local_radius,

                "local_window_max":
                    (
                        2
                        *
                        cfg.local_radius
                        +
                        1
                    ),

                "harrier_direct_k":
                    cfg.harrier_direct_k,

                "bm25_direct_k":
                    cfg.bm25_direct_k,

                "max_candidates":
                    cfg.max_candidates,
            },

            "diagnostics": {
                "fused_parents_used":
                    len(
                        fused_parents
                    ),

                "parents_with_local_expansion":
                    local_parent_count,

                "local_seed_sources":
                    dict(
                        local_seed_sources
                    ),

                "unique_candidates":
                    len(
                        candidates
                    ),

                "source_counts":
                    dict(
                        source_counts
                    ),
            },

            "candidates":
                candidates,
        }
