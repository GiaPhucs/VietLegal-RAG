from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


@dataclass
class HybridRetrievalConfig:
    """
    Canonical VietLegal-RAG Hybrid Retrieval v1 baseline.

    These weights are reproducible baseline defaults,
    not claims of globally optimal weights.
    """

    dense_parent_k: int = 200
    harrier_chunk_k: int = 500
    bm25_parent_k: int = 200
    bm25_chunk_k: int = 200

    fusion_top_k: int = 50

    rrf_k: int = 60

    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "dense_parent": 1.0,
            "harrier_parent": 1.0,
            "bm25_parent": 1.0,
            "bm25_chunk_parent": 1.0,
        }
    )


def collapse_chunk_results_to_parent(
    chunk_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert a ranked chunk list into a ranked parent-document list.

    Parent rank is determined by the first/highest-ranked chunk
    belonging to each document.

    Example:
        chunk ranks:
            1 -> doc A
            2 -> doc A
            3 -> doc B

        parent ranks:
            1 -> doc A
            2 -> doc B
    """

    output: List[Dict[str, Any]] = []

    seen = set()


    for chunk in chunk_results:

        document_id = str(
            chunk.get(
                "document_id",
                ""
            )
        )


        if not document_id:
            continue


        if document_id in seen:
            continue


        seen.add(
            document_id
        )


        chunk_index = chunk.get(
            "chunk_index"
        )


        if (
            chunk_index is None
            and
            chunk.get(
                "chunk_id"
            )
        ):

            try:

                chunk_index = int(
                    str(
                        chunk[
                            "chunk_id"
                        ]
                    ).rsplit(
                        ":",
                        1
                    )[1]
                )

            except Exception:

                chunk_index = None


        output.append(
            {
                "rank":
                    len(output) + 1,

                "document_id":
                    document_id,

                "source_chunk_id":
                    chunk.get(
                        "chunk_id"
                    ),

                "source_chunk_index":
                    chunk_index,

                "source_chunk_rank":
                    chunk.get(
                        "rank"
                    ),

                "source_score":
                    chunk.get(
                        "score"
                    ),
            }
        )


    return output


def _rank_map(
    rows: List[Dict[str, Any]],
) -> Dict[str, int]:

    return {
        str(
            row[
                "document_id"
            ]
        ):
        int(
            row[
                "rank"
            ]
        )

        for row in rows
        if row.get(
            "document_id"
        )
    }


def weighted_rrf(
    *,
    dense_parent: List[Dict[str, Any]],
    harrier_parent: List[Dict[str, Any]],
    bm25_parent: List[Dict[str, Any]],
    bm25_chunk_parent: List[Dict[str, Any]],
    weights: Mapping[str, float],
    rrf_k: int = 60,
    top_k: int = 50,
) -> List[Dict[str, Any]]:
    """
    Weighted Reciprocal Rank Fusion.

    score(d) =
        Σ_s weight_s / (rrf_k + rank_s(d))
    """

    if rrf_k < 0:
        raise ValueError(
            "rrf_k must be >= 0"
        )


    signal_rows = {
        "dense_parent":
            dense_parent,

        "harrier_parent":
            harrier_parent,

        "bm25_parent":
            bm25_parent,

        "bm25_chunk_parent":
            bm25_chunk_parent,
    }


    rank_maps = {
        name:
            _rank_map(
                rows
            )

        for name, rows
        in signal_rows.items()
    }


    documents = set()


    for rank_map in rank_maps.values():

        documents.update(
            rank_map.keys()
        )


    # Gather metadata from parent-level sources.
    meta_by_doc: Dict[str, Dict[str, Any]] = {}


    for rows in [
        bm25_parent,
        dense_parent,
    ]:

        for row in rows:

            document_id = str(
                row.get(
                    "document_id",
                    ""
                )
            )


            if not document_id:
                continue


            meta = meta_by_doc.setdefault(
                document_id,
                {
                    "name": "",
                    "link": "",
                }
            )


            name = row.get(
                "name"
            )

            link = row.get(
                "link"
            )


            if name and not meta[
                "name"
            ]:

                meta[
                    "name"
                ] = name


            if link and not meta[
                "link"
            ]:

                meta[
                    "link"
                ] = link


    candidates = []


    for document_id in documents:

        score = 0.0

        source_count = 0


        ranks = {}


        for signal_name, rank_map in rank_maps.items():

            rank = rank_map.get(
                document_id
            )

            ranks[
                signal_name
            ] = rank


            if rank is None:
                continue


            source_count += 1


            score += (
                float(
                    weights.get(
                        signal_name,
                        1.0
                    )
                )
                /
                (
                    int(
                        rrf_k
                    )
                    +
                    int(
                        rank
                    )
                )
            )


        metadata = meta_by_doc.get(
            document_id,
            {}
        )


        existing_ranks = [
            r
            for r in ranks.values()
            if r is not None
        ]


        best_rank = (
            min(
                existing_ranks
            )
            if existing_ranks
            else 10**9
        )


        candidates.append(
            {
                "document_id":
                    document_id,

                "rrf_score":
                    float(
                        score
                    ),

                "source_count":
                    source_count,

                "best_source_rank":
                    best_rank,

                "dense_rank":
                    ranks[
                        "dense_parent"
                    ],

                "harrier_parent_rank":
                    ranks[
                        "harrier_parent"
                    ],

                "bm25_parent_rank":
                    ranks[
                        "bm25_parent"
                    ],

                "bm25_chunk_parent_rank":
                    ranks[
                        "bm25_chunk_parent"
                    ],

                "name":
                    metadata.get(
                        "name",
                        ""
                    ),

                "link":
                    metadata.get(
                        "link",
                        ""
                    ),
            }
        )


    # Stable deterministic tie breaking:
    #   1. higher RRF
    #   2. more supporting signals
    #   3. better individual source rank
    #   4. document_id
    candidates.sort(
        key=lambda row: (
            -row[
                "rrf_score"
            ],
            -row[
                "source_count"
            ],
            row[
                "best_source_rank"
            ],
            row[
                "document_id"
            ],
        )
    )


    output = candidates[
        :max(
            0,
            int(
                top_k
            )
        )
    ]


    for rank, row in enumerate(
        output,
        start=1
    ):

        row[
            "rank"
        ] = rank


    return output


class HybridRetriever:
    """
    VietLegal-RAG 4-way hybrid retrieval runtime.

    Signals:
        Dense Parent
        Harrier Chunk -> Parent
        BM25 Parent
        BM25 Chunk -> Parent

    Supports:
        search_with_embeddings(...)
            Useful with precomputed query embeddings.

        search(...)
            Full natural-language path using query encoders.
    """

    def __init__(
        self,
        *,
        dense_parent_index,
        harrier_chunk_index,
        bm25_runtime,
        config: HybridRetrievalConfig | None = None,
    ):

        self.dense_parent_index = (
            dense_parent_index
        )

        self.harrier_chunk_index = (
            harrier_chunk_index
        )

        self.bm25_runtime = (
            bm25_runtime
        )

        self.config = (
            config
            or
            HybridRetrievalConfig()
        )


    def search_with_embeddings(
        self,
        *,
        query: str,
        dense_query_embedding,
        harrier_query_embedding,
    ) -> Dict[str, Any]:

        query = str(
            query
        ).strip()


        if not query:

            raise ValueError(
                "query must not be empty"
            )


        cfg = self.config


        # -----------------------------------------------------------------------------------------
        # 1. Dense Parent
        # -----------------------------------------------------------------------------------------

        dense_parent = (
            self.dense_parent_index
            .search_by_embedding(
                dense_query_embedding,
                top_k=cfg.dense_parent_k,
            )
        )


        # -----------------------------------------------------------------------------------------
        # 2. Harrier Chunk
        # -----------------------------------------------------------------------------------------

        harrier_chunks = (
            self.harrier_chunk_index
            .search_by_embedding(
                harrier_query_embedding,
                top_k=cfg.harrier_chunk_k,
            )
        )


        harrier_parent = (
            collapse_chunk_results_to_parent(
                harrier_chunks
            )
        )


        # -----------------------------------------------------------------------------------------
        # 3/4. BM25
        # -----------------------------------------------------------------------------------------

        bm25_parent = (
            self.bm25_runtime.parent.search(
                query,
                top_k=cfg.bm25_parent_k,
            )
        )


        bm25_chunks = (
            self.bm25_runtime.chunk.search(
                query,
                top_k=cfg.bm25_chunk_k,
            )
        )


        bm25_chunk_parent = (
            collapse_chunk_results_to_parent(
                bm25_chunks
            )
        )


        # -----------------------------------------------------------------------------------------
        # 5. Weighted RRF
        # -----------------------------------------------------------------------------------------

        fused = weighted_rrf(
            dense_parent=dense_parent,
            harrier_parent=harrier_parent,
            bm25_parent=bm25_parent,
            bm25_chunk_parent=bm25_chunk_parent,
            weights=cfg.weights,
            rrf_k=cfg.rrf_k,
            top_k=cfg.fusion_top_k,
        )


        return {
            "query":
                query,

            "config": {
                "dense_parent_k":
                    cfg.dense_parent_k,

                "harrier_chunk_k":
                    cfg.harrier_chunk_k,

                "bm25_parent_k":
                    cfg.bm25_parent_k,

                "bm25_chunk_k":
                    cfg.bm25_chunk_k,

                "fusion_top_k":
                    cfg.fusion_top_k,

                "rrf_k":
                    cfg.rrf_k,

                "weights":
                    dict(
                        cfg.weights
                    ),
            },

            "signal_counts": {
                "dense_parent":
                    len(
                        dense_parent
                    ),

                "harrier_chunk":
                    len(
                        harrier_chunks
                    ),

                "harrier_parent":
                    len(
                        harrier_parent
                    ),

                "bm25_parent":
                    len(
                        bm25_parent
                    ),

                "bm25_chunk":
                    len(
                        bm25_chunks
                    ),

                "bm25_chunk_parent":
                    len(
                        bm25_chunk_parent
                    ),
            },

            "candidates":
                fused,

            # Preserve direct chunk retrieval for the later
            # candidate/evidence assembly stage.
            "direct_chunks": {
                "harrier":
                    harrier_chunks,

                "bm25":
                    bm25_chunks,
            },
        }


    def search(
        self,
        *,
        query: str,
        dense_encoder,
        harrier_encoder,
    ) -> Dict[str, Any]:

        dense_query_embedding = (
            dense_encoder.encode(
                query
            )
        )

        harrier_query_embedding = (
            harrier_encoder.encode(
                query
            )
        )


        return self.search_with_embeddings(
            query=query,
            dense_query_embedding=dense_query_embedding,
            harrier_query_embedding=harrier_query_embedding,
        )
