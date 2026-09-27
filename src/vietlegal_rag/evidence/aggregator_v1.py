from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List
import hashlib


class EvidenceAggregatorV1:
    """
    Canonical VietLegal-RAG structured legal evidence aggregator.

    Input unit:
        V3-ranked canonical chunks.

    Output unit:
        legal evidence provision.

    Aggregation rule:
        same document_id
        +
        same detected legal provision

    Different legal provisions are never merged simply because
    they come from the same document.
    """


    PROVISION_FIELDS = [
        ("article", "article_title", "article"),
        (
            "numeric_section",
            "numeric_section_title",
            "numeric_section",
        ),
        (
            "standard_section",
            "section_title",
            "standard_section",
        ),
        (
            "appendix",
            "appendix_title",
            "appendix",
        ),
        (
            "roman_section",
            None,
            "roman_section",
        ),
        (
            "muc",
            None,
            "muc",
        ),
        (
            "chapter",
            None,
            "chapter",
        ),
    ]


    @classmethod
    def detect_provision(
        cls,
        chunk: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Detect the most specific available legal provision.

        Priority:
            Article
            Numeric section
            Standard section
            Appendix
            Roman section
            Mục
            Chapter
            Chunk fallback
        """

        for (
            value_field,
            title_field,
            unit_type,
        ) in cls.PROVISION_FIELDS:

            value = chunk.get(
                value_field
            )


            if (
                value is None
                or
                not str(
                    value
                ).strip()
            ):

                continue


            title = (
                chunk.get(
                    title_field
                )
                if title_field
                else None
            )


            return {
                "unit_type":
                    unit_type,

                "unit_label":
                    str(
                        value
                    ).strip(),

                "unit_title":
                    (
                        str(
                            title
                        ).strip()
                        if (
                            title is not None
                            and
                            str(
                                title
                            ).strip()
                        )
                        else None
                    ),
            }


        # No structural metadata available:
        # keep chunk independent.
        return {
            "unit_type":
                "chunk",

            "unit_label":
                str(
                    chunk[
                        "chunk_id"
                    ]
                ),

            "unit_title":
                None,
        }


    @staticmethod
    def _evidence_id(
        document_id: str,
        unit_type: str,
        unit_label: str,
    ) -> str:

        raw = (
            f"{document_id}|"
            f"{unit_type}|"
            f"{unit_label}"
        )


        digest = hashlib.sha1(
            raw.encode(
                "utf-8"
            )
        ).hexdigest()[
            :12
        ]


        return (
            f"ev_{digest}"
        )


    def aggregate(
        self,
        *,
        question: str,
        ranked_chunks: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:

        question = str(
            question
        ).strip()


        if not question:

            raise ValueError(
                "question must not be empty"
            )


        if not ranked_chunks:

            return {
                "question":
                    question,

                "original_top_k":
                    0,

                "evidence_unit_count":
                    0,

                "evidence_units":
                    [],
            }


        groups = OrderedDict()


        # -----------------------------------------------------------------------------------------
        # Group by exact legal provision.
        #
        # Iterating in V3-rank order means first insertion preserves
        # evidence priority.
        # -----------------------------------------------------------------------------------------

        for chunk in ranked_chunks:

            if not chunk.get(
                "chunk_id"
            ):

                raise ValueError(
                    "Evidence chunk missing chunk_id."
                )


            document_id = str(
                chunk.get(
                    "document_id",
                    ""
                )
            )


            if not document_id:

                raise ValueError(
                    "Evidence chunk missing document_id."
                )


            provision = (
                self.detect_provision(
                    chunk
                )
            )


            group_key = (
                document_id,
                provision[
                    "unit_type"
                ],
                provision[
                    "unit_label"
                ],
            )


            if group_key not in groups:

                groups[
                    group_key
                ] = {
                    "document_id":
                        document_id,

                    "document_name":
                        chunk.get(
                            "name"
                        ),

                    "document_link":
                        chunk.get(
                            "link"
                        ),

                    "unit_type":
                        provision[
                            "unit_type"
                        ],

                    "unit_label":
                        provision[
                            "unit_label"
                        ],

                    "unit_title":
                        provision[
                            "unit_title"
                        ],

                    "chunks":
                        [],
                }


            groups[
                group_key
            ][
                "chunks"
            ].append(
                chunk
            )


        # -----------------------------------------------------------------------------------------
        # Build units.
        # -----------------------------------------------------------------------------------------

        evidence_units = []


        for group in groups.values():

            chunks = list(
                group[
                    "chunks"
                ]
            )


            # Legal reading order.
            chunks_by_index = sorted(
                chunks,
                key=lambda row:
                    (
                        int(
                            row.get(
                                "chunk_index",
                                10**9
                            )
                        ),
                        str(
                            row[
                                "chunk_id"
                            ]
                        ),
                    )
            )


            # Ranking priority is still based on best V3-ranked chunk.
            best_v3_rank = min(
                int(
                    row.get(
                        "v3_rank",
                        10**9
                    )
                )
                for row in chunks
            )


            max_v3_score = max(
                float(
                    row.get(
                        "v3_score",
                        float(
                            "-inf"
                        )
                    )
                )
                for row in chunks
            )


            chunk_ids = [
                str(
                    row[
                        "chunk_id"
                    ]
                )
                for row in chunks_by_index
            ]


            chunk_indices = [
                int(
                    row[
                        "chunk_index"
                    ]
                )
                for row in chunks_by_index
            ]


            texts = []


            for row in chunks_by_index:

                text = (
                    row.get(
                        "text"
                    )
                    or
                    row.get(
                        "retrieval_text"
                    )
                    or
                    ""
                )


                text = str(
                    text
                ).strip()


                if text:

                    texts.append(
                        text
                    )


            # Deduplicate identical text fragments while preserving order.
            seen_text = set()
            unique_texts = []


            for text in texts:

                if text in seen_text:

                    continue


                seen_text.add(
                    text
                )

                unique_texts.append(
                    text
                )


            merged_text = "\n\n".join(
                unique_texts
            )


            sources = []


            for row in chunks:

                for source in row.get(
                    "sources",
                    []
                ):

                    if source not in sources:

                        sources.append(
                            source
                        )


            evidence_id = self._evidence_id(
                group[
                    "document_id"
                ],
                group[
                    "unit_type"
                ],
                group[
                    "unit_label"
                ],
            )


            evidence_units.append(
                {
                    "evidence_id":
                        evidence_id,

                    "document_id":
                        group[
                            "document_id"
                        ],

                    "document_name":
                        group[
                            "document_name"
                        ],

                    "document_link":
                        group[
                            "document_link"
                        ],

                    "unit_type":
                        group[
                            "unit_type"
                        ],

                    "unit_label":
                        group[
                            "unit_label"
                        ],

                    "unit_title":
                        group[
                            "unit_title"
                        ],

                    "chunk_ids":
                        chunk_ids,

                    "chunk_indices":
                        chunk_indices,

                    "chunk_count":
                        len(
                            chunk_ids
                        ),

                    "best_v3_rank":
                        best_v3_rank,

                    "max_v3_score":
                        max_v3_score,

                    "sources":
                        sources,

                    "text":
                        merged_text,
                }
            )


        # Evidence ordering follows the best chunk V3 rank.
        # Stable fallback uses document/provision IDs.
        evidence_units.sort(
            key=lambda row:
                (
                    row[
                        "best_v3_rank"
                    ],
                    -row[
                        "max_v3_score"
                    ],
                    row[
                        "document_id"
                    ],
                    row[
                        "unit_label"
                    ],
                )
        )


        for rank, unit in enumerate(
            evidence_units,
            start=1
        ):

            unit[
                "evidence_rank"
            ] = rank


        return {
            "question":
                question,

            "original_top_k":
                len(
                    ranked_chunks
                ),

            "evidence_unit_count":
                len(
                    evidence_units
                ),

            "aggregation_count":
                (
                    len(
                        ranked_chunks
                    )
                    -
                    len(
                        evidence_units
                    )
                ),

            "evidence_units":
                evidence_units,
        }
