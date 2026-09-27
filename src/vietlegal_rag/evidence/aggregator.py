from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Tuple


STRUCTURAL_PRIORITY = (
    "article",
    "standard_section",
    "numeric_section",
    "appendix",
    "table",
    "chapter",
    "muc",
    "roman_section",
)


def _nonempty(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def structural_identity(
    chunk: Dict[str, Any],
) -> Tuple[str, str, str]:
    """
    Map a chunk to the highest-priority legal structural unit.

    The canonical corpus has no reliable `clause` metadata field.
    This implementation intentionally does not invent one.
    """

    document_id = str(
        chunk.get(
            "document_id",
            ""
        )
    )

    for level in STRUCTURAL_PRIORITY:

        value = chunk.get(
            level
        )

        if _nonempty(value):

            return (
                document_id,
                level,
                str(value),
            )


    return (
        document_id,
        "chunk",
        str(
            chunk.get(
                "chunk_id",
                ""
            )
        ),
    )


def _citation_label(
    group: List[Dict[str, Any]],
    level: str,
    provision_id: str,
) -> str:

    first = group[0]

    document_name = (
        first.get("name")
        or
        first.get("document_name")
        or
        str(
            first.get(
                "document_id",
                ""
            )
        )
    )

    title_fields = {
        "article":
            "article_title",

        "standard_section":
            "section_title",

        "numeric_section":
            "numeric_section_title",

        "appendix":
            "appendix_title",

        "table":
            "table_title",

        "chapter":
            "chapter",

        "muc":
            "muc",

        "roman_section":
            "roman_section",
    }


    title = first.get(
        title_fields.get(
            level,
            ""
        )
    )


    parts = [
        str(document_name)
    ]


    if level != "chunk":

        parts.append(
            str(provision_id)
        )


    if _nonempty(title):

        if str(title) != str(
            provision_id
        ):

            parts.append(
                str(title)
            )


    return " — ".join(
        x
        for x in parts
        if _nonempty(x)
    )


def aggregate_top_chunks(
    chunks: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Structure-aware aggregation of already-reranked chunks.

    Important:
    - No retrieval expansion occurs here.
    - No chunk is dropped.
    - Rank ordering is preserved.
    - Reranker still operates on chunks BEFORE aggregation.
    """

    ranked = [
        dict(chunk)
        for chunk in chunks
    ]


    groups: OrderedDict[
        Tuple[str, str, str],
        List[Dict[str, Any]]
    ] = OrderedDict()


    for index, chunk in enumerate(
        ranked,
        start=1
    ):

        chunk.setdefault(
            "rank",
            index
        )

        key = structural_identity(
            chunk
        )

        groups.setdefault(
            key,
            []
        ).append(
            chunk
        )


    evidence_units = []


    for unit_index, (
        key,
        members,
    ) in enumerate(
        groups.items(),
        start=1
    ):

        document_id, level, provision_id = (
            key
        )


        members = sorted(
            members,
            key=lambda x:
                int(
                    x.get(
                        "rank",
                        10**9
                    )
                )
        )


        merged_text = "\n\n".join(
            str(
                member.get(
                    "text",
                    ""
                )
            ).strip()

            for member in members

            if str(
                member.get(
                    "text",
                    ""
                )
            ).strip()
        )


        ranks = [
            int(
                member.get(
                    "rank",
                    10**9
                )
            )
            for member in members
        ]


        scores = [
            member.get(
                "reranker_score"
            )

            for member in members

            if member.get(
                "reranker_score"
            )
            is not None
        ]


        evidence_id = (
            f"{document_id}::"
            f"{level}::"
            f"{provision_id}"
        )


        evidence_units.append(
            {
                "evidence_id":
                    evidence_id,

                "document_id":
                    document_id,

                "structural_level":
                    level,

                "provision_id":
                    provision_id,

                "citation_label":
                    _citation_label(
                        members,
                        level,
                        provision_id,
                    ),

                "best_top5_rank":
                    min(
                        ranks
                    ),

                "member_ranks":
                    ranks,

                "max_reranker_score":
                    (
                        max(scores)
                        if scores
                        else None
                    ),

                "merged_text":
                    merged_text,

                "chunks":
                    members,

                "link":
                    members[0].get(
                        "link"
                    ),
            }
        )


    evidence_units.sort(
        key=lambda x:
            x[
                "best_top5_rank"
            ]
    )


    # Zero-loss invariant.
    input_ids = [
        str(
            chunk.get(
                "chunk_id",
                ""
            )
        )
        for chunk in ranked
    ]


    output_ids = [
        str(
            chunk.get(
                "chunk_id",
                ""
            )
        )

        for unit in evidence_units

        for chunk in unit[
            "chunks"
        ]
    ]


    if sorted(
        input_ids
    ) != sorted(
        output_ids
    ):

        raise RuntimeError(
            "Evidence aggregation violated zero-loss invariant."
        )


    return evidence_units
