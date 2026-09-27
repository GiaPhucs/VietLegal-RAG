from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set


def _oracle_chunk_ids(
    oracle_row: Dict[str, Any],
) -> Set[str]:

    return {
        str(
            item[
                "chunk_id"
            ]
        )

        for item in oracle_row.get(
            "oracle_chunks",
            []
        )

        if item.get(
            "chunk_id"
        )
        is not None
    }


def _oracle_document_ids(
    oracle_row: Dict[str, Any],
) -> Set[str]:

    output = set()

    for item in oracle_row.get(
        "oracle_chunks",
        []
    ):

        value = (
            item.get(
                "document_id"
            )
            or
            item.get(
                "parent_id"
            )
        )

        if value is not None:

            output.add(
                str(value)
            )

    return output


def evaluate_silver_citations(
    predicted_units: Iterable[Dict[str, Any]],
    oracle_row: Dict[str, Any],
) -> Dict[str, float]:
    """
    Evaluate citations against answer-aware Silver Oracle evidence.

    These are NOT official gold citation annotations.
    """

    predicted = list(
        predicted_units
    )

    gold_chunks = _oracle_chunk_ids(
        oracle_row
    )

    gold_documents = _oracle_document_ids(
        oracle_row
    )


    exact_hits = 0
    parent_hits = 0

    cited_chunks = set()
    cited_documents = set()


    for unit in predicted:

        member_chunks = {
            str(
                chunk[
                    "chunk_id"
                ]
            )

            for chunk in unit.get(
                "chunks",
                []
            )

            if chunk.get(
                "chunk_id"
            )
            is not None
        }


        document_id = str(
            unit.get(
                "document_id",
                ""
            )
        )


        if (
            member_chunks
            &
            gold_chunks
        ):

            exact_hits += 1


        if (
            document_id
            and
            document_id
            in
            gold_documents
        ):

            parent_hits += 1


        cited_chunks |= (
            member_chunks
        )


        if document_id:

            cited_documents.add(
                document_id
            )


    n_pred = len(
        predicted
    )


    exact_precision = (
        exact_hits / n_pred
        if n_pred
        else 0.0
    )


    parent_precision = (
        parent_hits / n_pred
        if n_pred
        else 0.0
    )


    exact_recall = (
        len(
            cited_chunks
            &
            gold_chunks
        )
        /
        len(
            gold_chunks
        )

        if gold_chunks
        else 0.0
    )


    parent_recall = (
        len(
            cited_documents
            &
            gold_documents
        )
        /
        len(
            gold_documents
        )

        if gold_documents
        else 0.0
    )


    return {
        "silver_exact_precision":
            exact_precision,

        "silver_exact_recall":
            exact_recall,

        "silver_parent_precision":
            parent_precision,

        "silver_parent_recall":
            parent_recall,
    }
