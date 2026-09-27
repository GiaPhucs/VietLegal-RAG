from __future__ import annotations

from typing import Any, Dict, List
import re


class CitationRepairer:
    """
    Deterministic legal citation repair.

    Auto-repair is permitted ONLY when:
        - validator status == MISMATCH
        - the validator provides exactly ONE trusted evidence candidate

    Unsupported or ambiguous citations are never guessed.
    """

    TYPE_LABELS = {
        "nghi_dinh":
            "Nghị định",

        "thong_tu":
            "Thông tư",

        "thong_tu_lien_tich":
            "Thông tư liên tịch",

        "quyet_dinh":
            "Quyết định",

        "nghi_quyet":
            "Nghị quyết",

        "phap_lenh":
            "Pháp lệnh",
    }


    SUFFIX_DISPLAY = {
        "nd-cp":
            "NĐ-CP",

        "tt-bca":
            "TT-BCA",

        "tt-btc":
            "TT-BTC",

        "tt-bqp":
            "TT-BQP",

        "tt-bldtbxh":
            "TT-BLĐTBXH",

        "tt-bgdđt":
            "TT-BGDĐT",

        "ttlt-bca-bqp-bng":
            "TTLT-BCA-BQP-BNG",
    }


    @classmethod
    def _display_suffix(
        cls,
        suffix: str,
    ) -> str:

        suffix = str(
            suffix or ""
        ).strip().lower()


        if suffix in cls.SUFFIX_DISPLAY:

            return cls.SUFFIX_DISPLAY[
                suffix
            ]


        return suffix.upper()


    @classmethod
    def format_numbered_instrument(
        cls,
        candidate: Dict[str, Any],
    ) -> str:

        instrument_type = str(
            candidate.get(
                "type",
                ""
            )
        )


        if (
            instrument_type
            not in cls.TYPE_LABELS
        ):

            raise ValueError(
                "Unsupported instrument type for deterministic repair: "
                f"{instrument_type}"
            )


        number = str(
            candidate.get(
                "number",
                ""
            )
        ).strip()


        year = str(
            candidate.get(
                "year",
                ""
            )
        ).strip()


        suffix = cls._display_suffix(
            candidate.get(
                "suffix",
                ""
            )
        )


        if not (
            number
            and
            year
            and
            suffix
        ):

            raise ValueError(
                "Incomplete trusted citation candidate."
            )


        return (
            f"{cls.TYPE_LABELS[instrument_type]} "
            f"{number}/{year}/{suffix}"
        )


    @classmethod
    def repair(
        cls,
        *,
        answer: str,
        citation_validation: Dict[str, Any],
    ) -> Dict[str, Any]:

        repaired = str(
            answer
        )


        actions = []

        blocked = []


        for citation in citation_validation.get(
            "citations",
            []
        ):

            status = citation.get(
                "status"
            )


            raw = str(
                citation.get(
                    "raw",
                    ""
                )
            )


            if status == "VALID":

                continue


            if status == "UNSUPPORTED":

                blocked.append(
                    {
                        "raw":
                            raw,

                        "reason":
                            (
                                "unsupported_citation_"
                                "cannot_be_auto_repaired"
                            ),
                    }
                )

                continue


            if status != "MISMATCH":

                blocked.append(
                    {
                        "raw":
                            raw,

                        "reason":
                            (
                                "unknown_citation_status_"
                                f"{status}"
                            ),
                    }
                )

                continue


            candidates = citation.get(
                "evidence_candidates",
                []
            )


            # HARD deterministic requirement.
            if len(
                candidates
            ) != 1:

                blocked.append(
                    {
                        "raw":
                            raw,

                        "reason":
                            "ambiguous_mismatch",

                        "candidate_count":
                            len(
                                candidates
                            ),
                    }
                )

                continue


            candidate = candidates[
                0
            ]


            replacement = (
                cls.format_numbered_instrument(
                    candidate
                )
            )


            if raw not in repaired:

                blocked.append(
                    {
                        "raw":
                            raw,

                        "reason":
                            "citation_text_not_found_in_answer",
                    }
                )

                continue


            # Replace exactly one occurrence corresponding to
            # this validator citation.
            repaired = repaired.replace(
                raw,
                replacement,
                1,
            )


            actions.append(
                {
                    "original":
                        raw,

                    "replacement":
                        replacement,

                    "reason":
                        (
                            "single_trusted_evidence_"
                            "candidate"
                        ),

                    "evidence_ids":
                        candidate.get(
                            "evidence_ids",
                            []
                        ),
                }
            )


        return {
            "answer":
                repaired,

            "repair_count":
                len(
                    actions
                ),

            "actions":
                actions,

            "blocked_repairs":
                blocked,

            "all_repairable":
                len(
                    blocked
                )
                ==
                0,
        }


class FinalAnswerRenderer:
    """
    Render a validated answer with trusted [E#] evidence markers.

    Evidence markers are derived from validator best-evidence assignments,
    never from the generator's own citation claims.
    """


    @staticmethod
    def _label_map(
        evidence_units: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, str]:

        result = {}


        for unit in evidence_units:

            evidence_id = str(
                unit.get(
                    "evidence_id",
                    ""
                )
            )


            rank = unit.get(
                "evidence_rank"
            )


            if (
                evidence_id
                and
                rank is not None
            ):

                result[
                    evidence_id
                ] = (
                    f"[E{int(rank)}]"
                )


        return result


    @classmethod
    def render(
        cls,
        *,
        answer: str,
        evidence_units: List[
            Dict[str, Any]
        ],
        grounding_validation: Dict[str, Any],
    ) -> Dict[str, Any]:

        rendered = str(
            answer
        )


        label_map = cls._label_map(
            evidence_units
        )


        used_ids = []


        # Add evidence labels from last claim to first claim.
        #
        # This avoids earlier insertions interfering with later exact
        # substring searches.
        claims = list(
            grounding_validation.get(
                "claims",
                []
            )
        )


        for claim in reversed(
            claims
        ):

            if claim.get(
                "status"
            ) not in {
                "SUPPORTED",
                "PARTIAL",
            }:

                continue


            best = claim.get(
                "best_evidence"
            )


            if not best:

                continue


            evidence_id = str(
                best.get(
                    "evidence_id",
                    ""
                )
            )


            marker = label_map.get(
                evidence_id
            )


            if not marker:

                continue


            claim_text = str(
                claim.get(
                    "claim",
                    ""
                )
            ).strip()


            if not claim_text:

                continue


            # Avoid duplicate marker if renderer is called twice.
            replacement = (
                f"{claim_text} {marker}"
            )


            pos = rendered.rfind(
                claim_text
            )


            if pos < 0:

                continue


            tail_start = (
                pos
                +
                len(
                    claim_text
                )
            )


            already = rendered[
                tail_start:
                tail_start
                +
                len(
                    marker
                )
                +
                2
            ]


            if marker in already:

                continue


            rendered = (
                rendered[
                    :pos
                ]
                +
                replacement
                +
                rendered[
                    tail_start:
                ]
            )


            if evidence_id not in used_ids:

                used_ids.append(
                    evidence_id
                )


        # Reorder used evidence by canonical evidence rank.
        rank_by_id = {
            str(
                unit.get(
                    "evidence_id"
                )
            ):
                int(
                    unit.get(
                        "evidence_rank",
                        10**9
                    )
                )

            for unit in evidence_units
        }


        used_ids.sort(
            key=lambda evidence_id:
                rank_by_id.get(
                    evidence_id,
                    10**9,
                )
        )


        references = []


        unit_by_id = {
            str(
                unit.get(
                    "evidence_id"
                )
            ):
                unit

            for unit in evidence_units
        }


        for evidence_id in used_ids:

            unit = unit_by_id[
                evidence_id
            ]


            label = label_map[
                evidence_id
            ]


            document_name = str(
                unit.get(
                    "document_name",
                    ""
                )
            ).strip()


            unit_label = str(
                unit.get(
                    "unit_label",
                    ""
                )
            ).strip()


            unit_title = str(
                unit.get(
                    "unit_title",
                    ""
                )
            ).strip()


            reference_parts = [
                part
                for part in [
                    document_name,
                    unit_label,
                    unit_title,
                ]
                if part
            ]


            references.append(
                {
                    "label":
                        label,

                    "evidence_id":
                        evidence_id,

                    "evidence_rank":
                        unit.get(
                            "evidence_rank"
                        ),

                    "document_id":
                        unit.get(
                            "document_id"
                        ),

                    "document_name":
                        document_name,

                    "document_link":
                        unit.get(
                            "document_link"
                        ),

                    "unit_label":
                        unit_label,

                    "unit_title":
                        unit_title,

                    "display":
                        " — ".join(
                            reference_parts
                        ),
                }
            )


        if references:

            lines = [
                "",
                "",
                "Nguồn chứng cứ:",
            ]


            for ref in references:

                line = (
                    f'{ref["label"]} '
                    f'{ref["display"]}'
                )


                lines.append(
                    line
                )


            rendered = (
                rendered.rstrip()
                +
                "\n"
                +
                "\n".join(
                    lines
                )
            )


        return {
            "final_answer":
                rendered,

            "used_evidence_ids":
                used_ids,

            "references":
                references,
        }
