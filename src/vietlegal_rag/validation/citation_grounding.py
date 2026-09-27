from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple
import re
import unicodedata


class CitationGroundingValidator:
    """
    Canonical VietLegal-RAG Citation + Grounding Validator v1.

    Design principles
    -----------------
    1. Explicit legal instrument mismatch is a HARD citation failure.
    2. Article number alone does NOT establish citation validity.
    3. Citation fidelity and factual grounding are evaluated independently.
    4. Evidence is the source of truth.
    5. Generator output is never trusted as citation metadata.

    Example
    -------
    Evidence:
        Nghị định 49/2020/NĐ-CP

    Generated:
        Thông tư 49/2020/TT-BCA

    Result:
        citation = MISMATCH

    even if the generated substantive rule itself is supported.
    """

    # ---------------------------------------------------------------------------------------------
    # Legal instrument extraction
    # ---------------------------------------------------------------------------------------------

    NUMBERED_INSTRUMENT_RE = re.compile(
        r"\b("
        r"Nghị\s+định|"
        r"Thông\s+tư\s+liên\s+tịch|"
        r"Thông\s+tư|"
        r"Quyết\s+định|"
        r"Nghị\s+quyết|"
        r"Pháp\s+lệnh"
        r")"
        r"\s+(?:số\s+)?"
        r"(\d+[A-Za-z]?)"
        r"\s*/\s*"
        r"(\d{4})"
        r"\s*/\s*"
        r"([A-ZĐ0-9\-]+)"
        r"\b",
        flags=re.I,
    )


    NAMED_LAW_RE = re.compile(
        r"\b("
        r"Bộ\s+luật|"
        r"Luật"
        r")"
        r"\s+"
        r"([A-Za-zÀ-ỹĐđ0-9\s\-]+?)"
        r"(?:\s+năm)?"
        r"\s+"
        r"((?:19|20)\d{2})"
        r"\b",
        flags=re.I,
    )


    ARTICLE_RE = re.compile(
        r"\bĐiều\s+(\d+[A-Za-z]?)\b",
        flags=re.I,
    )


    # ---------------------------------------------------------------------------------------------
    # Text normalization
    # ---------------------------------------------------------------------------------------------

    @staticmethod
    def _fold(
        text: Any,
    ) -> str:

        text = str(
            text or ""
        )


        text = unicodedata.normalize(
            "NFKD",
            text
        )


        text = "".join(
            ch
            for ch in text
            if not unicodedata.combining(
                ch
            )
        )


        text = (
            text
            .replace(
                "Đ",
                "D"
            )
            .replace(
                "đ",
                "d"
            )
            .lower()
        )


        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()


        return text


    @classmethod
    def _canonical_type(
        cls,
        text: str,
    ) -> str:

        value = cls._fold(
            text
        )


        mapping = {
            "nghi dinh":
                "nghi_dinh",

            "thong tu":
                "thong_tu",

            "thong tu lien tich":
                "thong_tu_lien_tich",

            "quyet dinh":
                "quyet_dinh",

            "nghi quyet":
                "nghi_quyet",

            "phap lenh":
                "phap_lenh",

            "luat":
                "luat",

            "bo luat":
                "bo_luat",
        }


        return mapping.get(
            value,
            value.replace(
                " ",
                "_"
            )
        )


    @classmethod
    def _normalize_suffix(
        cls,
        value: str,
    ) -> str:

        value = cls._fold(
            value
        )


        value = re.sub(
            r"[^a-z0-9]+",
            "-",
            value
        ).strip(
            "-"
        )


        return value


    # ---------------------------------------------------------------------------------------------
    # Tokenization for grounding
    # ---------------------------------------------------------------------------------------------

    STOPWORDS = {
        "la",
        "va",
        "cua",
        "cho",
        "theo",
        "tai",
        "ve",
        "voi",
        "trong",
        "duoc",
        "cac",
        "mot",
        "nhung",
        "thi",
        "khi",
        "do",
        "nay",
        "neu",
        "tu",
        "den",
        "de",
        "co",
        "khong",
        "nguoi",
        "quy",
        "dinh",
        "nhu",
        "sau",
        "tren",
        "thuoc",
        "thuc",
        "hien",
    }


    @classmethod
    def _tokens(
        cls,
        text: str,
    ) -> List[str]:

        value = cls._fold(
            text
        )


        tokens = re.findall(
            r"[a-z0-9]+",
            value
        )


        return [
            token
            for token in tokens
            if (
                len(
                    token
                )
                > 1
                and
                token
                not in cls.STOPWORDS
            )
        ]


    @classmethod
    def _bigrams(
        cls,
        tokens: List[str],
    ) -> set:

        return {
            (
                tokens[
                    i
                ],
                tokens[
                    i + 1
                ],
            )

            for i in range(
                len(
                    tokens
                )
                -
                1
            )
        }


    # ---------------------------------------------------------------------------------------------
    # Parse numbered instruments from normal legal text
    # ---------------------------------------------------------------------------------------------

    @classmethod
    def extract_numbered_instruments(
        cls,
        text: str,
    ) -> List[Dict[str, Any]]:

        output = []


        for match in cls.NUMBERED_INSTRUMENT_RE.finditer(
            str(
                text or ""
            )
        ):

            raw = match.group(
                0
            )


            instrument_type = (
                cls._canonical_type(
                    match.group(
                        1
                    )
                )
            )


            number = (
                match.group(
                    2
                )
                .lower()
            )


            year = match.group(
                3
            )


            suffix = (
                cls._normalize_suffix(
                    match.group(
                        4
                    )
                )
            )


            output.append(
                {
                    "raw":
                        raw,

                    "type":
                        instrument_type,

                    "number":
                        number,

                    "year":
                        year,

                    "suffix":
                        suffix,

                    "canonical":
                        (
                            f"{instrument_type}|"
                            f"{number}|"
                            f"{year}|"
                            f"{suffix}"
                        ),
                }
            )


        return output


    # ---------------------------------------------------------------------------------------------
    # Parse instrument from slug-like document names
    #
    # Example:
    #   Nghi-dinh-49-2020-ND-CP-2020-huong-dan...
    #
    # becomes:
    #   Nghị định 49/2020/NĐ-CP
    # ---------------------------------------------------------------------------------------------

    @classmethod
    def extract_slug_instruments(
        cls,
        text: str,
    ) -> List[Dict[str, Any]]:

        original = str(
            text or ""
        )


        folded = cls._fold(
            original
        )


        slug = re.sub(
            r"[\s_]+",
            "-",
            folded
        )


        patterns = [
            (
                "nghi-dinh-",
                "nghi_dinh",
            ),
            (
                "thong-tu-lien-tich-",
                "thong_tu_lien_tich",
            ),
            (
                "thong-tu-",
                "thong_tu",
            ),
            (
                "quyet-dinh-",
                "quyet_dinh",
            ),
            (
                "nghi-quyet-",
                "nghi_quyet",
            ),
            (
                "phap-lenh-",
                "phap_lenh",
            ),
        ]


        output = []


        for prefix, instrument_type in patterns:

            pos = slug.find(
                prefix
            )


            if pos < 0:

                continue


            tail = slug[
                pos
                +
                len(
                    prefix
                ):
            ]


            parts = [
                part
                for part in tail.split(
                    "-"
                )
                if part
            ]


            if len(
                parts
            ) < 4:

                continue


            number = parts[
                0
            ]


            year = parts[
                1
            ]


            if not re.fullmatch(
                r"\d+[a-z]?",
                number
            ):

                continue


            if not re.fullmatch(
                r"(?:19|20)\d{2}",
                year
            ):

                continue


            suffix_parts = []


            for part in parts[
                2:
            ]:

                # Stop once the human-readable title starts.
                if re.fullmatch(
                    r"\d+",
                    part
                ):

                    break


                if (
                    len(
                        part
                    )
                    > 5
                ):

                    break


                suffix_parts.append(
                    part
                )


                if len(
                    suffix_parts
                ) >= 3:

                    break


            if not suffix_parts:

                continue


            suffix = "-".join(
                suffix_parts
            )


            output.append(
                {
                    "raw":
                        original,

                    "type":
                        instrument_type,

                    "number":
                        number,

                    "year":
                        year,

                    "suffix":
                        suffix,

                    "canonical":
                        (
                            f"{instrument_type}|"
                            f"{number}|"
                            f"{year}|"
                            f"{suffix}"
                        ),
                }
            )


        return output


    # ---------------------------------------------------------------------------------------------
    # Named law extraction
    #
    # Example:
    #   Luật Thi hành án hình sự năm 2019
    # ---------------------------------------------------------------------------------------------

    @classmethod
    def extract_named_laws(
        cls,
        text: str,
    ) -> List[Dict[str, Any]]:

        text = str(
            text or ""
        )


        # Slug → readable spacing.
        searchable = re.sub(
            r"[-_]+",
            " ",
            text
        )


        output = []


        for match in cls.NAMED_LAW_RE.finditer(
            searchable
        ):

            law_type = (
                cls._canonical_type(
                    match.group(
                        1
                    )
                )
            )


            name = cls._fold(
                match.group(
                    2
                )
            )


            name = re.sub(
                r"\s+",
                " ",
                name
            ).strip()


            # Avoid swallowing generic lead-in words.
            name = re.sub(
                r"^(ve|quy dinh ve)\s+",
                "",
                name
            )


            year = match.group(
                3
            )


            output.append(
                {
                    "raw":
                        match.group(
                            0
                        ),

                    "type":
                        law_type,

                    "name":
                        name,

                    "year":
                        year,

                    "canonical":
                        (
                            f"{law_type}|"
                            f"{name}|"
                            f"{year}"
                        ),
                }
            )


        return output


    # ---------------------------------------------------------------------------------------------
    # Evidence catalog
    # ---------------------------------------------------------------------------------------------

    @classmethod
    def build_evidence_catalog(
        cls,
        evidence_units: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:

        numbered = {}

        named_laws = {}

        article_units = defaultdict(
            list
        )


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


            source_parts = [
                unit.get(
                    "document_name",
                    ""
                ),
                unit.get(
                    "unit_label",
                    ""
                ),
                unit.get(
                    "unit_title",
                    ""
                ),
                unit.get(
                    "text",
                    ""
                ),
            ]


            source_text = "\n".join(
                str(
                    item or ""
                )
                for item in source_parts
            )


            # Normal text extraction.
            for item in cls.extract_numbered_instruments(
                source_text
            ):

                key = item[
                    "canonical"
                ]


                numbered.setdefault(
                    key,
                    {
                        **item,
                        "evidence_ids":
                            [],
                        "evidence_ranks":
                            [],
                    }
                )


                if evidence_id not in numbered[
                    key
                ][
                    "evidence_ids"
                ]:

                    numbered[
                        key
                    ][
                        "evidence_ids"
                    ].append(
                        evidence_id
                    )


                if rank not in numbered[
                    key
                ][
                    "evidence_ranks"
                ]:

                    numbered[
                        key
                    ][
                        "evidence_ranks"
                    ].append(
                        rank
                    )


            # Slug extraction.
            for item in cls.extract_slug_instruments(
                unit.get(
                    "document_name",
                    ""
                )
            ):

                key = item[
                    "canonical"
                ]


                numbered.setdefault(
                    key,
                    {
                        **item,
                        "evidence_ids":
                            [],
                        "evidence_ranks":
                            [],
                    }
                )


                if evidence_id not in numbered[
                    key
                ][
                    "evidence_ids"
                ]:

                    numbered[
                        key
                    ][
                        "evidence_ids"
                    ].append(
                        evidence_id
                    )


                if rank not in numbered[
                    key
                ][
                    "evidence_ranks"
                ]:

                    numbered[
                        key
                    ][
                        "evidence_ranks"
                    ].append(
                        rank
                    )


            # Named laws.
            for item in cls.extract_named_laws(
                source_text
            ):

                key = item[
                    "canonical"
                ]


                named_laws.setdefault(
                    key,
                    {
                        **item,
                        "evidence_ids":
                            [],
                        "evidence_ranks":
                            [],
                    }
                )


                if evidence_id not in named_laws[
                    key
                ][
                    "evidence_ids"
                ]:

                    named_laws[
                        key
                    ][
                        "evidence_ids"
                    ].append(
                        evidence_id
                    )


                if rank not in named_laws[
                    key
                ][
                    "evidence_ranks"
                ]:

                    named_laws[
                        key
                    ][
                        "evidence_ranks"
                    ].append(
                        rank
                    )


            # Article metadata.
            label = str(
                unit.get(
                    "unit_label",
                    ""
                )
            )


            article_match = cls.ARTICLE_RE.search(
                label
            )


            if article_match:

                article_units[
                    article_match.group(
                        1
                    ).lower()
                ].append(
                    {
                        "evidence_id":
                            evidence_id,

                        "evidence_rank":
                            rank,
                    }
                )


        return {
            "numbered_instruments":
                numbered,

            "named_laws":
                named_laws,

            "article_units":
                dict(
                    article_units
                ),
        }


    # ---------------------------------------------------------------------------------------------
    # Citation validation
    # ---------------------------------------------------------------------------------------------

    @classmethod
    def validate_citations(
        cls,
        answer: str,
        evidence_units: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:

        catalog = cls.build_evidence_catalog(
            evidence_units
        )


        catalog_numbered = catalog[
            "numbered_instruments"
        ]


        catalog_named = catalog[
            "named_laws"
        ]


        results = []


        # Numbered legal instruments.
        for citation in cls.extract_numbered_instruments(
            answer
        ):

            canonical = citation[
                "canonical"
            ]


            if canonical in catalog_numbered:

                matched = catalog_numbered[
                    canonical
                ]


                results.append(
                    {
                        **citation,

                        "kind":
                            "numbered_instrument",

                        "status":
                            "VALID",

                        "reason":
                            "exact_instrument_match",

                        "evidence_ids":
                            matched[
                                "evidence_ids"
                            ],

                        "evidence_ranks":
                            matched[
                                "evidence_ranks"
                            ],
                    }
                )


                continue


            # HARD mismatch:
            # same number + same year exists in evidence,
            # but instrument type and/or suffix differs.
            same_number_year = [
                item
                for item in catalog_numbered.values()
                if (
                    item[
                        "number"
                    ]
                    ==
                    citation[
                        "number"
                    ]
                    and
                    item[
                        "year"
                    ]
                    ==
                    citation[
                        "year"
                    ]
                )
            ]


            if same_number_year:

                results.append(
                    {
                        **citation,

                        "kind":
                            "numbered_instrument",

                        "status":
                            "MISMATCH",

                        "reason":
                            (
                                "same_number_year_but_"
                                "different_instrument_identity"
                            ),

                        "evidence_candidates":
                            [
                                {
                                    "type":
                                        item[
                                            "type"
                                        ],

                                    "number":
                                        item[
                                            "number"
                                        ],

                                    "year":
                                        item[
                                            "year"
                                        ],

                                    "suffix":
                                        item[
                                            "suffix"
                                        ],

                                    "evidence_ids":
                                        item[
                                            "evidence_ids"
                                        ],
                                }

                                for item in same_number_year
                            ],
                    }
                )


            else:

                results.append(
                    {
                        **citation,

                        "kind":
                            "numbered_instrument",

                        "status":
                            "UNSUPPORTED",

                        "reason":
                            "instrument_not_found_in_evidence",

                        "evidence_ids":
                            [],
                    }
                )


        # Named laws.
        for citation in cls.extract_named_laws(
            answer
        ):

            canonical = citation[
                "canonical"
            ]


            if canonical in catalog_named:

                matched = catalog_named[
                    canonical
                ]


                results.append(
                    {
                        **citation,

                        "kind":
                            "named_law",

                        "status":
                            "VALID",

                        "reason":
                            "named_law_match",

                        "evidence_ids":
                            matched[
                                "evidence_ids"
                            ],

                        "evidence_ranks":
                            matched[
                                "evidence_ranks"
                            ],
                    }
                )


            else:

                # Fuzzy named-law match:
                # same year + strong name containment.
                candidates = []


                citation_tokens = set(
                    cls._tokens(
                        citation[
                            "name"
                        ]
                    )
                )


                for item in catalog_named.values():

                    if (
                        item[
                            "year"
                        ]
                        !=
                        citation[
                            "year"
                        ]
                    ):

                        continue


                    evidence_tokens = set(
                        cls._tokens(
                            item[
                                "name"
                            ]
                        )
                    )


                    if not citation_tokens:

                        continue


                    overlap = (
                        len(
                            citation_tokens
                            &
                            evidence_tokens
                        )
                        /
                        len(
                            citation_tokens
                        )
                    )


                    if overlap >= 0.80:

                        candidates.append(
                            item
                        )


                if candidates:

                    results.append(
                        {
                            **citation,

                            "kind":
                                "named_law",

                            "status":
                                "VALID",

                            "reason":
                                "named_law_fuzzy_match",

                            "evidence_ids":
                                candidates[
                                    0
                                ][
                                    "evidence_ids"
                                ],

                            "evidence_ranks":
                                candidates[
                                    0
                                ][
                                    "evidence_ranks"
                                ],
                        }
                    )


                else:

                    results.append(
                        {
                            **citation,

                            "kind":
                                "named_law",

                            "status":
                                "UNSUPPORTED",

                            "reason":
                                "named_law_not_found_in_evidence",

                            "evidence_ids":
                                [],
                        }
                    )


        # Article mentions are reported but NEVER treated as
        # sufficient document citation evidence.
        articles = [
            {
                "raw":
                    match.group(
                        0
                    ),

                "article":
                    match.group(
                        1
                    ),

                "evidence_candidates":
                    catalog[
                        "article_units"
                    ].get(
                        match.group(
                            1
                        ).lower(),
                        [],
                    ),

                "document_validation":
                    "NOT_ESTABLISHED_BY_ARTICLE_NUMBER_ALONE",
            }

            for match in cls.ARTICLE_RE.finditer(
                answer
            )
        ]


        valid = sum(
            row[
                "status"
            ]
            ==
            "VALID"

            for row in results
        )


        mismatch = sum(
            row[
                "status"
            ]
            ==
            "MISMATCH"

            for row in results
        )


        unsupported = sum(
            row[
                "status"
            ]
            ==
            "UNSUPPORTED"

            for row in results
        )


        total = len(
            results
        )


        precision = (
            valid
            /
            total
            if total
            else 1.0
        )


        if (
            mismatch
            >
            0
        ):

            status = (
                "FAIL_MISMATCH"
            )


        elif (
            unsupported
            >
            0
        ):

            status = (
                "FAIL_UNSUPPORTED"
            )


        else:

            status = (
                "PASS"
            )


        return {
            "status":
                status,

            "explicit_citation_count":
                total,

            "valid_count":
                valid,

            "mismatch_count":
                mismatch,

            "unsupported_count":
                unsupported,

            "citation_precision":
                precision,

            "citations":
                results,

            "article_mentions":
                articles,

            "catalog":
                catalog,
        }


    # ---------------------------------------------------------------------------------------------
    # Claim segmentation
    # ---------------------------------------------------------------------------------------------

    @classmethod
    def split_claims(
        cls,
        answer: str,
    ) -> List[str]:

        answer = str(
            answer or ""
        )


        # Newlines in legal answers usually carry semantic structure.
        parts = re.split(
            r"(?<=[.!?])\s+|\n+",
            answer
        )


        output = []


        for part in parts:

            part = part.strip()


            part = re.sub(
                r"^\d+\.\s*",
                "",
                part
            ).strip()


            tokens = cls._tokens(
                part
            )


            if len(
                tokens
            ) < 3:

                continue


            output.append(
                part
            )


        return output


    @classmethod
    def _is_attribution_only(
        cls,
        claim: str,
    ) -> bool:

        folded = cls._fold(
            claim
        )


        tokens = cls._tokens(
            claim
        )


        return (
            folded.startswith(
                "can cu "
            )
            and
            len(
                tokens
            )
            <=
            35
            and
            (
                "quy dinh nhu sau"
                in folded
                or
                folded.endswith(
                    "quy dinh"
                )
                or
                folded.endswith(
                    "huong dan"
                )
            )
        )


    # ---------------------------------------------------------------------------------------------
    # Claim/evidence lexical support
    # ---------------------------------------------------------------------------------------------

    @classmethod
    def _support_score(
        cls,
        claim: str,
        evidence_text: str,
    ) -> float:

        claim_tokens = cls._tokens(
            claim
        )

        evidence_tokens = cls._tokens(
            evidence_text
        )


        if not claim_tokens:

            return 0.0


        evidence_token_set = set(
            evidence_tokens
        )


        token_recall = (
            sum(
                token in evidence_token_set
                for token in claim_tokens
            )
            /
            len(
                claim_tokens
            )
        )


        claim_bigrams = cls._bigrams(
            claim_tokens
        )


        evidence_bigrams = cls._bigrams(
            evidence_tokens
        )


        if claim_bigrams:

            bigram_recall = (
                len(
                    claim_bigrams
                    &
                    evidence_bigrams
                )
                /
                len(
                    claim_bigrams
                )
            )

        else:

            bigram_recall = (
                token_recall
            )


        score = (
            0.65
            *
            token_recall
            +
            0.35
            *
            bigram_recall
        )


        # Numeric-anchor consistency.
        claim_numbers = set(
            re.findall(
                r"\b\d+(?:[.,]\d+)?\b",
                cls._fold(
                    claim
                )
            )
        )


        evidence_numbers = set(
            re.findall(
                r"\b\d+(?:[.,]\d+)?\b",
                cls._fold(
                    evidence_text
                )
            )
        )


        if (
            claim_numbers
            and
            not claim_numbers.issubset(
                evidence_numbers
            )
        ):

            score *= 0.75


        return float(
            score
        )


    # ---------------------------------------------------------------------------------------------
    # Grounding validation
    # ---------------------------------------------------------------------------------------------

    @classmethod
    def validate_grounding(
        cls,
        answer: str,
        evidence_units: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:

        claims = cls.split_claims(
            answer
        )


        rows = []


        for index, claim in enumerate(
            claims,
            start=1,
        ):

            attribution_only = (
                cls._is_attribution_only(
                    claim
                )
            )


            claim_citation = (
                cls.validate_citations(
                    claim,
                    evidence_units,
                )
            )


            best_score = 0.0

            best_evidence = None


            for unit in evidence_units:

                evidence_text = (
                    str(
                        unit.get(
                            "unit_label",
                            ""
                        )
                    )
                    +
                    "\n"
                    +
                    str(
                        unit.get(
                            "unit_title",
                            ""
                        )
                    )
                    +
                    "\n"
                    +
                    str(
                        unit.get(
                            "text",
                            ""
                        )
                    )
                )


                score = cls._support_score(
                    claim,
                    evidence_text,
                )


                if score > best_score:

                    best_score = score

                    best_evidence = {
                        "evidence_id":
                            unit.get(
                                "evidence_id"
                            ),

                        "evidence_rank":
                            unit.get(
                                "evidence_rank"
                            ),

                        "unit_label":
                            unit.get(
                                "unit_label"
                            ),

                        "score":
                            score,
                    }


            # Explicit citation mismatch overrides lexical overlap.
            if (
                claim_citation[
                    "mismatch_count"
                ]
                >
                0
            ):

                status = (
                    "UNSUPPORTED_CITATION_MISMATCH"
                )


            elif best_score >= 0.72:

                status = (
                    "SUPPORTED"
                )


            elif best_score >= 0.45:

                status = (
                    "PARTIAL"
                )


            else:

                status = (
                    "UNSUPPORTED"
                )


            rows.append(
                {
                    "claim_id":
                        index,

                    "claim":
                        claim,

                    "attribution_only":
                        attribution_only,

                    "status":
                        status,

                    "support_score":
                        best_score,

                    "best_evidence":
                        best_evidence,

                    "citation_status":
                        claim_citation[
                            "status"
                        ],

                    "citations":
                        claim_citation[
                            "citations"
                        ],
                }
            )


        # Attribution-only statements are handled by the citation validator
        # and excluded from the factual grounding denominator.
        substantive = [
            row
            for row in rows
            if not row[
                "attribution_only"
            ]
        ]


        supported = sum(
            row[
                "status"
            ]
            ==
            "SUPPORTED"

            for row in substantive
        )


        partial = sum(
            row[
                "status"
            ]
            ==
            "PARTIAL"

            for row in substantive
        )


        unsupported = sum(
            row[
                "status"
            ]
            in {
                "UNSUPPORTED",
                "UNSUPPORTED_CITATION_MISMATCH",
            }

            for row in substantive
        )


        total = len(
            substantive
        )


        supported_rate = (
            supported
            /
            total
            if total
            else 1.0
        )


        unsupported_rate = (
            unsupported
            /
            total
            if total
            else 0.0
        )


        evidence_ids_used = {
            row[
                "best_evidence"
            ][
                "evidence_id"
            ]

            for row in substantive

            if (
                row.get(
                    "best_evidence"
                )
                and
                row[
                    "status"
                ]
                in {
                    "SUPPORTED",
                    "PARTIAL",
                }
            )
        }


        evidence_coverage = (
            len(
                evidence_ids_used
            )
            /
            len(
                evidence_units
            )
            if evidence_units
            else 0.0
        )


        if (
            supported_rate
            >=
            0.80
            and
            unsupported_rate
            <=
            0.05
        ):

            status = (
                "PASS"
            )


        elif (
            supported_rate
            >=
            0.60
        ):

            status = (
                "PARTIAL"
            )


        else:

            status = (
                "FAIL"
            )


        return {
            "status":
                status,

            "claim_count_total":
                len(
                    rows
                ),

            "substantive_claim_count":
                total,

            "attribution_claim_count":
                (
                    len(
                        rows
                    )
                    -
                    total
                ),

            "supported_claims":
                supported,

            "partial_claims":
                partial,

            "unsupported_claims":
                unsupported,

            "supported_claim_rate":
                supported_rate,

            "unsupported_claim_rate":
                unsupported_rate,

            "evidence_coverage":
                evidence_coverage,

            "claims":
                rows,
        }


    # ---------------------------------------------------------------------------------------------
    # Full validation
    # ---------------------------------------------------------------------------------------------

    @classmethod
    def validate(
        cls,
        *,
        answer: str,
        evidence_units: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:

        citations = cls.validate_citations(
            answer,
            evidence_units,
        )


        grounding = cls.validate_grounding(
            answer,
            evidence_units,
        )


        # Shipping gate:
        #
        # A grounded answer with a wrong legal citation is still not safe
        # to release as final output.
        if citations[
            "status"
        ] != "PASS":

            final_gate = (
                "REQUIRES_CITATION_REPAIR"
            )


        elif grounding[
            "status"
        ] == "FAIL":

            final_gate = (
                "REQUIRES_GROUNDING_REPAIR"
            )


        elif grounding[
            "status"
        ] == "PARTIAL":

            final_gate = (
                "REVIEW"
            )


        else:

            final_gate = (
                "PASS"
            )


        return {
            "final_gate":
                final_gate,

            "citation_validation":
                citations,

            "grounding_validation":
                grounding,
        }
