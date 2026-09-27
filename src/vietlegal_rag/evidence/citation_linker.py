from __future__ import annotations

from collections import Counter
import math
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


TOKEN_RE = re.compile(
    r"\w+",
    flags=re.UNICODE
)


STOPWORDS = {
    "và", "là", "có", "được", "của", "theo",
    "tại", "trong", "về", "cho", "với", "các",
    "những", "một", "này", "đó", "thì", "khi",
    "do", "để", "từ", "như", "trên", "sau",
    "trước", "hoặc", "bởi", "nếu", "đối",
    "quy", "định", "pháp", "luật",
}


LEGAL_REF_RE = re.compile(
    r"\b"
    r"(nghị\s+định|nghi\s+dinh|"
    r"thông\s+tư|thong\s+tu|"
    r"quyết\s+định|quyet\s+dinh)"
    r"\s+(?:số\s+)?"
    r"(\d+)"
    r"\s*[/\-]\s*"
    r"(\d{4})",
    flags=re.I | re.UNICODE
)


SLUG_REF_RE = re.compile(
    r"\b"
    r"(nghi[\s\-_]+dinh|"
    r"thong[\s\-_]+tu|"
    r"quyet[\s\-_]+dinh)"
    r"[\s\-_]+"
    r"(\d+)"
    r"[\s\-_]+"
    r"(\d{4})",
    flags=re.I
)


ARTICLE_RE = re.compile(
    r"\bđiều\s+(\d+[a-z]?)\b",
    flags=re.I | re.UNICODE
)


def normalize(text: Any) -> str:

    value = unicodedata.normalize(
        "NFKC",
        str(text or "")
    )

    return re.sub(
        r"\s+",
        " ",
        value
    ).strip()


def strip_accents(text: str) -> str:

    value = unicodedata.normalize(
        "NFD",
        str(text)
    )

    return "".join(
        ch
        for ch in value
        if unicodedata.category(ch) != "Mn"
    ).replace(
        "đ",
        "d"
    ).replace(
        "Đ",
        "D"
    )


def meaningful_tokens(
    text: Any,
) -> List[str]:

    output = []

    for token in TOKEN_RE.findall(
        normalize(
            text
        ).lower()
    ):

        if token.isdigit():
            continue

        if len(token) <= 1:
            continue

        if token in STOPWORDS:
            continue

        output.append(
            token
        )

    return output


def extract_legal_refs(
    text: Any,
) -> Set[Tuple[str, str, str]]:

    value = normalize(
        text
    ).lower()

    refs = set()

    for match in LEGAL_REF_RE.finditer(
        value
    ):

        legal_type = strip_accents(
            match.group(1)
        ).lower()

        legal_type = re.sub(
            r"\s+",
            " ",
            legal_type
        )

        refs.add(
            (
                legal_type,
                match.group(2),
                match.group(3),
            )
        )

    return refs


def extract_slug_refs(
    text: Any,
) -> Set[Tuple[str, str, str]]:

    value = strip_accents(
        normalize(
            text
        )
    ).lower()

    refs = set()

    for match in SLUG_REF_RE.finditer(
        value
    ):

        legal_type = re.sub(
            r"[\s\-_]+",
            " ",
            match.group(1)
        )

        refs.add(
            (
                legal_type,
                match.group(2),
                match.group(3),
            )
        )

    return refs


def extract_articles(
    text: Any,
) -> Set[str]:

    return {
        x.lower()
        for x in ARTICLE_RE.findall(
            normalize(
                text
            )
        )
    }


def split_claims(
    text: str,
) -> List[str]:

    text = str(
        text or ""
    ).replace(
        "\r\n",
        "\n"
    )

    claims = []


    for line in text.split(
        "\n"
    ):

        line = line.strip()

        if not line:
            continue

        # Do NOT split on semicolons:
        # legal enumerations frequently use them.
        parts = re.split(
            r"(?<=[.!?])\s+",
            line
        )

        for part in parts:

            part = part.strip()

            if part:
                claims.append(
                    part
                )


    merged = []


    for claim in claims:

        if (
            merged
            and
            len(
                meaningful_tokens(
                    claim
                )
            ) < 4
        ):

            merged[-1] += (
                " "
                +
                claim
            )

        else:

            merged.append(
                claim
            )


    return (
        merged
        or
        [normalize(text)]
    )


class LegalAwareCitationLinker:
    """
    Deterministic claim-to-evidence linker used in V2.1 evaluation.

    This is a lexical grounding proxy.
    It is NOT a semantic entailment model.
    """

    def __init__(
        self,
        corpus_units: Iterable[Dict[str, Any]],
    ):

        units = list(
            corpus_units
        )

        self.n_docs = max(
            1,
            len(
                units
            )
        )

        self.df = Counter()


        for unit in units:

            seen = set(
                meaningful_tokens(
                    self.evidence_text(
                        unit
                    )
                )
            )

            for token in seen:

                self.df[
                    token
                ] += 1


    @staticmethod
    def evidence_text(
        unit: Dict[str, Any],
    ) -> str:

        return " ".join(
            [
                normalize(
                    unit.get(
                        "document_name",
                        ""
                    )
                ),

                normalize(
                    unit.get(
                        "citation_label",
                        ""
                    )
                ),

                normalize(
                    unit.get(
                        "merged_text",
                        ""
                    )
                ),
            ]
        ).strip()


    def idf(
        self,
        token: str,
    ) -> float:

        return (
            math.log(
                (
                    self.n_docs + 1
                )
                /
                (
                    self.df.get(
                        token,
                        0
                    )
                    +
                    1
                )
            )
            +
            1.0
        )


    def lexical_score(
        self,
        claim: str,
        evidence: str,
    ) -> Tuple[float, int]:

        claim_tokens = set(
            meaningful_tokens(
                claim
            )
        )

        evidence_tokens = set(
            meaningful_tokens(
                evidence
            )
        )


        if (
            not claim_tokens
            or
            not evidence_tokens
        ):

            return (
                0.0,
                0,
            )


        overlap = (
            claim_tokens
            &
            evidence_tokens
        )


        denom = sum(
            self.idf(token)
            for token in claim_tokens
        )


        numer = sum(
            self.idf(token)
            for token in overlap
        )


        return (
            (
                numer / denom
                if denom
                else 0.0
            ),
            len(overlap),
        )


    @staticmethod
    def unit_legal_refs(
        unit: Dict[str, Any],
    ) -> Set[Tuple[str, str, str]]:

        refs = set()

        refs |= extract_legal_refs(
            unit.get(
                "citation_label",
                ""
            )
        )

        refs |= extract_legal_refs(
            unit.get(
                "merged_text",
                ""
            )
        )

        refs |= extract_slug_refs(
            unit.get(
                "document_name",
                ""
            )
        )

        refs |= extract_slug_refs(
            unit.get(
                "citation_label",
                ""
            )
        )

        return refs


    def align(
        self,
        claim: str,
        evidence_units: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        claim_refs = extract_legal_refs(
            claim
        )

        claim_articles = extract_articles(
            claim
        )

        candidates = []


        for unit in evidence_units:

            evidence = self.evidence_text(
                unit
            )

            lexical, overlap_count = (
                self.lexical_score(
                    claim,
                    evidence
                )
            )

            unit_refs = self.unit_legal_refs(
                unit
            )

            unit_articles = extract_articles(
                (
                    unit.get(
                        "citation_label",
                        ""
                    )
                    +
                    " "
                    +
                    unit.get(
                        "merged_text",
                        ""
                    )
                )
            )


            document_match: Optional[bool] = None


            if claim_refs and unit_refs:

                document_match = bool(
                    claim_refs
                    &
                    unit_refs
                )


            if document_match is False:

                score = 0.0

            else:

                score = lexical


                if (
                    claim_articles
                    &
                    unit_articles
                ):

                    score = min(
                        1.0,
                        score + 0.05
                    )


                if document_match is True:

                    score = min(
                        1.0,
                        score + 0.10
                    )


            candidates.append(
                {
                    "score":
                        score,

                    "lexical_score":
                        lexical,

                    "overlap_count":
                        overlap_count,

                    "document_match":
                        document_match,

                    "article_match":
                        bool(
                            claim_articles
                            &
                            unit_articles
                        ),

                    "unit":
                        unit,
                }
            )


        if not candidates:

            return {
                "status":
                    "UNSUPPORTED",

                "support_score":
                    0.0,

                "evidence_id":
                    None,
            }


        candidates.sort(
            key=lambda x:
                (
                    x[
                        "score"
                    ],
                    -int(
                        x[
                            "unit"
                        ].get(
                            "best_top5_rank",
                            999
                        )
                    ),
                ),
            reverse=True
        )


        best = candidates[0]

        score = best[
            "score"
        ]

        overlap_count = best[
            "overlap_count"
        ]

        document_match = best[
            "document_match"
        ]


        if (
            document_match is True
            and
            score >= 0.45
            and
            overlap_count >= 3
        ):

            status = "SUPPORTED"


        elif (
            score >= 0.72
            and
            overlap_count >= 5
        ):

            status = "SUPPORTED"


        elif (
            document_match is True
            and
            score >= 0.28
            and
            overlap_count >= 2
        ):

            status = "PARTIAL"


        elif (
            score >= 0.52
            and
            overlap_count >= 4
        ):

            status = "PARTIAL"


        else:

            status = "UNSUPPORTED"


        linked = (
            status
            !=
            "UNSUPPORTED"
        )

        unit = best[
            "unit"
        ]


        return {
            "status":
                status,

            "support_score":
                float(
                    score
                ),

            "lexical_score":
                float(
                    best[
                        "lexical_score"
                    ]
                ),

            "overlap_count":
                int(
                    overlap_count
                ),

            "document_match":
                document_match,

            "article_match":
                bool(
                    best[
                        "article_match"
                    ]
                ),

            "evidence_id":
                (
                    unit.get(
                        "evidence_id"
                    )
                    if linked
                    else None
                ),

            "document_id":
                (
                    unit.get(
                        "document_id"
                    )
                    if linked
                    else None
                ),

            "citation_label":
                (
                    unit.get(
                        "citation_label"
                    )
                    if linked
                    else None
                ),
        }


    def align_answer(
        self,
        answer: str,
        evidence_units: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        output = []

        for index, claim in enumerate(
            split_claims(
                answer
            ),
            start=1
        ):

            result = self.align(
                claim,
                evidence_units,
            )

            output.append(
                {
                    "unit_index":
                        index,

                    "text":
                        claim,

                    **result,
                }
            )

        return output
