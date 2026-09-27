from vietlegal_rag import VietLegalRAG
from vietlegal_rag.evidence import (
    LegalAwareCitationLinker,
    aggregate_top_chunks,
)


chunks = [
    {
        "chunk_id":
            "100:00001",

        "document_id":
            100,

        "name":
            "Nghi-dinh-12-2023-ND-CP",

        "article":
            "Điều 1",

        "article_title":
            "Phạm vi điều chỉnh",

        "text":
            "Điều 1. Phạm vi điều chỉnh của Nghị định này.",

        "reranker_score":
            0.95,

        "rank":
            1,
    },

    {
        "chunk_id":
            "100:00002",

        "document_id":
            100,

        "name":
            "Nghi-dinh-12-2023-ND-CP",

        "article":
            "Điều 1",

        "article_title":
            "Phạm vi điều chỉnh",

        "text":
            "Nghị định 12/2023/NĐ-CP quy định phạm vi điều chỉnh.",

        "reranker_score":
            0.90,

        "rank":
            2,
    },
]


units = aggregate_top_chunks(
    chunks
)


assert len(
    units
) == 1


assert len(
    units[0][
        "chunks"
    ]
) == 2


linker = LegalAwareCitationLinker(
    units
)


result = linker.align(
    (
        "Căn cứ Điều 1 "
        "Nghị định 12/2023/NĐ-CP."
    ),
    units,
)


assert result[
    "status"
] in {
    "SUPPORTED",
    "PARTIAL",
}


print(
    "✅ evidence aggregator PASS"
)

print(
    "✅ legal citation linker PASS"
)

print(
    "✅ public package smoke test PASS"
)
