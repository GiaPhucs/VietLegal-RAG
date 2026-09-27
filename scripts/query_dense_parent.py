from __future__ import annotations

import argparse
import json

from vietlegal_rag.retrieval import (
    DenseParentIndex,
    SentenceTransformerQueryEncoder,
)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Query VietLegal-RAG dense parent index."
        )
    )


    parser.add_argument(
        "--query",
        required=True,
    )

    parser.add_argument(
        "--embeddings",
        required=True,
    )

    parser.add_argument(
        "--metadata",
        required=True,
    )

    parser.add_argument(
        "--cache-meta",
        default=None,
    )

    parser.add_argument(
        "--model",
        default=(
            "AITeamVN/"
            "Vietnamese_Embedding_v2"
        ),
    )

    parser.add_argument(
        "--revision",
        default=None,
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )


    args = parser.parse_args()


    index = DenseParentIndex(
        embeddings_path=args.embeddings,
        metadata_path=args.metadata,
        cache_meta_path=args.cache_meta,
    )


    encoder = SentenceTransformerQueryEncoder(
        model_id=args.model,
        revision=args.revision,
        device=args.device,
        max_seq_length=2048,
    )


    results = index.search(
        query=args.query,
        encoder=encoder,
        top_k=args.top_k,
    )


    print(
        json.dumps(
            {
                "query":
                    args.query,

                "results":
                    results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
