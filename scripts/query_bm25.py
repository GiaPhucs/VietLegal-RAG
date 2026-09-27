from __future__ import annotations

import argparse
import json

from vietlegal_rag.retrieval import (
    BM25Runtime,
)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Query VietLegal-RAG "
            "persistent BM25 runtime"
        )
    )

    parser.add_argument(
        "--query",
        required=True,
    )

    parser.add_argument(
        "--chunk-db",
        required=True,
    )

    parser.add_argument(
        "--parent-db",
        required=True,
    )

    parser.add_argument(
        "--chunk-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--parent-k",
        type=int,
        default=5,
    )


    args = parser.parse_args()


    with BM25Runtime(
        chunk_db=args.chunk_db,
        parent_db=args.parent_db,
    ) as runtime:

        result = runtime.search(
            args.query,
            chunk_k=args.chunk_k,
            parent_k=args.parent_k,
        )


    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )
    )


if __name__ == "__main__":
    main()
