from __future__ import annotations

import argparse
import json

import numpy as np

from vietlegal_rag.retrieval import (
    HarrierChunkIndex,
)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Query VietLegal-RAG Harrier chunk index "
            "using a precomputed query embedding."
        )
    )

    parser.add_argument(
        "--query-embedding",
        required=True,
        help=(
            "Path to .npy vector. "
            "Accepted shapes: [D] or [1,D]."
        ),
    )

    parser.add_argument(
        "--shard-dir",
        required=True,
    )

    parser.add_argument(
        "--metadata-dir",
        required=True,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )


    args = parser.parse_args()


    q = np.load(
        args.query_embedding
    )


    if (
        q.ndim == 2
        and
        q.shape[0] == 1
    ):

        q = q[0]


    index = HarrierChunkIndex(
        shard_dir=args.shard_dir,
        metadata_dir=args.metadata_dir,
    )


    results = index.search_by_embedding(
        q,
        top_k=args.top_k,
    )


    print(
        json.dumps(
            {
                "chunks":
                    index.chunk_count,

                "dimension":
                    index.dimension,

                "shards":
                    index.shard_count,

                "results":
                    results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
