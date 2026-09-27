from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
import re

import numpy as np


SHARD_RE = re.compile(
    r"harrier_chunk_emb_(\d+)\.npy$"
)

META_RE = re.compile(
    r"harrier_chunk_meta_(\d+)\.json$"
)


def _load_meta(path: Path) -> List[Dict[str, Any]]:

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        obj = json.load(
            f
        )


    if isinstance(
        obj,
        list
    ):

        return obj


    if isinstance(
        obj,
        dict
    ):

        for key in [
            "rows",
            "items",
            "chunks",
            "metadata",
            "data",
        ]:

            value = obj.get(
                key
            )

            if isinstance(
                value,
                list
            ):

                return value


    raise ValueError(
        f"Unsupported Harrier metadata schema: {path}"
    )


class HarrierChunkIndex:
    """
    Sharded dense chunk index for VietLegal-RAG.

    Runtime design:
    - embeddings remain memory-mapped on disk
    - each shard is scanned with matrix-vector product
    - local Top-K retrieved per shard
    - global Top-K merged afterward

    This avoids loading the complete ~2 GB matrix into RAM.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        metadata_dir: str | Path,
    ):

        self.shard_dir = Path(
            shard_dir
        )

        self.metadata_dir = Path(
            metadata_dir
        )


        if not self.shard_dir.exists():

            raise FileNotFoundError(
                self.shard_dir
            )


        if not self.metadata_dir.exists():

            raise FileNotFoundError(
                self.metadata_dir
            )


        self.shard_paths = sorted(
            self.shard_dir.glob(
                "harrier_chunk_emb_*.npy"
            )
        )

        self.meta_paths = sorted(
            self.metadata_dir.glob(
                "harrier_chunk_meta_*.json"
            )
        )


        if not self.shard_paths:

            raise ValueError(
                "No Harrier embedding shards found."
            )


        if (
            len(
                self.shard_paths
            )
            !=
            len(
                self.meta_paths
            )
        ):

            raise ValueError(
                "Harrier shard count does not match metadata shard count."
            )


        self._shard_pairs = []

        self._dimension = None

        self._document_count = 0


        for shard_path, meta_path in zip(
            self.shard_paths,
            self.meta_paths,
        ):

            emb_match = SHARD_RE.search(
                shard_path.name
            )

            meta_match = META_RE.search(
                meta_path.name
            )


            if (
                not emb_match
                or
                not meta_match
                or
                emb_match.group(1)
                !=
                meta_match.group(1)
            ):

                raise ValueError(
                    "Harrier shard numbering mismatch: "
                    f"{shard_path.name} vs {meta_path.name}"
                )


            emb = np.load(
                shard_path,
                mmap_mode="r"
            )

            meta = _load_meta(
                meta_path
            )


            if emb.ndim != 2:

                raise ValueError(
                    f"Expected rank-2 embedding shard: {shard_path}"
                )


            if (
                emb.shape[0]
                !=
                len(meta)
            ):

                raise ValueError(
                    f"Embedding/meta row mismatch: {shard_path}"
                )


            if self._dimension is None:

                self._dimension = int(
                    emb.shape[1]
                )


            elif (
                emb.shape[1]
                !=
                self._dimension
            ):

                raise ValueError(
                    "Inconsistent Harrier embedding dimensions."
                )


            self._document_count += int(
                emb.shape[0]
            )


            self._shard_pairs.append(
                (
                    shard_path,
                    meta_path,
                )
            )


    @property
    def dimension(
        self,
    ) -> int:

        return int(
            self._dimension
        )


    @property
    def chunk_count(
        self,
    ) -> int:

        return int(
            self._document_count
        )


    @property
    def shard_count(
        self,
    ) -> int:

        return len(
            self._shard_pairs
        )


    @staticmethod
    def _normalize_query(
        query_embedding,
    ) -> np.ndarray:

        q = np.asarray(
            query_embedding,
            dtype=np.float32
        ).reshape(
            -1
        )


        norm = float(
            np.linalg.norm(
                q
            )
        )


        if norm <= 0:

            raise ValueError(
                "Query embedding has zero norm."
            )


        return (
            q
            /
            norm
        )


    def search_by_embedding(
        self,
        query_embedding,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search all Harrier embedding shards.

        Assumption:
            cached document embeddings are already normalized.

        Query is normalized defensively.
        """

        if top_k <= 0:
            return []


        q = self._normalize_query(
            query_embedding
        )


        if (
            q.shape[0]
            !=
            self.dimension
        ):

            raise ValueError(
                f"Query dimension {q.shape[0]} "
                f"does not match Harrier index dimension "
                f"{self.dimension}."
            )


        candidates = []


        for shard_id, (
            shard_path,
            meta_path,
        ) in enumerate(
            self._shard_pairs
        ):

            emb = np.load(
                shard_path,
                mmap_mode="r"
            )


            scores = (
                emb
                @
                q
            )


            local_k = min(
                int(
                    top_k
                ),
                int(
                    emb.shape[0]
                ),
            )


            if local_k <= 0:
                continue


            if (
                local_k
                <
                emb.shape[0]
            ):

                idx = np.argpartition(
                    scores,
                    -local_k
                )[
                    -local_k:
                ]

                idx = idx[
                    np.argsort(
                        scores[
                            idx
                        ]
                    )[
                        ::-1
                    ]
                ]


            else:

                idx = np.argsort(
                    scores
                )[
                    ::-1
                ]


            meta = _load_meta(
                meta_path
            )


            for local_idx in idx:

                local_idx = int(
                    local_idx
                )

                row = dict(
                    meta[
                        local_idx
                    ]
                )


                chunk_id = str(
                    row.get(
                        "chunk_id",
                        ""
                    )
                )

                document_id = str(
                    row.get(
                        "document_id",
                        ""
                    )
                )


                candidates.append(
                    {
                        "chunk_id":
                            chunk_id,

                        "document_id":
                            document_id,

                        "chunk_index":
                            row.get(
                                "chunk_index"
                            ),

                        "score":
                            float(
                                scores[
                                    local_idx
                                ]
                            ),

                        "shard_id":
                            shard_id,

                        "shard_row":
                            local_idx,

                        "retriever":
                            "vietlegal_harrier_chunk",
                    }
                )


        candidates.sort(
            key=lambda x:
                x[
                    "score"
                ],
            reverse=True,
        )


        output = candidates[
            :int(
                top_k
            )
        ]


        for rank, row in enumerate(
            output,
            start=1
        ):

            row[
                "rank"
            ] = rank


        return output


class HarrierQueryEncoder:
    """
    Canonical VietLegal-RAG Harrier query encoder.

    Queries MUST use the legal-retrieval instruction prefix.
    """

    MODEL_ID = "mainguyen9/vietlegal-harrier-0.6b"

    REVISION = (
        "9dc54de4a363ae03cbcec2dcf553ad592562ec85"
    )

    MAX_SEQ_LENGTH = 512

    QUERY_PROMPT = (
        "Instruct: Given a Vietnamese legal question, "
        "retrieve relevant legal passages that answer the question\n"
        "Query: "
    )


    def __init__(
        self,
        device: str | None = None,
    ):

        try:

            from sentence_transformers import (
                SentenceTransformer,
            )

        except ImportError as exc:

            raise ImportError(
                "sentence-transformers is required "
                "for HarrierQueryEncoder."
            ) from exc


        kwargs = {
            "revision":
                self.REVISION,
        }


        if device is not None:

            kwargs[
                "device"
            ] = device


        self.model = SentenceTransformer(
            self.MODEL_ID,
            **kwargs
        )

        self.model.max_seq_length = (
            self.MAX_SEQ_LENGTH
        )


    def encode(
        self,
        query: str,
    ) -> np.ndarray:

        query = str(
            query
        ).strip()


        if not query:

            raise ValueError(
                "query must not be empty"
            )


        prompted = (
            self.QUERY_PROMPT
            +
            query
        )


        vector = self.model.encode(
            [
                prompted
            ],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]


        return np.asarray(
            vector,
            dtype=np.float32
        )
