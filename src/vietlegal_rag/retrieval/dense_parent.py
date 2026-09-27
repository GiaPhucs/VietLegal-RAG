from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Protocol
import json

import numpy as np


class QueryEncoder(Protocol):
    """
    Minimal query encoder contract.

    Any encoder can be used as long as:
        encode(query: str) -> numpy-like vector [dimension]
    """

    def encode(
        self,
        query: str,
    ):
        ...


class DenseParentIndex:
    """
    VietLegal-RAG dense parent retrieval runtime.

    Canonical cache:
        AITeamVN/Vietnamese_Embedding_v2

    Similarity:
        cosine via inner product because cached vectors are normalized.
    """

    def __init__(
        self,
        embeddings_path: str | Path,
        metadata_path: str | Path,
        cache_meta_path: str | Path | None = None,
    ):

        self.embeddings_path = Path(
            embeddings_path
        )

        self.metadata_path = Path(
            metadata_path
        )

        self.cache_meta_path = (
            Path(
                cache_meta_path
            )
            if cache_meta_path
            else None
        )


        if not self.embeddings_path.exists():

            raise FileNotFoundError(
                self.embeddings_path
            )


        if not self.metadata_path.exists():

            raise FileNotFoundError(
                self.metadata_path
            )


        self.embeddings = np.load(
            self.embeddings_path,
            mmap_mode="r",
        )


        with self.metadata_path.open(
            "r",
            encoding="utf-8"
        ) as f:

            self.metadata = json.load(
                f
            )


        self.cache_meta: Dict[str, Any] = {}


        if (
            self.cache_meta_path
            and
            self.cache_meta_path.exists()
        ):

            with self.cache_meta_path.open(
                "r",
                encoding="utf-8"
            ) as f:

                self.cache_meta = json.load(
                    f
                )


        self._validate()


    def _validate(self):

        if self.embeddings.ndim != 2:

            raise ValueError(
                "Dense parent embeddings must be rank-2."
            )


        if (
            self.embeddings.shape[0]
            !=
            len(
                self.metadata
            )
        ):

            raise ValueError(
                "Embedding row count does not match metadata."
            )


        if self.cache_meta:

            expected_documents = self.cache_meta.get(
                "documents"
            )

            expected_dimension = self.cache_meta.get(
                "dimension"
            )


            if (
                expected_documents is not None
                and
                int(
                    expected_documents
                )
                !=
                self.embeddings.shape[0]
            ):

                raise ValueError(
                    "Cache document count mismatch."
                )


            if (
                expected_dimension is not None
                and
                int(
                    expected_dimension
                )
                !=
                self.embeddings.shape[1]
            ):

                raise ValueError(
                    "Cache embedding dimension mismatch."
                )


    @property
    def dimension(
        self,
    ) -> int:

        return int(
            self.embeddings.shape[1]
        )


    @property
    def document_count(
        self,
    ) -> int:

        return int(
            self.embeddings.shape[0]
        )


    @staticmethod
    def _normalize_vector(
        vector: np.ndarray,
    ) -> np.ndarray:

        vector = np.asarray(
            vector,
            dtype=np.float32
        ).reshape(
            -1
        )


        norm = np.linalg.norm(
            vector
        )


        if norm <= 0:

            raise ValueError(
                "Query embedding has zero norm."
            )


        return (
            vector
            /
            norm
        )


    def search_by_embedding(
        self,
        query_embedding,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Search directly with a query embedding.

        Cached document vectors are expected to be normalized.
        Query vector is normalized defensively before dot product.
        """

        if top_k <= 0:
            return []


        q = self._normalize_vector(
            query_embedding
        )


        if q.shape[0] != self.dimension:

            raise ValueError(
                f"Query dimension {q.shape[0]} "
                f"does not match index dimension {self.dimension}."
            )


        # Matrix-vector cosine similarity.
        scores = (
            self.embeddings
            @
            q
        )


        k = min(
            int(
                top_k
            ),
            self.document_count,
        )


        # Faster than sorting all 8,512 rows.
        if k < self.document_count:

            candidate_idx = np.argpartition(
                scores,
                -k
            )[
                -k:
            ]

            candidate_idx = candidate_idx[
                np.argsort(
                    scores[
                        candidate_idx
                    ]
                )[
                    ::-1
                ]
            ]

        else:

            candidate_idx = np.argsort(
                scores
            )[
                ::-1
            ]


        output = []


        for rank, idx in enumerate(
            candidate_idx,
            start=1
        ):

            idx = int(
                idx
            )

            meta = dict(
                self.metadata[
                    idx
                ]
            )


            output.append(
                {
                    "rank":
                        rank,

                    "document_id":
                        str(
                            meta.get(
                                "document_id",
                                ""
                            )
                        ),

                    "name":
                        meta.get(
                            "name",
                            ""
                        ),

                    "link":
                        meta.get(
                            "link",
                            ""
                        ),

                    "score":
                        float(
                            scores[
                                idx
                            ]
                        ),

                    "row_index":
                        idx,

                    "retriever":
                        "dense_parent_vietnamese_embedding_v2",
                }
            )


        return output


    def search(
        self,
        query: str,
        encoder: QueryEncoder,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Encode a natural-language query using an injected encoder
        and search the persistent parent embedding matrix.
        """

        query = str(
            query
        ).strip()


        if not query:

            raise ValueError(
                "query must not be empty"
            )


        query_embedding = encoder.encode(
            query
        )


        return self.search_by_embedding(
            query_embedding,
            top_k=top_k,
        )


class SentenceTransformerQueryEncoder:
    """
    Optional encoder backed by sentence-transformers.

    Model is loaded lazily.

    Example:
        encoder = SentenceTransformerQueryEncoder(
            model_id="AITeamVN/Vietnamese_Embedding_v2",
            revision="18b44161e041bf1d3a333ab5144b5b7b93f914d2",
        )
    """

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        device: str | None = None,
        max_seq_length: int | None = None,
    ):

        try:

            from sentence_transformers import (
                SentenceTransformer,
            )

        except ImportError as exc:

            raise ImportError(
                "sentence-transformers is required "
                "for SentenceTransformerQueryEncoder."
            ) from exc


        kwargs = {}


        if revision:

            kwargs[
                "revision"
            ] = revision


        if device:

            kwargs[
                "device"
            ] = device


        self.model = SentenceTransformer(
            model_id,
            **kwargs
        )


        if max_seq_length is not None:

            self.model.max_seq_length = int(
                max_seq_length
            )


    def encode(
        self,
        query: str,
    ) -> np.ndarray:

        vector = self.model.encode(
            [
                str(
                    query
                )
            ],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[
            0
        ]


        return np.asarray(
            vector,
            dtype=np.float32
        )
