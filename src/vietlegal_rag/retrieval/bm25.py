from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import re
import sqlite3
import unicodedata


TOKEN_RE = re.compile(
    r"\w+",
    flags=re.UNICODE
)


def normalize_text(text: str) -> str:
    """
    Canonical VietLegal-RAG BM25 normalization.

    - Unicode NFKC
    - lowercase
    - preserve Vietnamese diacritics
    """
    return unicodedata.normalize(
        "NFKC",
        str(text or "")
    ).lower()


def tokenize(text: str) -> List[str]:
    """
    Canonical BM25 tokenizer.
    """
    return TOKEN_RE.findall(
        normalize_text(text)
    )


def build_fts_query(text: str) -> Optional[str]:
    """
    Convert a natural-language query to an FTS5 OR query.

    Duplicate tokens are removed while preserving order.
    """
    tokens = list(
        dict.fromkeys(
            tokenize(text)
        )
    )

    if not tokens:
        return None

    return " OR ".join(
        f'"{token}"'
        for token in tokens
    )


def _readonly_connection(
    path: Path
) -> sqlite3.Connection:

    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(path)

    uri = (
        f"file:{path.as_posix()}"
        "?mode=ro"
    )

    conn = sqlite3.connect(
        uri,
        uri=True
    )

    conn.row_factory = sqlite3.Row

    return conn


class BM25ChunkRetriever:
    """
    Persistent chunk-level BM25 retriever.

    Expected DB:
        bm25_chunk_v1.sqlite
    """

    def __init__(
        self,
        db_path: str | Path,
    ):

        self.db_path = Path(
            db_path
        )

        self.conn = _readonly_connection(
            self.db_path
        )


    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Dict]:

        if top_k <= 0:
            return []

        fts_query = build_fts_query(
            query
        )

        if not fts_query:
            return []


        rows = self.conn.execute(
            """
            SELECT
                m.chunk_id,
                m.document_id,
                m.chunk_index,
                -bm25(chunk_fts) AS score
            FROM chunk_fts
            JOIN chunk_meta AS m
                ON m.rowid = chunk_fts.rowid
            WHERE chunk_fts MATCH ?
            ORDER BY bm25(chunk_fts)
            LIMIT ?
            """,
            (
                fts_query,
                int(top_k),
            )
        ).fetchall()


        return [
            {
                "rank":
                    rank,

                "chunk_id":
                    row["chunk_id"],

                "document_id":
                    row["document_id"],

                "chunk_index":
                    int(
                        row["chunk_index"]
                    ),

                "score":
                    float(
                        row["score"]
                    ),

                "retriever":
                    "bm25_chunk_v1",
            }

            for rank, row in enumerate(
                rows,
                start=1
            )
        ]


    def close(self):
        self.conn.close()


    def __enter__(self):
        return self


    def __exit__(
        self,
        exc_type,
        exc,
        tb
    ):
        self.close()


class BM25ParentRetriever:
    """
    Persistent document/parent-level BM25 retriever.

    Expected DB:
        bm25_parent_v1.sqlite
    """

    def __init__(
        self,
        db_path: str | Path,
    ):

        self.db_path = Path(
            db_path
        )

        self.conn = _readonly_connection(
            self.db_path
        )


    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Dict]:

        if top_k <= 0:
            return []

        fts_query = build_fts_query(
            query
        )

        if not fts_query:
            return []


        rows = self.conn.execute(
            """
            SELECT
                m.document_id,
                m.name,
                m.link,
                -bm25(parent_fts) AS score
            FROM parent_fts
            JOIN parent_meta AS m
                ON m.rowid = parent_fts.rowid
            WHERE parent_fts MATCH ?
            ORDER BY bm25(parent_fts)
            LIMIT ?
            """,
            (
                fts_query,
                int(top_k),
            )
        ).fetchall()


        return [
            {
                "rank":
                    rank,

                "document_id":
                    row["document_id"],

                "name":
                    row["name"],

                "link":
                    row["link"],

                "score":
                    float(
                        row["score"]
                    ),

                "retriever":
                    "bm25_parent_v1",
            }

            for rank, row in enumerate(
                rows,
                start=1
            )
        ]


    def close(self):
        self.conn.close()


    def __enter__(self):
        return self


    def __exit__(
        self,
        exc_type,
        exc,
        tb
    ):
        self.close()


class BM25Runtime:
    """
    Unified VietLegal-RAG BM25 runtime.

    Usage:

        runtime = BM25Runtime(
            chunk_db="...",
            parent_db="...",
        )

        results = runtime.search(
            "Hồ sơ quyết toán thuế...",
            chunk_k=20,
            parent_k=20,
        )
    """

    def __init__(
        self,
        chunk_db: str | Path,
        parent_db: str | Path,
    ):

        self.chunk = (
            BM25ChunkRetriever(
                chunk_db
            )
        )

        self.parent = (
            BM25ParentRetriever(
                parent_db
            )
        )


    def search(
        self,
        query: str,
        chunk_k: int = 20,
        parent_k: int = 20,
    ) -> Dict:

        query = str(
            query
        ).strip()

        if not query:
            raise ValueError(
                "query must not be empty"
            )


        return {
            "query":
                query,

            "chunk_results":
                self.chunk.search(
                    query,
                    top_k=chunk_k,
                ),

            "parent_results":
                self.parent.search(
                    query,
                    top_k=parent_k,
                ),
        }


    def close(self):
        self.chunk.close()
        self.parent.close()


    def __enter__(self):
        return self


    def __exit__(
        self,
        exc_type,
        exc,
        tb
    ):
        self.close()
