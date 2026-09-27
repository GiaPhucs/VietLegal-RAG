from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
import sqlite3


class CorpusStore:
    """
    Persistent canonical chunk store for VietLegal-RAG.

    Used by:
        - candidate pool assembly
        - reranker hydration
        - evidence aggregation
        - citation rendering

    The store is read-only at inference time.
    """

    def __init__(
        self,
        db_path: str | Path,
    ):

        self.db_path = Path(
            db_path
        ).expanduser().resolve()


        if not self.db_path.exists():

            raise FileNotFoundError(
                self.db_path
            )


        uri = (
            f"file:{self.db_path.as_posix()}"
            "?mode=ro"
        )


        self.conn = sqlite3.connect(
            uri,
            uri=True
        )

        self.conn.row_factory = (
            sqlite3.Row
        )


    @staticmethod
    def _row_to_dict(
        row,
    ) -> Dict[str, Any]:

        result = dict(
            row
        )


        metadata_json = result.get(
            "metadata_json"
        )


        if metadata_json:

            try:

                result[
                    "metadata"
                ] = json.loads(
                    metadata_json
                )

            except Exception:

                result[
                    "metadata"
                ] = {}


        result.pop(
            "metadata_json",
            None
        )


        return result


    def get_chunk(
        self,
        chunk_id: str,
    ) -> Dict[str, Any] | None:

        row = self.conn.execute(
            """
            SELECT *
            FROM chunks
            WHERE chunk_id = ?
            """,
            (
                str(
                    chunk_id
                ),
            )
        ).fetchone()


        if row is None:
            return None


        return self._row_to_dict(
            row
        )


    def get_chunks(
        self,
        chunk_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Batch hydrate chunks while preserving input order.
        """

        chunk_ids = [
            str(
                x
            )
            for x in chunk_ids
        ]


        if not chunk_ids:
            return []


        placeholders = ",".join(
            "?"
            for _ in chunk_ids
        )


        rows = self.conn.execute(
            f"""
            SELECT *
            FROM chunks
            WHERE chunk_id IN (
                {placeholders}
            )
            """,
            chunk_ids,
        ).fetchall()


        by_id = {
            str(
                row[
                    "chunk_id"
                ]
            ):
            self._row_to_dict(
                row
            )

            for row in rows
        }


        return [
            by_id[
                chunk_id
            ]

            for chunk_id in chunk_ids
            if chunk_id in by_id
        ]


    def get_document_chunks(
        self,
        document_id: str,
    ) -> List[Dict[str, Any]]:

        rows = self.conn.execute(
            """
            SELECT *
            FROM chunks
            WHERE document_id = ?
            ORDER BY chunk_index
            """,
            (
                str(
                    document_id
                ),
            )
        ).fetchall()


        return [
            self._row_to_dict(
                row
            )
            for row in rows
        ]


    def get_neighbors(
        self,
        chunk_id: str,
        radius: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Return ±radius chunks from the same parent document.
        """

        radius = max(
            0,
            int(
                radius
            )
        )


        source = self.conn.execute(
            """
            SELECT
                document_id,
                chunk_index
            FROM chunks
            WHERE chunk_id = ?
            """,
            (
                str(
                    chunk_id
                ),
            )
        ).fetchone()


        if source is None:
            return []


        document_id = str(
            source[
                "document_id"
            ]
        )

        chunk_index = int(
            source[
                "chunk_index"
            ]
        )


        rows = self.conn.execute(
            """
            SELECT *
            FROM chunks
            WHERE
                document_id = ?
                AND
                chunk_index BETWEEN ? AND ?
            ORDER BY chunk_index
            """,
            (
                document_id,
                max(
                    0,
                    chunk_index - radius
                ),
                chunk_index + radius,
            )
        ).fetchall()


        return [
            self._row_to_dict(
                row
            )
            for row in rows
        ]


    def count_chunks(
        self,
    ) -> int:

        return int(
            self.conn.execute(
                """
                SELECT COUNT(*)
                FROM chunks
                """
            ).fetchone()[0]
        )


    def count_documents(
        self,
    ) -> int:

        return int(
            self.conn.execute(
                """
                SELECT COUNT(
                    DISTINCT document_id
                )
                FROM chunks
                """
            ).fetchone()[0]
        )


    def close(
        self,
    ):

        self.conn.close()


    def __enter__(
        self,
    ):

        return self


    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ):

        self.close()
