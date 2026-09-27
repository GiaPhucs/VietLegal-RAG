"""
VietLegal-RAG v2.1

Public portfolio reference implementation.

The release contains verified post-retrieval components and
dependency-injected contracts for retrieval, reranking and generation.
"""

__version__ = "2.1.0"

from .pipeline import VietLegalRAG

__all__ = [
    "VietLegalRAG",
]

from .pipeline_runtime import PipelineRuntime
