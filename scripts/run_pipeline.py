#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import os
import sys
import time


REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


DEFAULT_ARTIFACT_ROOT = Path(
    os.environ.get(
        "VIETLEGAL_RAG_ARTIFACT_ROOT",
        str(
            REPO_ROOT
            / "artifacts"
        ),
    )
)


def runtime_paths(
    root,
):

    root = Path(
        root
    )


    return {
        "dense_embeddings":
            root
            / "retrieval_cache/vietnamese_embedding_v2_parent/parent_embeddings_fp32.npy",

        "dense_metadata":
            root
            / "retrieval_cache/vietnamese_embedding_v2_parent/parent_metadata.json",

        "dense_cache_meta":
            root
            / "retrieval_cache/vietnamese_embedding_v2_parent/cache_meta.json",

        "harrier_shards":
            root
            / "retrieval_cache/vietlegal_harrier_06b/chunk_shards_fp32",

        "harrier_metadata":
            root
            / "retrieval_cache/vietlegal_harrier_06b/chunk_meta",

        "bm25_chunk":
            root
            / "retrieval_cache/bm25_runtime_v1/bm25_chunk_v1.sqlite",

        "bm25_parent":
            root
            / "retrieval_cache/bm25_runtime_v1/bm25_parent_v1.sqlite",

        "corpus_db":
            root
            / "retrieval_cache/corpus_runtime_v1/corpus_chunks_v1.sqlite",

        "v3_adapter":
            root
            / (
                "results/v3_cycle/v3_reranker_training/"
                "v3_3_training_final_v2_ampfix/best_adapter"
            ),

        "generator_adapter":
            root
            / (
                "results/v3_cycle/generator_stage2/"
                "g2_8_stage2_v2/"
                "g2_8b_stage2_v2_final_adapter"
            ),

        "generator_tokenizer":
            root
            / (
                "results/v3_cycle/generator_stage2/"
                "g2_4_stage1_adapter_runtime_compat"
            ),
    }


def check_paths(
    artifact_root,
):

    paths = runtime_paths(
        artifact_root
    )


    missing = [
        (
            name,
            path,
        )
        for name, path
        in paths.items()
        if not path.exists()
    ]


    if missing:

        print(
            "Missing runtime artifacts:",
            file=sys.stderr,
        )


        for name, path in missing:

            print(
                f"- {name}: {path}",
                file=sys.stderr,
            )


        raise SystemExit(
            1
        )


    print(
        "ARTIFACT_PATH_CHECK_PASS"
    )


    return paths


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run VietLegal-RAG."
        )
    )


    parser.add_argument(
        "--question",
        type=str,
        default=None,
    )


    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
    )


    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
    )


    parser.add_argument(
        "--check-only",
        action="store_true",
    )


    parser.add_argument(
        "--rerank-batch-size",
        type=int,
        default=16,
    )


    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
    )


    args = parser.parse_args()


    artifact_root = (
        args.artifact_root
        .expanduser()
        .resolve()
    )


    p = check_paths(
        artifact_root
    )


    from vietlegal_rag.pipeline_runtime import (
        PipelineRuntime,
    )


    if args.check_only:

        print(
            "PACKAGE_IMPORT_PASS"
        )

        print(
            "PipelineRuntime version:",
            PipelineRuntime.VERSION
        )

        return


    if not args.question:

        parser.error(
            "--question required unless --check-only"
        )


    import torch


    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU required."
        )


    from vietlegal_rag.retrieval import (
        BM25Runtime,
        CandidatePoolAssembler,
        CandidatePoolConfig,
        CorpusStore,
        DenseParentIndex,
        HarrierChunkIndex,
        HybridRetrievalConfig,
        HybridRetriever,
        SentenceTransformerQueryEncoder,
        HarrierQueryEncoder,
    )

    from vietlegal_rag.reranking import (
        V3Reranker,
    )

    from vietlegal_rag.evidence.aggregator_v1 import (
        EvidenceAggregatorV1,
    )

    from vietlegal_rag.generation import (
        GeneratorRuntime,
    )

    from vietlegal_rag.validation import (
        CitationGroundingValidator,
    )

    from vietlegal_rag.rendering import (
        CitationRepairer,
        FinalAnswerRenderer,
    )


    dense_index = DenseParentIndex(
        embeddings_path=
            p[
                "dense_embeddings"
            ],

        metadata_path=
            p[
                "dense_metadata"
            ],

        cache_meta_path=
            p[
                "dense_cache_meta"
            ],
    )


    harrier_index = HarrierChunkIndex(
        shard_dir=
            p[
                "harrier_shards"
            ],

        metadata_dir=
            p[
                "harrier_metadata"
            ],
    )


    bm25 = BM25Runtime(
        chunk_db=
            p[
                "bm25_chunk"
            ],

        parent_db=
            p[
                "bm25_parent"
            ],
    )


    hybrid = HybridRetriever(
        dense_parent_index=
            dense_index,

        harrier_chunk_index=
            harrier_index,

        bm25_runtime=
            bm25,

        config=
            HybridRetrievalConfig(
                dense_parent_k=200,
                harrier_chunk_k=500,
                bm25_parent_k=200,
                bm25_chunk_k=200,
                fusion_top_k=50,
                rrf_k=60,

                weights={
                    "dense_parent":
                        1.0,

                    "harrier_parent":
                        1.0,

                    "bm25_parent":
                        1.0,

                    "bm25_chunk_parent":
                        1.0,
                },
            ),
    )


    corpus_store = CorpusStore(
        p[
            "corpus_db"
        ]
    )


    assembler = CandidatePoolAssembler(
        corpus_store=
            corpus_store,

        config=
            CandidatePoolConfig(
                fused_parent_k=50,
                local_radius=1,
                harrier_direct_k=500,
                bm25_direct_k=200,
                max_candidates=900,
            ),
    )


    dense_encoder = SentenceTransformerQueryEncoder(
        model_id=
            "AITeamVN/Vietnamese_Embedding_v2",

        revision=
            "18b44161e041bf1d3a333ab5144b5b7b93f914d2",

        device=
            "cuda",

        max_seq_length=
            2048,
    )


    harrier_encoder = HarrierQueryEncoder(
        device=
            "cuda"
    )


    reranker = V3Reranker(
        adapter_path=
            p[
                "v3_adapter"
            ],

        device=
            "cuda",
    )


    generator = GeneratorRuntime(
        adapter_path=
            p[
                "generator_adapter"
            ],

        tokenizer_path=
            p[
                "generator_tokenizer"
            ],
    )


    pipeline = PipelineRuntime(
        hybrid_retriever=
            hybrid,

        dense_query_encoder=
            dense_encoder,

        harrier_query_encoder=
            harrier_encoder,

        candidate_assembler=
            assembler,

        reranker=
            reranker,

        evidence_aggregator=
            EvidenceAggregatorV1(),

        generator=
            generator,

        validator=
            CitationGroundingValidator,

        citation_repairer=
            CitationRepairer,

        final_renderer=
            FinalAnswerRenderer,
    )


    start = time.time()


    result = pipeline.run(
        args.question,

        evidence_top_k=
            5,

        rerank_batch_size=
            args.rerank_batch_size,

        generator_max_new_tokens=
            args.max_new_tokens,
    )


    elapsed = (
        time.time()
        -
        start
    )


    print(
        "\n"
        +
        "=" * 100
    )

    print(
        "VietLegal-RAG"
    )

    print(
        "=" * 100
    )

    print(
        "Status:",
        result[
            "status"
        ]
    )

    print(
        f"Runtime: {elapsed:.2f}s"
    )


    if result.get(
        "final_answer"
    ):

        print(
            "\n"
            +
            result[
                "final_answer"
            ]
        )


    if args.output_json:

        args.output_json.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        payload = {
            "status":
                result[
                    "status"
                ],

            "question":
                result[
                    "question"
                ],

            "final_answer":
                result.get(
                    "final_answer"
                ),

            "raw_answer":
                result.get(
                    "raw_answer"
                ),

            "initial_validation":
                result.get(
                    "initial_validation"
                ),

            "citation_repair":
                result.get(
                    "citation_repair"
                ),

            "final_validation":
                result.get(
                    "final_validation"
                ),

            "timings":
                result.get(
                    "timings"
                ),
        }


        args.output_json.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


        print(
            "Saved:",
            args.output_json
        )


    try:

        corpus_store.close()

    except Exception:

        pass


    try:

        bm25.close()

    except Exception:

        pass


    if (
        result[
            "status"
        ]
        !=
        "PASS"
    ):

        raise SystemExit(
            2
        )


if __name__ == "__main__":

    main()
