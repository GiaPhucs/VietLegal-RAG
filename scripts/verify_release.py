#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import sys


REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


def main():

    parser = argparse.ArgumentParser(
        description="Verify VietLegal-RAG source release."
    )


    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
    )


    args = parser.parse_args()


    required_source = [
        "pyproject.toml",
        "README.md",
        "RUNTIME.md",
        "artifact_manifest.json",
        "requirements-runtime.txt",
        "src/vietlegal_rag/pipeline_runtime.py",
        "src/vietlegal_rag/retrieval/bm25.py",
        "src/vietlegal_rag/retrieval/dense_parent.py",
        "src/vietlegal_rag/retrieval/harrier_chunk.py",
        "src/vietlegal_rag/retrieval/candidate_pool.py",
        "src/vietlegal_rag/reranking/v3.py",
        "src/vietlegal_rag/evidence/aggregator_v1.py",
        "src/vietlegal_rag/generation/generator.py",
        "src/vietlegal_rag/validation/citation_grounding.py",
        "src/vietlegal_rag/rendering/final_answer.py",
        "scripts/download_artifacts.py",
        "scripts/run_pipeline.py",
    ]


    missing_source = [
        item
        for item in required_source
        if not (
            REPO_ROOT
            / item
        ).exists()
    ]


    if missing_source:

        print(
            "SOURCE_RELEASE_FAIL"
        )


        for item in missing_source:

            print(
                "MISSING:",
                item
            )


        raise SystemExit(
            1
        )


    manifest = json.loads(
        (
            REPO_ROOT
            / "artifact_manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )


    assert manifest[
        "files"
    ]


    assert all(
        entry.get(
            "sha256"
        )
        for entry in manifest[
            "files"
        ]
    )


    print(
        "SOURCE_RELEASE_PASS"
    )

    print(
        "Manifest files:",
        manifest[
            "file_count"
        ]
    )


    if args.artifact_root is not None:

        root = (
            args.artifact_root
            .expanduser()
            .resolve()
        )


        failures = []


        for entry in manifest[
            "files"
        ]:

            path = (
                root
                / entry[
                    "path"
                ]
            )


            if not path.exists():

                failures.append(
                    (
                        entry[
                            "path"
                        ],
                        "missing",
                    )
                )

                continue


            if (
                path.stat().st_size
                !=
                int(
                    entry[
                        "size_bytes"
                    ]
                )
            ):

                failures.append(
                    (
                        entry[
                            "path"
                        ],
                        "size mismatch",
                    )
                )


        if failures:

            print(
                "ARTIFACT_LAYOUT_FAIL"
            )


            for path, reason in failures[
                :50
            ]:

                print(
                    path,
                    reason
                )


            raise SystemExit(
                2
            )


        print(
            "ARTIFACT_LAYOUT_PASS"
        )


if __name__ == "__main__":

    main()
