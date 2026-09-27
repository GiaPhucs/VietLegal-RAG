from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class V3Reranker:
    """
    Canonical VietLegal-RAG V3 chunk reranker.

    Base model:
        AITeamVN/Vietnamese_Reranker

    Revision:
        f536976248403314225d7fdfdbc87f0e9516a54e

    Adapter:
        canonical V3 LoRA best_adapter

    Input:
        question + canonical chunk text

    max_length:
        2304

    Inference:
        CUDA FP16 autocast

    Score:
        raw scalar sequence-classification logit

    Ranking:
        descending raw logit using stable Python sorting

    IMPORTANT:
        Chunk-level only.
        Whole documents/articles must never be passed to V3.
    """

    BASE_MODEL = (
        "AITeamVN/Vietnamese_Reranker"
    )

    BASE_REVISION = (
        "f536976248403314225d7fdfdbc87f0e9516a54e"
    )

    MAX_LENGTH = 2304


    def __init__(
        self,
        adapter_path: str | Path,
        device: str = "cuda",
    ):

        import torch

        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )

        from peft import (
            PeftModel,
        )


        self.torch = torch

        self.device = torch.device(
            device
        )


        if (
            self.device.type
            ==
            "cuda"
            and
            not torch.cuda.is_available()
        ):

            raise RuntimeError(
                "CUDA requested but unavailable."
            )


        self.adapter_path = Path(
            adapter_path
        ).expanduser().resolve()


        if not self.adapter_path.exists():

            raise FileNotFoundError(
                self.adapter_path
            )


        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                self.BASE_MODEL,
                revision=self.BASE_REVISION,
                trust_remote_code=True,
                use_fast=True,
            )
        )


        base_model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                self.BASE_MODEL,
                revision=self.BASE_REVISION,
                trust_remote_code=True,
            )
        )


        self.model = (
            PeftModel
            .from_pretrained(
                base_model,
                str(
                    self.adapter_path
                ),
            )
        )


        self.model = (
            self.model
            .to(
                self.device
            )
        )

        self.model.eval()


    def rerank(
        self,
        question: str,
        candidates: List[
            Dict[str, Any]
        ],
        batch_size: int = 24,
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:

        torch = self.torch


        question = str(
            question
        ).strip()


        if not question:

            raise ValueError(
                "question must not be empty"
            )


        if not candidates:

            return []


        batch_size = max(
            1,
            int(
                batch_size
            )
        )


        scores = []


        for start in range(
            0,
            len(
                candidates
            ),
            batch_size,
        ):

            batch = candidates[
                start:
                start
                +
                batch_size
            ]


            passages = []


            for row in batch:

                if not row.get(
                    "chunk_id"
                ):

                    raise ValueError(
                        "V3 candidate missing chunk_id."
                    )


                # Protect frozen architecture.
                if isinstance(
                    row.get(
                        "chunks"
                    ),
                    list
                ):

                    raise ValueError(
                        "V3 reranker is chunk-only; "
                        "whole-document inputs are forbidden."
                    )


                passage = (
                    row.get(
                        "text"
                    )
                    or
                    row.get(
                        "retrieval_text"
                    )
                    or
                    ""
                )


                passage = str(
                    passage
                )


                if not passage.strip():

                    raise ValueError(
                        "V3 candidate contains empty text: "
                        f'{row["chunk_id"]}'
                    )


                passages.append(
                    passage
                )


            questions = [
                question
            ] * len(
                passages
            )


            encoded = self.tokenizer(
                questions,
                passages,

                padding=True,
                truncation=True,

                max_length=
                    self.MAX_LENGTH,

                return_tensors=
                    "pt",
            )


            encoded = {
                key:
                    value.to(
                        self.device,
                        non_blocking=True,
                    )

                for key, value
                in encoded.items()
            }


            with torch.inference_mode():

                if (
                    self.device.type
                    ==
                    "cuda"
                ):

                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                    ):

                        logits = (
                            self.model(
                                **encoded
                            )
                            .logits
                            .float()
                            .view(
                                -1
                            )
                        )

                else:

                    logits = (
                        self.model(
                            **encoded
                        )
                        .logits
                        .float()
                        .view(
                            -1
                        )
                    )


            values = (
                logits
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )


            if not np.isfinite(
                values
            ).all():

                raise FloatingPointError(
                    "Non-finite V3 reranker scores."
                )


            scores.extend(
                values.tolist()
            )


        if (
            len(
                scores
            )
            !=
            len(
                candidates
            )
        ):

            raise RuntimeError(
                "V3 score count mismatch."
            )


        output = []


        for candidate, score in zip(
            candidates,
            scores,
        ):

            row = dict(
                candidate
            )

            row[
                "v3_score"
            ] = float(
                score
            )

            output.append(
                row
            )


        # IMPORTANT:
        # Python sort is stable.
        # Equal V3 scores preserve candidate-pool input order,
        # matching the historical V3 implementation.
        output.sort(
            key=lambda row:
                row[
                    "v3_score"
                ],
            reverse=True,
        )


        for rank, row in enumerate(
            output,
            start=1
        ):

            row[
                "v3_rank"
            ] = rank


        if top_k is not None:

            output = output[
                :max(
                    0,
                    int(
                        top_k
                    )
                )
            ]


        return output
