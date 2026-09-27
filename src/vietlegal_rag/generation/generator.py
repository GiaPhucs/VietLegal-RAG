from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import hashlib
import time


class GeneratorRuntime:
    """
    Canonical VietLegal-RAG generator runtime v1.

    Architecture
    ------------
    Qwen/Qwen3.5-2B
        +
    Stage2-v2 FINAL LoRA adapter

    Runtime
    -------
    - 4-bit NF4
    - float16 compute
    - greedy decoding
    - thinking disabled
    - max input tokens = 3072

    IMPORTANT
    ---------
    Generated legal citations MUST NOT be treated as verified.

    The generator may produce a semantically correct answer while
    hallucinating or altering the legal instrument name.

    Therefore downstream Citation/Grounding Validator is mandatory.
    """

    BASE_MODEL = (
        "Qwen/Qwen3.5-2B"
    )

    MAX_INPUT_TOKENS = 3072

    DEFAULT_MAX_NEW_TOKENS = 512

    PROMPT_VERSION = (
        "vietlegal_generator_prompt_v1"
    )

    SYSTEM_PROMPT = (
        "Bạn là trợ lý trả lời câu hỏi pháp luật Việt Nam. "
        "Hãy trả lời dựa trên các tài liệu được cung cấp. "
        "Ưu tiên thông tin có căn cứ trong tài liệu và không tự bịa thêm "
        "quy định pháp luật không có trong ngữ cảnh."
    )


    def __init__(
        self,
        *,
        adapter_path: str | Path,
        tokenizer_path: str | Path,
        device_map: str = "auto",
    ):

        import torch

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        from peft import (
            PeftModel,
        )


        if not torch.cuda.is_available():

            raise RuntimeError(
                "GeneratorRuntime v1 requires CUDA."
            )


        self.torch = torch

        self.adapter_path = Path(
            adapter_path
        ).expanduser().resolve()

        self.tokenizer_path = Path(
            tokenizer_path
        ).expanduser().resolve()


        if not self.adapter_path.exists():

            raise FileNotFoundError(
                self.adapter_path
            )


        if not self.tokenizer_path.exists():

            raise FileNotFoundError(
                self.tokenizer_path
            )


        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                str(
                    self.tokenizer_path
                ),
                trust_remote_code=True,
            )
        )


        if (
            self.tokenizer.eos_token_id
            !=
            248046
        ):

            raise RuntimeError(
                "Unexpected EOS token id: "
                f"{self.tokenizer.eos_token_id}"
            )


        if (
            self.tokenizer.pad_token_id
            !=
            248044
        ):

            raise RuntimeError(
                "Unexpected PAD token id: "
                f"{self.tokenizer.pad_token_id}"
            )


        quant_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        )


        base_model = (
            AutoModelForCausalLM
            .from_pretrained(
                self.BASE_MODEL,
                quantization_config=quant_config,
                device_map=device_map,
                trust_remote_code=True,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        )


        # Stage2-v2 FINAL is cumulative from Stage1 training.
        # DO NOT stack Stage1 again.
        self.model = (
            PeftModel
            .from_pretrained(
                base_model,
                str(
                    self.adapter_path
                ),
            )
        )


        self.model.eval()


    # =============================================================================================
    # PROMPT
    # =============================================================================================

    def build_prompt(
        self,
        *,
        question: str,
        evidence: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:

        question = str(
            question
        ).strip()


        if not question:

            raise ValueError(
                "question must not be empty"
            )


        if not evidence:

            raise ValueError(
                "evidence must not be empty"
            )


        document_blocks = []


        for i, ev in enumerate(
            evidence,
            start=1,
        ):

            text = str(
                ev.get(
                    "text",
                    ""
                )
            ).strip()


            if not text:

                raise ValueError(
                    f"Evidence {i} has empty text."
                )


            heading = (
                ev.get(
                    "heading"
                )
                or
                ev.get(
                    "unit_label"
                )
                or
                f"Tài liệu {i}"
            )


            document_blocks.append(
                (
                    f"[{i}] {heading}\n"
                    f"{text}"
                )
            )


        evidence_text = "\n\n".join(
            document_blocks
        )


        user_prompt = (
            "Dựa trên các tài liệu pháp lý dưới đây, hãy trả lời câu hỏi. "
            "Chỉ sử dụng thông tin có trong tài liệu; nếu các tài liệu có phạm vi "
            "áp dụng khác nhau thì cần phân biệt rõ.\n\n"
            f"Câu hỏi: {question}\n\n"
            f"{evidence_text}"
        )


        messages = [
            {
                "role":
                    "system",

                "content":
                    self.SYSTEM_PROMPT,
            },
            {
                "role":
                    "user",

                "content":
                    user_prompt,
            },
        ]


        rendered = (
            self.tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )


        input_ids = (
            self.tokenizer(
                rendered,
                add_special_tokens=False,
            )[
                "input_ids"
            ]
        )


        token_count = len(
            input_ids
        )


        if (
            token_count
            >
            self.MAX_INPUT_TOKENS
        ):

            raise ValueError(
                "Generator prompt exceeds "
                f"{self.MAX_INPUT_TOKENS} tokens: "
                f"{token_count}"
            )


        return {
            "prompt_version":
                self.PROMPT_VERSION,

            "rendered_prompt":
                rendered,

            "input_tokens":
                token_count,

            "prompt_sha256":
                hashlib.sha256(
                    rendered.encode(
                        "utf-8"
                    )
                ).hexdigest(),
        }


    # =============================================================================================
    # GENERATE
    # =============================================================================================

    def generate(
        self,
        *,
        question: str,
        evidence: List[
            Dict[str, Any]
        ],
        max_new_tokens: int | None = None,
    ) -> Dict[str, Any]:

        torch = self.torch


        prompt_info = self.build_prompt(
            question=question,
            evidence=evidence,
        )


        rendered = prompt_info[
            "rendered_prompt"
        ]


        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        )


        device = next(
            self.model.parameters()
        ).device


        inputs = {
            key:
                value.to(
                    device
                )

            for key, value
            in inputs.items()
        }


        max_new_tokens = (
            self.DEFAULT_MAX_NEW_TOKENS
            if max_new_tokens is None
            else int(
                max_new_tokens
            )
        )


        if max_new_tokens <= 0:

            raise ValueError(
                "max_new_tokens must be > 0"
            )


        torch.cuda.reset_peak_memory_stats()


        start = time.time()


        with torch.inference_mode():

            generated = (
                self.model.generate(
                    **inputs,

                    max_new_tokens=
                        max_new_tokens,

                    do_sample=
                        False,

                    num_beams=
                        1,

                    eos_token_id=
                        self.tokenizer.eos_token_id,

                    pad_token_id=
                        self.tokenizer.pad_token_id,

                    use_cache=
                        True,
                )
            )


        generation_seconds = (
            time.time()
            -
            start
        )


        input_len = int(
            inputs[
                "input_ids"
            ].shape[
                1
            ]
        )


        new_tokens = generated[
            0,
            input_len:
        ]


        output_tokens = int(
            new_tokens.shape[
                0
            ]
        )


        answer = (
            self.tokenizer
            .decode(
                new_tokens,
                skip_special_tokens=True,
            )
            .replace(
                "<think>",
                ""
            )
            .replace(
                "</think>",
                ""
            )
            .strip()
        )


        if not answer:

            raise RuntimeError(
                "Generator returned empty answer."
            )


        peak_vram_gb = (
            torch.cuda.max_memory_allocated()
            /
            1024**3
        )


        return {
            "answer":
                answer,

            "prompt_version":
                prompt_info[
                    "prompt_version"
                ],

            "prompt_sha256":
                prompt_info[
                    "prompt_sha256"
                ],

            "input_tokens":
                prompt_info[
                    "input_tokens"
                ],

            "output_tokens":
                output_tokens,

            "generation_seconds":
                generation_seconds,

            "peak_vram_gb":
                peak_vram_gb,

            "runtime": {
                "base_model":
                    self.BASE_MODEL,

                "adapter":
                    str(
                        self.adapter_path
                    ),

                "quantization":
                    "4-bit NF4",

                "compute_dtype":
                    "float16",

                "do_sample":
                    False,

                "num_beams":
                    1,

                "thinking":
                    False,

                "max_input_tokens":
                    self.MAX_INPUT_TOKENS,

                "max_new_tokens":
                    max_new_tokens,
            },

            # IMPORTANT:
            # downstream validator must decide this.
            "citation_validated":
                False,

            "grounding_validated":
                False,
        }
