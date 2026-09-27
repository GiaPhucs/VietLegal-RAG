# VietLegal-RAG Runtime

Canonical runtime:

```text
question
  -> hybrid retrieval
  -> weighted RRF
  -> candidate pool
  -> V3 reranker
  -> structured evidence
  -> Stage2-v2 generator
  -> citation/grounding validator
  -> deterministic repair
  -> post-repair validation
  -> trusted final answer
```

## Verified environment

```text
torch=2.11.0+cu128
transformers=5.16.1
peft=0.20.0
bitsandbytes=0.50.1
sentence-transformers=5.7.0
```

## Install

```bash
pip install -r requirements-runtime.txt
pip install -e . --no-deps
```

## Deploy artifacts from an existing runtime

```bash
python scripts/download_artifacts.py --source-root /path/to/Task2 --artifact-root ./artifacts
```

## Verify

```bash
python scripts/run_pipeline.py --check-only
```

## Run

```bash
python scripts/run_pipeline.py --question "Thời điểm nào sẽ thông báo phạm nhân hết hạn chấp hành án phạt tù?"
```

Canonical runtime requires CUDA.

Generated legal citations are not trusted automatically.
The final answer is returned only when the validation shipping gate reaches PASS.
