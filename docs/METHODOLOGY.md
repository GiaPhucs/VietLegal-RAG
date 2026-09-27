# Evaluation Methodology

## Evaluation split

- V2 Dev600
- 600 fixed questions
- question-only inference

## Retrieval metrics

- Exact Recall@5/@20/@50
- Parent Recall@5/@20/@50
- Any Exact@5
- Any Parent@5
- MRR@20

## Citation metrics

Because the dataset has no official gold citation annotations, citation quality is evaluated against answer-aware Silver Oracle evidence.

### Exact citation

A predicted citation is considered an exact Silver hit when the cited Evidence Unit contains at least one Silver Oracle chunk.

### Parent citation

A predicted citation is considered a parent hit when its `document_id` matches a Silver Oracle document.

## Grounding proxy

The V2.1 grounding audit uses deterministic legal-aware lexical alignment.

Signals include:

- IDF-weighted lexical overlap;
- explicit legal-document matching;
- article-reference overlap;
- legal-document mismatch rejection.

The resulting SUPPORTED / PARTIAL / UNSUPPORTED labels are diagnostic proxies and not semantic entailment ground truth.

## Historical metrics

METEOR and ROUGE-L are preserved only as historical competition-era generation metrics.