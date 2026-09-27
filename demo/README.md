# Demo Interface Contract

The final demo should expose the full evidence-grounded QA flow rather than only displaying a generated answer.

## Input

```json
{
  "question": "Câu hỏi pháp luật tiếng Việt"
}
```

## Recommended output

```json
{
  "answer": "...",
  "citations": [
    {
      "document_id": "...",
      "citation_label": "...",
      "evidence_id": "...",
      "quote": "..."
    }
  ],
  "grounding": {
    "status": "SUPPORTED | PARTIAL | UNSUPPORTED"
  }
}
```

## UI recommendation

Display:

1. user question;
2. concise answer;
3. expandable legal evidence;
4. document/provision citation;
5. grounding status;
6. optional retrieval diagnostics for portfolio/demo mode.