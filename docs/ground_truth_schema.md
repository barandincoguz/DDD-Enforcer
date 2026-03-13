# Ground Truth Schema

## Purpose

Ground truth files define the expected DDD violations for each benchmark file and optionally provide expected traceability targets for retrieval evaluation.

## Top-Level Shape

```json
{
  "benchmark_id": "sample-ecommerce-smoke",
  "files": [
    {
      "path": "extension/backend/tests/test_violations.py",
      "expected_violations": [
        {
          "type": "SynonymViolation",
          "focus": "ClientManager",
          "source_sections": ["Glossary"]
        }
      ]
    }
  ]
}
```

## Fields

- `benchmark_id`
  - String identifier matching the manifest.
- `files`
  - Array of file-level annotations.
- `files[].path`
  - Repo-relative source file path.
- `files[].expected_violations`
  - Array of expected violations.
- `files[].expected_violations[].type`
  - One of the six research-facing violation categories.
- `files[].expected_violations[].focus`
  - Stable focus string used to align predicted and expected violations.
  - Recommended default: the primary class/function/file name referenced by the violation.
- `files[].expected_violations[].source_sections`
  - Optional list of section labels or requirement identifiers used for Top-1 / Top-3 retrieval evaluation.

## Matching Rule

- The scorer builds a key as `"{type}:{focus.lower()}"`.
- Prediction keys are derived from the violation type and the first quoted token in the violation message.
- To keep matching reliable, annotation prompts or manual guidelines should require the focus string to match the main symbol or term referenced in the expected violation.

## Recommended Annotation Guidance

- Annotate only violations that should count in the benchmark.
- Use consistent focus strings across annotators.
- If retrieval evaluation matters, record the exact expected section labels or requirement IDs.
- Store annotator notes separately from the ground-truth JSON used for scoring.
