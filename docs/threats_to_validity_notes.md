# Threats To Validity Notes

## Construct Validity

- The implementation supports six research-facing violation categories, but the benchmark definitions still determine what counts as a true violation.
- Focus-string based matching simplifies scoring but can undercount semantically equivalent annotations if labeling guidance is inconsistent.

## Internal Validity

- Provider behavior may vary across runs even with low temperature; repeated runs are required for reliable latency/cost summaries.
- The naive baseline is intentionally weak by design. Any manuscript claim should present it as a lower-bound baseline, not an exhaustive alternative.

## External Validity

- The repo currently includes only a smoke-test benchmark fixture.
- The AST parser is Python-only, so findings cannot be generalized to other implementation languages without new parser front ends.

## Conclusion Validity

- Statistical power depends on the number of annotated violations and repeated runs.
- Cost and latency claims are only trustworthy when reported alongside provider/model versions and manifest metadata.

## Operational Validity

- IDE behavior is save-triggered and full-file; “real-time” should not be overstated beyond that scope.
- Retrieval quality depends on the quality and granularity of SRS sections indexed into ChromaDB.
