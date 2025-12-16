"""
RAG Pipeline Benchmark Script

Measures retrieval time, relevance scores, and section accuracy.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add parent directories to path for imports
script_dir = Path(__file__).parent
rag_benchmarking_dir = script_dir.parent
core_dir = rag_benchmarking_dir.parent
backend_dir = core_dir.parent

sys.path.insert(0, str(core_dir))  # For rag_pipeline
sys.path.insert(0, str(backend_dir))

from rag_pipeline import RAGPipeline


def load_test_cases(test_cases_file: str) -> List[Dict[str, Any]]:
    """Load test cases from JSON file."""
    test_cases_path = Path(test_cases_file)

    if not test_cases_path.exists():
        raise FileNotFoundError(f"Test cases file not found: {test_cases_path}")

    with open(test_cases_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_cases = data.get("test_cases", [])
    print(f"Loaded {len(test_cases)} test cases")

    return test_cases


def setup_pipeline(document_path: str) -> tuple[RAGPipeline, int, str]:
    """Initialize pipeline and index document."""
    # Use temp directory for benchmark
    persist_dir = str(Path(__file__).parent.parent / "chromaDB")

    pipeline = RAGPipeline(persist_directory=persist_dir)

    # Load and index the SRS document
    srs_path = Path(document_path)

    if not srs_path.exists():
        raise FileNotFoundError(f"SRS document not found: {srs_path}")

    raw_text = srs_path.read_text(encoding="utf-8")

    # Extract doc_id and doc_name from filename
    document_name = srs_path.name
    doc_id = document_name.replace(".txt", "")

    chunk_count = pipeline.index_document(
        raw_text=raw_text,
        doc_id=doc_id,
        doc_name=document_name,
        doc_type="srs"
    )

    return pipeline, chunk_count, document_name


def run_benchmark(pipeline: RAGPipeline, test_cases: List[Dict[str, Any]], iterations: int) -> List[Dict[str, Any]]:
    """Run benchmark on all test cases."""
    results = []

    for test in test_cases:
        timings = []
        sources = None

        for _ in range(iterations):
            start = time.perf_counter()
            sources = pipeline.retrieve_source(
                test["violation_type"],
                test["message"]
            )
            end = time.perf_counter()
            timings.append((end - start) * 1000)  # ms

        # Calculate average time
        avg_time = statistics.mean(timings) if iterations > 1 else timings[0]

        # Get top-1 result metrics
        top1_relevance = sources[0]["relevance_score"] if sources else 0
        top1_section = sources[0]["section"] if sources else ""
        top1_correct = check_section_match(top1_section, test["expected_section"])

        # Check if expected section is in top-3
        top3_sections = [s["section"] for s in sources[:3]] if sources else []
        top3_correct = any(
            check_section_match(s, test["expected_section"])
            for s in top3_sections
        )

        results.append({
            "test_id": test["id"],
            "violation_type": test["violation_type"],
            "expected_section": test["expected_section"],
            "retrieved_section": top1_section,
            "avg_time_ms": round(avg_time, 2),
            "relevance_score": round(top1_relevance, 3),
            "top1_correct": top1_correct,
            "top3_correct": top3_correct,
        })

    return results


def check_section_match(retrieved: str, expected: str) -> bool:
    """Check if retrieved section matches expected section."""
    if not retrieved:
        return False

    # Extract section number from retrieved section name
    # e.g., "3.1 Customer Management Context" -> "3.1"
    parts = retrieved.split()
    if parts:
        section_num = parts[0].rstrip(".")
        # Exact match or starts with expected
        return section_num == expected or section_num.startswith(expected + ".")
    return False


def calculate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary metrics across all test cases."""
    times = [r["avg_time_ms"] for r in results]
    relevances = [r["relevance_score"] for r in results]
    top1_correct = sum(1 for r in results if r["top1_correct"])
    top3_correct = sum(1 for r in results if r["top3_correct"])
    total = len(results)

    return {
        "avg_time_ms": round(statistics.mean(times), 2),
        "avg_relevance": round(statistics.mean(relevances), 3),
        "top1_accuracy_pct": round(top1_correct / total * 100, 1),
        "top3_accuracy_pct": round(top3_correct / total * 100, 1),
        "top1_correct": top1_correct,
        "top3_correct": top3_correct,
        "total": total,
    }


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Benchmark RAG pipeline with custom test cases"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the SRS document file (e.g., ecommerce_srs.txt)",
    )
    parser.add_argument(
        "--testcases",
        type=str,
        required=True,
        help="Path to test cases JSON file",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations per test case (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON file (optional)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RAG Pipeline Benchmark")
    print("=" * 60)

    # Load test cases
    print("\nLoading test cases...")
    test_cases = load_test_cases(args.testcases)

    # Setup pipeline
    print(f"Indexing document: {Path(args.data).name}")
    pipeline, chunk_count, document_name = setup_pipeline(args.data)
    print(f"Indexed {chunk_count} chunks\n")

    # Run benchmark
    print(f"Running benchmark: {len(test_cases)} tests x {args.iterations} iterations")
    results = run_benchmark(pipeline, test_cases, args.iterations)

    # Calculate summary
    summary = calculate_summary(results)

    # Print summary
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Average retrieval time: {summary['avg_time_ms']:.2f} ms")
    print(f"Average relevance score: {summary['avg_relevance']:.3f}")
    print(f"\nAccuracy:")
    print(f"  Top-1: {summary['top1_accuracy_pct']}% ({summary['top1_correct']}/{summary['total']})")
    print(f"  Top-3: {summary['top3_accuracy_pct']}% ({summary['top3_correct']}/{summary['total']})")
    print("=" * 60)

    # Save results if output path provided
    if args.output:
        output = {
            "document": document_name,
            "chunks": chunk_count,
            "iterations": args.iterations,
            "summary": summary,
            "results": results,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
