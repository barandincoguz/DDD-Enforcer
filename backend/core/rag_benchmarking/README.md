# RAG Pipeline Benchmarking

A flexible benchmarking system for testing the RAG (Retrieval-Augmented Generation) pipeline across multiple domains with comprehensive test cases.

## Quick Start

Run the benchmark with default test cases (E-Commerce domain):

```bash
python script/benchmark_rag.py
```

Test different domains:

```bash
# Healthcare domain
python script/benchmark_rag.py --test-cases data/healthcare_test_cases.json

# Banking domain
python script/benchmark_rag.py --test-cases data/banking_test_cases.json
```

## Available Domains

### 1. E-Commerce Platform (ShopEase)
- **Document**: `ecommerce_srs.txt`
- **Test Cases**: `ecommerce_test_cases.json` (50 tests)
- **Contexts**: Customer Management, Order Processing, Inventory Management, Payment Processing
- **Key Terms**: Customer, Order, Product, Payment

### 2. Healthcare System (MediCare Pro)
- **Document**: `healthcare_srs.txt`
- **Test Cases**: `healthcare_test_cases.json` (50 tests)
- **Contexts**: Patient Management, Appointment Scheduling, Treatment Planning, Prescription Management
- **Key Terms**: Patient, Appointment, Treatment, Prescription

### 3. Banking Platform (SecureBank)
- **Document**: `banking_srs.txt`
- **Test Cases**: `banking_test_cases.json` (50 tests)
- **Contexts**: Account Management, Transaction Processing, Loan Management, Card Services
- **Key Terms**: Account, Transaction, Loan, Card

## Usage

### Basic Usage

```bash
# E-Commerce domain (default)
python script/benchmark_rag.py

# Healthcare domain
python script/benchmark_rag.py --test-cases data/healthcare_test_cases.json

# Banking domain
python script/benchmark_rag.py --test-cases data/banking_test_cases.json

# Custom output path
python script/benchmark_rag.py --test-cases data/healthcare_test_cases.json --output results/healthcare_custom.json
```

### Command-Line Arguments

- `--test-cases PATH`: Path to test cases JSON file (default: `data/ecommerce_test_cases.json`)
- `--output PATH`: Path to save results JSON file (default: auto-generated based on domain)

## Test Case Structure

Each domain has a comprehensive test suite with 50 test cases covering:

1. **NamingViolations** (16 tests)
   - Testing synonym detection for core domain terms
   - Verifying banned terms are caught
   - Context-specific terminology enforcement

2. **BannedTermViolations** (8 tests)
   - Manager, Helper, Util, Data, Handler, Service
   - Generic terms that violate DDD principles

3. **ContextViolations** (8 tests)
   - Invalid cross-context dependencies
   - Proper dependency patterns
   - Communication through appropriate contexts

4. **DomainEventViolations** (4 tests)
   - Event naming and triggers
   - Domain event lifecycle

5. **ValueObjectViolations** (4 tests)
   - Proper value object implementation
   - Immutability and structure

6. **EntityViolations** (4 tests)
   - Entity identity and attributes
   - Required properties

7. **NamingConventionViolations** (2 tests)
   - PascalCase enforcement
   - Class naming standards

8. **AggregateViolations** (2 tests)
   - Aggregate boundaries
   - Aggregate root relationships

9. **Additional Tests** (2 tests)
   - Domain-specific edge cases

## Creating Custom Test Cases

### Step 1: Create Your SRS Document

Create a new `.txt` file in the `data/` directory following the same structure:

```
data/
└── your_domain_srs.txt
```

### Step 2: Create Your Test Cases JSON

Create a JSON file with this structure:

```json
{
  "description": "Description of your domain",
  "document": "your_domain_srs.txt",
  "test_cases": [
    {
      "id": 1,
      "violation_type": "NamingViolation",
      "message": "Using 'IncorrectTerm' instead of 'CorrectTerm'",
      "expected_section": "3.1",
      "expected_section_name": "Context Name"
    }
  ]
}
```

**Important**: The `document` field tells the benchmark which SRS file to load.

### Step 3: Run Your Benchmark

```bash
python script/benchmark_rag.py --test-cases data/your_domain_test_cases.json
```

The script will:
1. Load your test cases
2. Read the document specified in the JSON
3. Index the document in the RAG pipeline
4. Run all tests
5. Save results to `results/benchmark_results_your_domain.json`

## Test Case Schema

Each test case requires:

- **id**: Unique identifier (integer)
- **violation_type**: Type of DDD violation (string)
- **message**: The violation message to test (string)
- **expected_section**: Expected section number from SRS (string, e.g., "3.1")
- **expected_section_name**: Human-readable section name (string)

## Understanding Results

The benchmark outputs:

1. **Console Summary**
   - Average retrieval time (ms)
   - Relevance scores (top-1 and top-3)
   - Section accuracy percentages

2. **JSON Results File** (in `results/` directory)
   - Configuration details
   - Cold start time
   - Aggregate metrics
   - Detailed results per test case

### Key Metrics

- **Top-1 Accuracy**: % of cases where the best match is correct
- **Top-3 Accuracy**: % of cases where correct section appears in top 3 results
- **Relevance Score**: Similarity score from embedding model (0-1)
- **Retrieval Time**: Time to retrieve and rank results (milliseconds)

## Directory Structure

```
benchmarking/
├── script/
│   └── benchmark_rag.py                    # Main benchmarking script
├── data/
│   ├── ecommerce_srs.txt                   # E-commerce domain document
│   ├── ecommerce_test_cases.json           # E-commerce test suite (50 tests)
│   ├── healthcare_srs.txt                  # Healthcare domain document
│   ├── healthcare_test_cases.json          # Healthcare test suite (50 tests)
│   ├── banking_srs.txt                     # Banking domain document
│   └── banking_test_cases.json             # Banking test suite (50 tests)
└── results/
    ├── benchmark_results_ecommerce.json    # E-commerce results
    ├── benchmark_results_healthcare.json   # Healthcare results
    └── benchmark_results_banking.json      # Banking results
```

## Comparing Across Domains

You can benchmark all domains and compare results:

```bash
# Run all domains
python script/benchmark_rag.py --test-cases data/ecommerce_test_cases.json
python script/benchmark_rag.py --test-cases data/healthcare_test_cases.json
python script/benchmark_rag.py --test-cases data/banking_test_cases.json

# Compare results in the results/ directory
```

This helps you understand:
- How well the RAG system generalizes across domains
- Which violation types are easiest/hardest to detect
- Performance consistency across different terminology sets

## Tips

- Each test case file automatically loads its corresponding SRS document
- Results are automatically saved with domain-specific filenames
- Use different domains to test generalization of your RAG system
- Create custom domains to test industry-specific terminology
- All domains have the same test structure (50 tests) for fair comparison

## Example Workflow

```bash
# 1. Test e-commerce domain
python script/benchmark_rag.py

# 2. Test healthcare domain
python script/benchmark_rag.py --test-cases data/healthcare_test_cases.json

# 3. Test banking domain
python script/benchmark_rag.py --test-cases data/banking_test_cases.json

# 4. Create custom domain
# - Create data/logistics_srs.txt
# - Create data/logistics_test_cases.json
# - Run: python script/benchmark_rag.py --test-cases data/logistics_test_cases.json
```

## Adding New Domains

1. Create a new SRS document in `data/` following the existing structure
2. Create a test cases JSON file with the `document` field pointing to your SRS
3. Run the benchmark with your test cases file
4. The system handles everything automatically

The benchmark system is completely data-driven - just add your domain files and test cases!
