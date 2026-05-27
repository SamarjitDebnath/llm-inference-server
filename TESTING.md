# Testing Guide

## Overview

This project includes a comprehensive test suite with:
- **Unit tests** for core modules
- **Integration tests** for API endpoints and components
- **Latency benchmarks** for performance monitoring
- **Load pattern tests** for stress testing

## Quick Start

### Install Dependencies

```bash
make dev
```

This installs the project and all dev dependencies (pytest, black, isort, pylint, flake8, psutil).

### Run All Tests

```bash
make test-all
```

### Run Specific Test Suites

```bash
# Unit tests only
make test-unit

# Latency benchmarks only
make test-latency

# Integration tests only
pytest tests/test_integration.py -v
```

## Test Structure

```
tests/
├── __init__.py
├── conftest.py           # Pytest configuration and shared fixtures
├── test_unit.py          # Unit tests for core modules
├── test_integration.py   # Integration tests for API and components
└── test_latency.py       # Latency benchmarks and performance tests
```

## Test Categories

### Unit Tests (`test_unit.py`)

Tests for individual components:
- **Scheduler**: Request queue and request structures
- **Tokenizer**: Encoding and decoding functionality
- **API**: Routes and server initialization
- **Settings**: Configuration loading
- **Logger**: Logging setup and methods

Run with:
```bash
make test-unit
```

### Integration Tests (`test_integration.py`)

Tests for component interactions:
- FastAPI app initialization
- Scheduler request handling
- Error resilience

Run with:
```bash
pytest tests/test_integration.py -v
```

### Latency Benchmarks (`test_latency.py`)

Performance benchmarks measuring:

#### 1. **Tokenizer Latency** (`test_tokenizer_latency`)
- Measures time to tokenize a prompt
- 50 iterations
- **Target**: < 5ms mean latency

#### 2. **Batch Throughput** (`test_batch_throughput`)
- Measures processing speed of multiple prompts
- 10 batch runs
- **Target**: < 50ms per batch

#### 3. **Latency Stability** (`test_repeated_inference_stability`)
- Checks for latency degradation over time
- 100 consecutive runs
- **Target**: < 20% degradation from start to end

#### 4. **Concurrent Request Latency** (`test_concurrent_request_latency`)
- Simulates multiple concurrent requests
- 10 concurrent runs
- Measures throughput (prompts/sec)

#### 5. **Memory Under Load** (`test_memory_under_load`)
- Tracks memory usage over 500 iterations
- **Target**: < 100MB memory increase

Run with:
```bash
make test-latency
```

## Fixtures

Defined in `conftest.py`:

- `event_loop`: Async event loop for async tests
- `project_root`: Path to project root directory
- `test_prompt`: Single test prompt string
- `test_prompts`: List of 5 test prompts for batch testing

Use in tests:
```python
def test_something(test_prompt, test_prompts):
    # test_prompt is a single string
    # test_prompts is a list of strings
    pass
```

## Code Quality Tools

### Format Code

```bash
make format
```

Runs:
- `black` for code formatting
- `isort` for import sorting

### Lint Code

```bash
make lint
```

Runs:
- `pylint` for code analysis
- `flake8` for style guide enforcement

### Check Without Modifying

```bash
make check
```

Runs format check + lint without modifying files.

## Running Tests in CI/CD

Tests run automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

See `.github/workflows/tests.yml` for the GitHub Actions workflow.

## Performance Targets

| Test | Target | Description |
|------|--------|-------------|
| Tokenizer Latency | < 5ms | Single prompt encoding |
| Batch Throughput | < 50ms | Processing 5 prompts |
| Latency Stability | < 20% degradation | 100 runs comparison |
| Memory Under Load | < 100MB increase | 500 iterations |

## Debugging Tests

### Run with Verbose Output

```bash
pytest tests/ -v -s
```

The `-s` flag shows print statements.

### Run Specific Test

```bash
pytest tests/test_latency.py::TestLatencyBenchmarks::test_tokenizer_latency -v -s
```

### Run with Code Coverage

```bash
pytest tests/ --cov=. --cov-report=html
```

View HTML report:
```bash
open htmlcov/index.html
```

## Adding New Tests

1. Create test function with `test_` prefix in appropriate file
2. Use fixtures from `conftest.py`
3. Add docstring explaining what's tested
4. For latency tests, include performance target

Example:
```python
def test_my_feature(test_prompt):
    """Test my feature with a single prompt"""
    # Arrange
    my_obj = MyClass()
    
    # Act
    result = my_obj.do_something(test_prompt)
    
    # Assert
    assert result is not None
```

## Makefile Commands

```bash
make help          # Show all available commands
make install       # Install project (no dev dependencies)
make dev           # Install with dev dependencies
make run           # Run server (dev mode)
make run-prod      # Run server (production mode)
make test-unit     # Run unit tests
make test-latency  # Run latency benchmarks
make test-all      # Run all tests
make format        # Format code
make lint          # Lint code
make check         # Check without modifying
make logs          # Tail application logs
make clean         # Remove cache and artifacts
```

## Notes

- Tests skip gracefully if required modules (like `tokenizer_service`) are not available
- Latency tests include detailed performance reports with min/max/mean/median/p95/p99 metrics
- Memory tests require `psutil` (included in dev dependencies)
- GitHub Actions runs on Python 3.11 and 3.12
