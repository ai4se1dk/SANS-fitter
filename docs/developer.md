# Developer Guide

## Setting Up the Development Environment

1.  Clone the repository:
    ```bash
    git clone https://github.com/ai4se1dk/SANS-fitter.git
    cd SANS-fitter
    ```

2.  Install development dependencies:
    ```bash
    pip install -e ".[dev,docs]"
    ```
    Or using Pixi:
    ```bash
    pixi install
    ```

## Running Tests

We use `pytest` for testing.

To run all tests:
```bash
pytest
```

Or using the provided script:
```bash
python run_tests.py
```

To run tests with coverage:
```bash
pytest --cov=sans_fitter
```

## Building Documentation

The documentation is built using MkDocs.

To serve the documentation locally:
```bash
mkdocs serve
```

To build the static site:
```bash
mkdocs build
```

## Code Style

This project follows PEP 8 guidelines. We use `ruff` for linting and formatting.

To check for linting errors:
```bash
ruff check .
```

To format code:
```bash
ruff format .
```
