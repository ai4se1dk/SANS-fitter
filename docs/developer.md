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

## Architecture

### Overview

The SANS-fitter package is organized into modular components for maintainability and extensibility:

```
src/sans_fitter/
├── __init__.py           # Package exports
├── sans_fitter.py        # Main SANSFitter class
└── parameter_manager.py  # Parameter management
```

### Key Components

#### SANSFitter Class

The main interface for SANS data fitting. Responsibilities:
- Data loading and management
- Model selection and kernel management
- Orchestrating fitting operations
- Result visualization and export

#### ParameterManager Class

Encapsulates all parameter-related operations. Responsibilities:
- Parameter initialization from model kernels
- Parameter validation and bounds management
- Structure factor parameter linking
- Parameter state management (backup/restore)

This separation improves code organization by:
- **Single Responsibility**: Each class has a clear, focused purpose
- **Testability**: Parameter logic can be tested independently
- **Maintainability**: Parameter management changes don't affect core fitting logic
- **Extensibility**: Easy to add new parameter features without modifying SANSFitter

### Design Decisions

**Parameter Access**: The `SANSFitter.params` property provides backward-compatible access to parameters while delegating to `ParameterManager` internally.

**Structure Factor Handling**: Structure factor logic is split between:
- Model loading (SANSFitter)
- Parameter state management (ParameterManager)

This allows clean separation while maintaining consistency.
