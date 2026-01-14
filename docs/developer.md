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
├── __init__.py            # Package exports
├── sans_fitter.py         # Main SANSFitter class
├── parameter_manager.py   # Parameter management
├── fitting_engine.py      # Abstract fitting engine interface
├── bumps_engine.py        # BUMPS fitting implementation
└── scipy_engine.py        # Scipy fitting implementation
```

### Key Components

#### SANSFitter Class

The main interface for SANS data fitting. Responsibilities:
- Data loading and management
- Model selection and kernel management
- Orchestrating fitting operations via strategy pattern
- Result visualization and export

#### ParameterManager Class

Encapsulates all parameter-related operations. Responsibilities:
- Parameter initialization from model kernels
- Parameter validation and bounds management
- Structure factor parameter linking
- Parameter state management (backup/restore)

#### FittingEngine Strategy Pattern

Abstract base class defining the interface for optimization engines:
- **FittingEngine**: Abstract interface
- **BumpsFittingEngine**: BUMPS optimization implementation
- **ScipyFittingEngine**: scipy.optimize implementation

Benefits:
- **Easy Extensibility**: Add new engines without modifying SANSFitter
- **Independent Testing**: Each engine can be tested in isolation
- **Swappable Backends**: Users can switch engines without code changes
- **Clean Separation**: Fitting logic separated from orchestration

### Design Decisions

**Parameter Access**: The `SANSFitter.params` property provides backward-compatible access to parameters while delegating to `ParameterManager` internally.

**Structure Factor Handling**: Structure factor logic is split between:
- Model loading (SANSFitter)
- Parameter state management (ParameterManager)

This allows clean separation while maintaining consistency.

**Fitting Engine Selection**: Engines are registered in `SANSFitter.__init__` and selected dynamically based on user's engine choice, enabling runtime flexibility.
