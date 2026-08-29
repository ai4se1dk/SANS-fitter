# Installation

## Prerequisites

SANS Fitter requires Python 3.10 or later.

## Installing from Source

### Using pip

You can install SANS Fitter directly from the repository using pip:

```bash
# Clone the repository
git clone https://github.com/ai4se1dk/SANS-fitter.git
cd SANS-fitter

# Install the package
pip install -e .
```

To install with development dependencies (for running tests or building documentation):

```bash
pip install -e ".[dev,docs]"
```

The Jupyter stack is an optional extra — the library itself does not need it.
Install it only to run the notebooks in `notebooks/`:

```bash
pip install -e ".[notebooks]"
```

### Using Pixi

If you use [Pixi](https://prefix.dev/) for package management:

```bash
# Clone the repository
git clone https://github.com/ai4se1dk/SANS-fitter.git
cd SANS-fitter

# Install dependencies
pixi install
```

## Dependencies

SANS Fitter relies on the following libraries:

- **numpy**: Numerical computing
- **plotly**: Interactive plotting and visualization
- **scipy**: Scientific computing and optimization
- **sasmodels**: SANS model calculations
- **sasdata**: SANS data loading
- **bumps**: Bayesian Uncertainty Modeling for Parameter Selection (fitting engine)

These will be automatically installed when you install SANS Fitter.
