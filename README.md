# SANS Model Fitter

[![Tests](https://github.com/ai4se1dk/SANS-fitter/actions/workflows/ci.yml/badge.svg)](https://github.com/ai4se1dk/SANS-fitter/actions/workflows/ci.yml)
[![Docs](https://github.com/ai4se1dk/SANS-fitter/actions/workflows/docs.yml/badge.svg)](https://ai4se1dk.github.io/SANS-fitter/)
[![PyPI badge](https://img.shields.io/pypi/v/sans-fitter.svg)](https://pypi.python.org/pypi/sans-fitter)
[![Docs badge](https://img.shields.io/badge/docs-built-blue)](https://ai4se1dk.github.io/SANS-fitter/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/2ac22dd161034cdebb7478c395aff59c)](https://app.codacy.com/gh/ai4se1dk/SANS-fitter/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)


A flexible, model-agnostic Python template for fitting Small-Angle Neutron Scattering (SANS) data using the SasModels library.

## Features

- **Model-Agnostic Design**: Works with any model from the SasModels library (cylinder, sphere, core_shell, etc.)
- **Multiple Fitting Engines**: Supports both BUMPS (default) and LMFit optimization engines
- **Flexible Data Loading**: Reads CSV, XML, and HDF5 formats via sasdata
- **Q-Range Restriction**: Fit only a chosen [qmin, qmax] window (e.g. trim beam-stop or background-dominated points)
- **Dataset Arithmetic**: Add, subtract, multiply, and divide datasets (or scale by constants) with propagated uncertainties via `data_ops` — e.g. background subtraction and transmission correction before fitting
- **User-Friendly Parameter Management**: Easy-to-use interface for setting parameter values, bounds, and fitting flags
- **Interactive Visualization**: Automatic plotting of data, fitted model, and residuals with Plotly
- **Bayesian Analysis**: Posterior sampling with BUMPS DREAM (MCMC) plus corner, marginal, predictive-band, correlation, and trace plots
- **P(r) Inversion**: Model-free pair distance distribution analysis (indirect Fourier transform) via `pr_inversion` — D_max exploration, automatic regularization/term selection, and Rg/I(0)/positivity diagnostics
- **Result Export**: Save fitted parameters and curves to CSV files

## Installation

### Option 1: Using pip (recommended for users)

```bash
# Clone the repository
git clone https://github.com/ai4se1dk/SANS-fitter.git
cd SANS-fitter

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Option 2: Using Pixi (recommended for development)

```bash
# Clone the repository
git clone https://github.com/ai4se1dk/SANS-fitter.git
cd SANS-fitter

# Install dependencies with Pixi
pixi install

# Run tests
pixi run test

# Run demo notebook
pixi run run-demo
```

## Quick Start

```python
from sans_fitter import SANSFitter

# Create fitter instance
fitter = SANSFitter()

# Load your data
fitter.load_data('my_sans_data.csv')

# Set the model (any model from SasModels!)
fitter.set_model('cylinder')

# Optionally restrict the Q range used for fitting
fitter.set_q_range(qmin=0.01, qmax=0.3)

# View initial parameter values
fitter.get_params()

# Configure parameters for fitting
fitter.set_param('radius', value=20, min=1, max=100, vary=True)
fitter.set_param('length', value=400, min=10, max=1000, vary=True)
fitter.set_param('scale', value=0.1, min=0, max=1, vary=True)
fitter.set_param('background', value=0.01, min=0, max=1, vary=True)

# View current parameters
fitter.get_params()

# Perform the fit (using BUMPS by default)
result = fitter.fit(engine='bumps', method='amoeba')

# Visualize results
fitter.plot_results(show_residuals=True)

# Save results
fitter.save_results('fit_results.csv')
```

## Switching Models

The fitter is completely model-agnostic. Simply load a different model:

```python
# Try with a sphere model instead
fitter.set_model('sphere')
fitter.get_params()  # See different parameters!

fitter.set_param('radius', value=25, min=5, max=100, vary=True)
result = fitter.fit()
```

## Switching Fitting Engines

```python
# Use BUMPS (default)
result = fitter.fit(engine='bumps', method='amoeba')

# Or use LMFit
result = fitter.fit(engine='lmfit', method='leastsq')
```

## Working with Structure Factors

Combine any SasModels form factor with an interaction model to capture correlated systems.

```python
fitter.set_model('sphere')

# Apply a structure factor (creates sphere@hardsphere product model)
fitter.set_structure_factor('hardsphere', radius_effective_mode='link_radius')

# Inspect linked parameters and run the fit as usual
fitter.get_params()
result = fitter.fit()

# Remove the structure factor to go back to the pure form factor
fitter.remove_structure_factor()
```

- **Supported structure factors:** `hardsphere`, `hayter_msa`, `squarewell`, `stickyhardsphere`.
- **Radius handling:** use `radius_effective_mode='link_radius'` to keep `radius_effective` equal to the form-factor `radius`, or leave the default `unconstrained` to fit it independently.
- **State helpers:** `get_structure_factor()` returns the active structure factor so notebooks/scripts can branch as needed.

## Bayesian Analysis

Sample the full posterior distribution of the varying parameters with the
DREAM MCMC sampler (built into BUMPS — no extra dependencies):

```python
fitter.set_model('sphere')
fitter.set_param('radius', value=50, min=10, max=200, vary=True)
fitter.set_param('scale', value=0.1, min=0.01, max=1.0, vary=True)

# Run the Bayesian fit (prints point estimates + credible intervals + diagnostics)
result = fitter.fit_bayesian(samples=10000, burn=200)

# Corner plot of the posterior
fitter.plot_posterior_pairs()

# More displays
fitter.plot_param_distribution('radius')      # marginal posterior
fitter.plot_posterior_predictive()            # 95% credible band over the data
fitter.plot_param_correlations()              # correlation heatmap
fitter.plot_trace()                           # MCMC chain traces

# Raw chain access / export
posterior = fitter.get_posterior()
posterior.save_posterior_csv('posterior_chain.csv')
```

See the [User Guide](https://ai4se1dk.github.io/SANS-fitter/usage/) for details.

### Available Methods

**BUMPS methods:**
- `'amoeba'` - Nelder-Mead simplex (default, robust)
- `'lm'` - Levenberg-Marquardt
- `'newton'` - Newton's method
- `'de'` - Differential evolution

**LMFit methods:**
- `'leastsq'` - Levenberg-Marquardt (default)
- `'least_squares'` - Trust Region Reflective
- `'differential_evolution'` - Global optimizer
- `'powell'`, `'nelder'`, etc.

## Demo Notebooks

- [notebooks/sans_fitter_demo.ipynb](notebooks/sans_fitter_demo.ipynb) — comprehensive demonstration of the fitting workflow with examples.
- [notebooks/bayesian_sampling.ipynb](notebooks/bayesian_sampling.ipynb) — Bayesian posterior sampling API (`fit_bayesian()`) and the associated posterior plots.
- [notebooks/pr_inversion_demo.ipynb](notebooks/pr_inversion_demo.ipynb) — model-free P(r) inversion: D_max exploration, automatic inversion, and diagnostics.


## Design Philosophy

This implementation follows a **template pattern** where:

1. The core fitting logic is abstracted into a reusable class
2. Models are loaded dynamically from SasModels - no hardcoded model assumptions
3. Parameters are discovered automatically from the model definition
4. Multiple optimization engines are supported through a unified interface
5. The user maintains full control over parameter initialization and bounds

## Implementation Details

### Engine Adapters

The fitter implements adapter patterns for both BUMPS and LMFit:

- **BUMPS**: Uses native `sasmodels.bumps_model` integration
- **LMFit**: Uses `sasmodels.direct_model.DirectModel` with a custom residual function

### Parameter Management

Parameters are stored internally with:
- `value`: Current/initial value
- `min`, `max`: Bounds
- `vary`: Fitting flag
- `description`: From model metadata

This allows the fitter to work with any model without prior knowledge of its parameters.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for the full text.

## References

- SasModels: https://github.com/SasView/sasmodels
- BUMPS: https://github.com/bumps/bumps
- LMFit: https://lmfit.github.io/lmfit-py/
- Plotly: https://plotly.com/python/
