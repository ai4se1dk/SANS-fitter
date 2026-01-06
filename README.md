# SANS Model Fitter

[![Tests](https://github.com/ai4se1dk/SANS-fitter/actions/workflows/tests.yml/badge.svg)](https://github.com/ai4se1dk/SANS-fitter/actions/workflows/tests.yml)
[![Docs](https://github.com/ai4se1dk/SANS-fitter/actions/workflows/docs.yml/badge.svg)](https://ai4se1dk.github.io/SANS-fitter/)
[![codecov](https://codecov.io/gh/ai4se1dk/SANS-fitter/graph/badge.svg)](https://codecov.io/gh/ai4se1dk/SANS-fitter)

A flexible, model-agnostic Python template for fitting Small-Angle Neutron Scattering (SANS) data using the SasModels library.

## Features

- **Model-Agnostic Design**: Works with any model from the SasModels library (cylinder, sphere, core_shell, etc.)
- **Multiple Fitting Engines**: Supports both BUMPS (default) and LMFit optimization engines
- **Flexible Data Loading**: Reads CSV, XML, and HDF5 formats via sasdata
- **User-Friendly Parameter Management**: Easy-to-use interface for setting parameter values, bounds, and fitting flags
- **Comprehensive Visualization**: Automatic plotting of data, fitted model, and residuals
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

## Demo Notebook

See [sans_fitter_demo.ipynb](sans_fitter_demo.ipynb) for a comprehensive demonstration with examples.


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

## Web Application

A Streamlit-based web application is now available for interactive SANS data analysis with a user-friendly interface!

### Features

- 📤 **Data Upload**: Upload your SANS datasets (CSV or .dat files)
- 🤖 **AI-Assisted Model Selection**: Get intelligent model suggestions based on your data
- 🎯 **Manual Model Selection**: Choose from all available SasModels
- ⚙️ **Interactive Parameter Tuning**: Adjust parameters with real-time UI controls
- 📊 **Interactive Plots**: Visualize data and fits with Plotly's zoom, pan, and export features
- 💾 **Export Results**: Save fitted parameters and curves to CSV

### Quick Start (Web App)

```bash
# Install web application dependencies
pip install -e ".[web]"

# Run the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Using the Web Application

1. **Upload Data**: Use the sidebar to upload your SANS data file (CSV or .dat format with Q, I, dI columns) or load the example dataset
2. **Select Model**: 
   - **Manual**: Choose from dropdown of all SasModels models
   - **AI-Assisted**: Optionally provide an OpenAI API key for AI-powered suggestions, or use built-in heuristics
3. **Configure Parameters**: Set initial values, bounds, and which parameters to fit
4. **Run Fit**: Choose optimization engine (BUMPS or LMFit) and method, then click "Run Fit"
5. **View Results**: Interactive plots show data with error bars and fitted curve
6. **Export**: Download fitted parameters as CSV

### Web App Deployment

#### Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account and deploy from the repository
4. Set `app.py` as the main file

#### Heroku

```bash
# Create Procfile
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

#### Docker

```bash
# Build image
docker build -t sans-fitter-app .

# Run container
docker run -p 8501:8501 sans-fitter-app
```

### API Integration

The web app supports optional AI-powered model suggestions via the OpenAI API:

1. Get an API key from [platform.openai.com](https://platform.openai.com)
2. Enter the key in the sidebar when using AI-Assisted mode
3. Or set as environment variable: `export OPENAI_API_KEY=your-key-here`

**Note**: The app also works without an API key using built-in heuristic suggestions.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for the full text.

## References

- SasModels: https://github.com/SasView/sasmodels
- BUMPS: https://github.com/bumps/bumps
- LMFit: https://lmfit.github.io/lmfit-py/
- Streamlit: https://streamlit.io
- Plotly: https://plotly.com/python/
