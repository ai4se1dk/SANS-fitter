# User Guide

This guide provides detailed instructions on how to use SANS Fitter for your data analysis.

## Basic Workflow

The typical workflow involves:
1.  Loading data
2.  Selecting a model
3.  Configuring parameters
4.  Fitting
5.  Visualizing and saving results

### 1. Loading Data

Use `load_data` to import your SANS data. The fitter supports various formats including CSV, XML (CanSAS), and HDF5 (NXcanSAS) via the `sasdata` library.

```python
from sans_fitter import SANSFitter

fitter = SANSFitter()
fitter.load_data('path/to/data.csv')
```

### 2. Selecting a Model

You can use any model available in the [SasModels library](https://www.sasview.org/docs/user/models/index.html).

```python
# Load a cylinder model
fitter.set_model('cylinder')

# Or a sphere model
fitter.set_model('sphere')
```

### 3. Configuring Parameters

Once a model is loaded, you can inspect and modify its parameters.

```python
# View all parameters
fitter.get_params()

# Set parameter values and bounds
fitter.set_param('radius', value=20, min=10, max=50, vary=True)
fitter.set_param('length', value=400, vary=False)  # Fix this parameter
```

-   `value`: The initial guess for the parameter.
-   `min` / `max`: The lower and upper bounds for the fit.
-   `vary`: Set to `True` to fit this parameter, `False` to keep it fixed.

### 4. Fitting

SANS Fitter supports two fitting engines: **BUMPS** and **LMFit**.

#### Using BUMPS (Default)

BUMPS is robust and offers several optimization methods.

```python
# Default method (Nelder-Mead simplex)
result = fitter.fit(engine='bumps', method='amoeba')

# Differential Evolution
result = fitter.fit(engine='bumps', method='de')
```

#### Using LMFit

LMFit provides access to SciPy's optimization algorithms.

```python
# Levenberg-Marquardt
result = fitter.fit(engine='lmfit', method='leastsq')
```

### 5. Visualization and Export

After fitting, you can plot the results and save them.

```python
# Plot data, fit, and residuals
fitter.plot_results(show_residuals=True, log_scale=True)

# Save results to CSV
fitter.save_results('fit_results.csv')
```

`plot_results` returns the plotly figure. In scripts it opens the plot
automatically; in Jupyter notebooks the returned figure is rendered by the
notebook itself, so the plot appears exactly once. Pass `show=True` or
`show=False` to override this behaviour.

Error bars are drawn from the `dI` column (vertical) and, when present, the
`dQ` resolution column (horizontal). Columnar text/CSV files are read in the
order `Q, I, dI, dQ` — if your file stores `dQ` in the third column, it will
be misinterpreted as `dI`. The summary printed by `load_data()` shows which
columns were detected.

## Advanced Usage

### Restricting the Q Range

Real datasets often contain points you do not want to fit: beam-stop
spillover at low Q or background-dominated points at high Q. Use
`set_q_range` to restrict the fit without editing the data file.

```python
# Fit only points with 0.01 <= Q <= 0.3 Å⁻¹
fitter.set_q_range(qmin=0.01, qmax=0.3)

# Either bound may be given alone; the other resets to the full range
fitter.set_q_range(qmax=0.3)

# Inspect and restore
fitter.get_q_range()    # -> (qmin, qmax)
fitter.reset_q_range()  # back to the full data range
```

The restriction applies to both fitting engines. Excluded points still
appear in plots (grayed out, labelled "Excluded Data"), but the fitted
curve, residuals, χ², and the CSV export only cover the fitted range.
The range can be changed freely between fits — each fit result remembers
the range it was fitted with.

### Structure Factors

You can combine a form factor with a structure factor to model interacting systems.

```python
fitter.set_model('sphere')
fitter.set_structure_factor('hardsphere')
```

Supported structure factors include:
-   `hardsphere`
-   `hayter_msa`
-   `squarewell`
-   `stickyhardsphere`

### Effective Radius

When using a structure factor, you often need to define an effective radius. You can link this to the form factor's radius.

```python
# Link effective radius to the sphere radius
fitter.set_structure_factor('hardsphere', radius_effective_mode='link_radius')
```

## Polydispersity

SANS Fitter supports polydispersity, which models size distributions in your samples. Many real samples have a distribution of particle sizes rather than a single monodisperse size.

### Checking Polydispersity Support

Not all model parameters support polydispersity. Check which parameters are polydisperse:

```python
# Check if model supports polydispersity
if fitter.supports_polydispersity():
    # Get list of polydisperse parameters
    pd_params = fitter.get_polydisperse_parameters()
    print(f"Polydisperse parameters: {pd_params}")
```

### Configuring Polydispersity

Configure polydispersity for a specific parameter:

```python
# Set polydispersity width (relative, 0.0 = monodisperse, 0.1 = 10% width)
fitter.set_pd_param('radius', pd_width=0.1)

# Configure all PD options
fitter.set_pd_param(
    'radius',
    pd_width=0.15,      # 15% polydispersity
    pd_n=50,            # Number of quadrature points (default: 35)
    pd_nsigma=4.0,      # Number of sigmas to include (default: 3.0)
    pd_type='gaussian', # Distribution type
    vary=True           # Allow pd_width to vary during fitting
)

# Get current PD configuration
pd_config = fitter.get_pd_param('radius')
print(pd_config)  # {'pd': 0.15, 'pd_n': 50, 'pd_nsigma': 4.0, 'pd_type': 'gaussian', 'vary': True, 'active': True}
```

### Distribution Types

SANS Fitter supports several polydispersity distribution types:

- `gaussian` - Gaussian/normal distribution (default)
- `rectangle` - Uniform/rectangular distribution
- `lognormal` - Log-normal distribution
- `schulz` - Schulz distribution (common for polymers)
- `boltzmann` - Boltzmann distribution

```python
# Use Schulz distribution for polymer samples
fitter.set_pd_param('radius', pd_width=0.2, pd_type='schulz')
```

### Enabling/Disabling Polydispersity

You can globally enable or disable polydispersity:

```python
# Enable polydispersity globally
fitter.enable_polydispersity(True)

# Check if enabled
if fitter.is_polydispersity_enabled():
    print("Polydispersity is enabled")

# Disable polydispersity (values are preserved)
fitter.enable_polydispersity(False)
```

### Viewing Polydispersity Parameters

Display all polydispersity parameter settings:

```python
# Print PD parameter table
fitter.get_pd_params()
```

### Fitting with Polydispersity

When fitting with polydispersity, you can choose to fix or vary the polydispersity width:

```python
# Set up model and polydispersity
fitter.set_model('sphere')
fitter.set_param('radius', value=50, min=10, max=200, vary=True)
fitter.set_pd_param('radius', pd_width=0.1, vary=True)  # Fit the PD width
fitter.enable_polydispersity(True)

# Fit - will optimize both radius and radius_pd
result = fitter.fit(engine='bumps')
```
