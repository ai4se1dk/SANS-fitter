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

### Dataset Operations

The `sans_fitter.data_ops` module manipulates datasets with arithmetic
operations — similar to SasView's *Data Operation* utility. Typical uses are
background subtraction, rescaling to absolute units, and transmission
correction.

```python
from sans_fitter import SANSFitter, data_ops

sample = data_ops.load('sample.csv')          # standalone loader, returns Data1D
background = data_ops.load('empty_cell.csv')

net = data_ops.subtract(sample, background)   # sample − background
net = data_ops.divide(net, 0.8)               # transmission correction

fitter = SANSFitter()
fitter.set_data(net)                          # inject the in-memory dataset
fitter.set_model('sphere')
fitter.fit()
```

Available operations — each returns a new, fit-ready `Data1D`:

| Function | Result |
|---|---|
| `data_ops.add(a, b)` | `a + b` |
| `data_ops.subtract(a, b)` | `a − b` (order matters) |
| `data_ops.multiply(a, b)` | `a × b` |
| `data_ops.divide(a, b)` | `a / b` (order matters) |

The second operand can be a dataset or a scalar. For two datasets,
uncertainties are propagated (`dI = sqrt(dI_a² + dI_b²)` for add/subtract,
relative errors in quadrature for multiply/divide) and both must share the
same Q grid (x-values matching within 1% — interpolation onto a common grid
is not yet supported). For a scalar, `multiply`/`divide` scale both `I` and
`dI`, while `add`/`subtract` shift `I` and leave `dI` unchanged; the Q grid
is never altered.

Every result records its provenance: the title becomes the operation (e.g.
`"sample.csv - empty_cell.csv"`) and a `Process` entry is appended, which
survives in saved CanSAS output.

Things to know:

- **Missing dI** on an operand triggers a warning — it is treated as zero in
  error propagation. Error-free data warns again at fit time: the `lmfit`
  engine falls back to unit weights, while `bumps` refuses to fit.
- **NaN points** propagate through the arithmetic and are masked in the
  result (excluded from fits); a warning reports the masked count.
- **Resolution (dQ) propagation** through arithmetic is not validated
  upstream — a warning is emitted when any operand carries resolution data.
  Treat resolution on results with care, especially for slit-smeared data.

`SANSFitter.set_data()` accepts any sasdata `Data1D` — arithmetic results,
simulated data, or datasets built programmatically — and validates and
normalizes it so it is fit-ready.

See `examples/data_operations_example.py` and
`notebooks/data_operations_demo.ipynb` for a complete walkthrough.

### P(r) Inversion

The `sans_fitter.pr_inversion` module recovers the real-space pair distance
distribution function P(r) from I(q) by indirect Fourier transform (Moore's
sine-basis expansion, as in SasView's Inversion perspective). It is
**model-free** — no sasmodels kernel is involved — and operates directly on
datasets (`fitter.data` or `data_ops` results). Typical use: monodisperse
protein solutions, where P(r) yields D_max, Rg and I(0) without assuming a
form factor.

For buffer-subtracted data (the usual protein case), pass
`fit_background=False` — the default fitted flat background can absorb I(0)
and bias Rg on already-subtracted data. Explore D_max **before** trusting an
inversion: every result is conditional on it.

```python
from sans_fitter import data_ops, pr_inversion

data = data_ops.load('protein.csv')

# 1. Find a stable D_max: look for the Rg/I(0) plateau and chi2 minimum
scan = pr_inversion.explore_dmax(data, d_max=120.0, fit_background=False)
scan.plot()                    # or scan.plot(quantity='all'), scan.format_summary()

# 2. One-shot inversion with automatic selection of n_terms and alpha
result = pr_inversion.auto_invert(data, d_max=120.0, fit_background=False)
print(result.format_summary())  # Rg, I(0), oscillations, positivity, diagnostics

# 3. Plots and export
result.plot_pr()               # P(r) with its 1-sigma band
result.plot_fit(data)          # data vs fit, residuals (data passed explicitly)
result.save_csv('pr_result.csv')
```

Explicit control is available through the individual functions:

| Function | Result |
|---|---|
| `invert(data, d_max, n_terms=10, alpha=0.0, fit_background=True, background=0.0, r_points=101, regularizer='corrected')` | Core inversion → `PrResult` |
| `estimate_n_terms(data, d_max, fit_background=True, ..., background=0.0)` | `NTermsEstimate(n_terms, alpha, message)`; its `alpha` is authoritative — use it directly |
| `estimate_alpha(data, d_max, n_terms, fit_background=True, ..., background=0.0)` | `AlphaEstimate(alpha, message)` |
| `auto_invert(data, d_max, ...)` | `estimate_n_terms` → `invert`, silent |
| `explore_dmax(data, d_max, ..., refit_alpha=False, background=0.0)` | `DmaxScan` over 0.9–1.1×d_max (25 points); raises when every point fails |

When working with a known fixed background (`fit_background=False`), pass the
same `background` value to the estimators and `explore_dmax` too — the
selection and the scan then operate on exactly the problem the final
inversion solves (`auto_invert` does this automatically).

Things to know:

- **P(r) can go negative.** The fit is unconstrained (unlike GNOM/ATSAS);
  the `positive_fraction` diagnostics quantify how positive the result is.
- **alpha and n_terms are heuristics, not physics.** `estimate_alpha`
  descends from a norm-balance suggestion and stops at spurious structure or
  the discrepancy principle (chi-squared per point near 1); `estimate_n_terms`
  prefers the smallest N that fits the data with a significantly positive
  P(r). Always inspect `format_summary()`.
- **Missing dI** triggers fabricated uncertainties
  (`max(0.05*|I|, 0.01*median|I|)`), a warning, and an
  `uncertainties_fabricated` flag on the result — chi-squared diagnostics are
  then not interpretable.
- **Q range is honoured**: the inversion uses the same accepted-point rule as
  the fit engines, so `fitter.set_q_range()` restricts it identically.
- **Shannon limits are checked**: warnings fire when `d_max > pi/q_min` or
  `n_terms` exceeds `q_max*d_max/pi` (the data cannot support either).
- **Slit smearing is not supported** (a warning fires on slit-smeared data);
  pinhole dQ resolution is ignored, as in SasView.
- **`regularizer='sasview'`** reproduces SasView's exact smoothing operator
  for comparison; the default `'corrected'` penalizes the true second
  derivative on a resolved grid (validated against SasView — identical for
  spheres, and it remains reliable above 20 terms where SasView's fixed
  20-point penalty grid degrades).
- **Uncertainties are conditional**: the covariance (and the P(r) band)
  assumes known Gaussian errors at the chosen alpha and D_max, and is biased
  by the regularization. The summary's "approx. chi2 per residual dof" uses
  the regularization-aware effective dof, not the parameter count.

See `examples/pr_inversion_example.py` and
`notebooks/pr_inversion_demo.ipynb` for a complete walkthrough.

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

## Bayesian / Uncertainty Analysis

Beyond point estimates, SANS-fitter can sample the full posterior
distribution of the varying parameters with the DREAM Markov chain Monte
Carlo sampler (via BUMPS, which is already a dependency — no extra
installs needed).

### Running a Bayesian Fit

```python
fitter.load_data('my_sans_data.csv')
fitter.set_model('sphere')
fitter.set_param('radius', value=50, min=10, max=200, vary=True)
fitter.set_param('scale', value=0.1, min=0.01, max=1.0, vary=True)

# Sample the posterior with DREAM
result = fitter.fit_bayesian(samples=10000, burn=200)
```

`fit_bayesian()` prints the usual point-estimate summary plus a posterior
table with the mean, median, standard deviation, 68%/95% credible
intervals, and convergence diagnostics (R-hat, effective sample size) for
each sampled parameter. The reported parameter values are the best
(maximum-likelihood) posterior sample, and `stderr` is the posterior 68%
credible half-width.

Sampler controls:

- `samples`: number of posterior samples to draw (default 10000)
- `burn`: burn-in generations discarded before sampling (default 200)
- `thin`: keep every nth sample (default 1)
- `pop`: chain population scale per varying parameter (default 10)

### Posterior Displays

All five displays follow the same `show` convention as `plot_results()`
and return Plotly figures:

```python
# Corner plot: marginal densities + pairwise sample clouds
fitter.plot_posterior_pairs()
fitter.plot_posterior_pairs(params=['radius', 'scale'])  # subset

# Marginal posterior for a single parameter
fitter.plot_param_distribution('radius')

# Posterior predictive check: 95% credible band over the data
fitter.plot_posterior_predictive()                    # band only
fitter.plot_posterior_predictive(style='band+draws')  # band + sampled curves
fitter.plot_posterior_predictive(n_draws=100)         # more model evaluations

# Correlation heatmap of the sampled parameters
fitter.plot_param_correlations()

# MCMC chain traces (convergence check)
fitter.plot_trace()
```

Note: `plot_posterior_predictive()` re-evaluates the model once per draw,
so large `n_draws` values can be slow, especially with polydispersity
enabled.

### Accessing the Posterior Programmatically

```python
posterior = fitter.get_posterior()

posterior.labels        # sampled parameter names (chain order)
posterior.samples       # ndarray [n_samples, n_params]
posterior.ci_95         # {name: (low, high)} 95% credible intervals
posterior.diagnostics   # {name: {'r_hat': ..., 'ess': ...}}

print(posterior.format_summary())

# Export the raw chain for external analysis (e.g. corner, arviz, pandas)
posterior.save_posterior_csv('posterior_chain.csv')
```

`save_results()` also includes the credible intervals in the CSV header
after a Bayesian fit.
