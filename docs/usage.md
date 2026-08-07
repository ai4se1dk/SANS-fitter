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

### Combining Models (Composite Models)

Datasets with several distinct features — for example a low-Q diffuse
scattering contribution plus a high-Q correlation peak — are often best
described by *several models fitted simultaneously* against the same data.
`set_models()` combines any sasmodels models into one fit:

```python
fitter = SANSFitter()
fitter.load_data('data.csv')

fitter.set_models('dab', 'peak_lorentz')
fitter.set_param('dab_cor_length', value=50, min=1, max=500, vary=True)
fitter.set_param('dab_scale', value=10, min=0.1, max=100, vary=True)
fitter.set_param('peak_lorentz_peak_pos', value=0.1, min=0.01, max=0.5, vary=True)
fitter.set_param('peak_lorentz_peak_hwhm', value=0.01, min=0.001, max=0.1, vary=True)
fitter.set_param('background', value=0.001, min=0, max=0.1, vary=True)

result = fitter.fit(engine='bumps')
fitter.plot_results(show_components=True)
```

**How the combination works.** With the default `operation='+'` the combined
intensity is

    I(q) = scale · [dab_scale·I_dab(q) + peak_lorentz_scale·I_peak(q)] + background

The global `scale` and `background` are shared by every component natively
(sasmodels' mixture semantics), while each component carries its own
`<name>_scale`. Varying the global `scale` together with a component scale is
degenerate — only their product is fitted — so `fit()` warns when both are
free. With `operation='*'` the part intensities multiply instead.

**Friendly parameter names.** Every component parameter is prefixed with the
model name (`dab_cor_length`, `peak_lorentz_peak_pos`). Give components custom
names (monikers) with keyword arguments — useful for long model names,
duplicates, or physics labels:

```python
fitter.set_models(small='sphere', large='sphere', shared=['sld', 'sld_solvent'])
fitter.set_param('small_radius', value=20, min=5, max=100, vary=True)
fitter.set_param('large_radius', value=200, min=50, max=1000, vary=True)
fitter.set_param('sld', value=4.0, vary=True)   # one knob drives both spheres
```

**Sharing parameters.** Each name in `shared=[...]` must exist in at least
two components; it becomes a single unprefixed parameter driving all of them
(the per-component versions disappear from the parameter list). This is the
one-line answer to "share SLD across models". Note that polydispersity
configuration stays per-component: after `shared=['radius']`,
`set_pd_param('small_radius', ...)` and `set_pd_param('large_radius', ...)`
still configure the two components independently.

**Structure factors on one part.** A component entry may itself contain `@`,
applying a structure factor to that part only:

```python
fitter.set_models('sphere@hardsphere', 'peak_lorentz')
```

(`@` binds tighter than `+`, so this is `(sphere@hardsphere) + peak_lorentz`.)
Applying `set_structure_factor()` to an already-composite model raises an
error — sasmodels cannot express `(A+B)@S`.

**Component curves.** After fitting a `'+'` mixture,
`plot_results(show_components=True)` overlays one dashed curve per component,
each drawn as `scale · part_scale · I_part(q)` (background excluded, shown
implicitly in the total curve). For `'*'` mixtures and atomic models the flag
is a documented no-op.

**Equality links.** For sharing that `shared=` cannot express — linking only
some components, or parameters with different names — use explicit links:

```python
fitter.link_params('large_sld', to='small_sld')      # follower mirrors target
fitter.link_params('shell_sld_core', to='small_sld') # different names work too
fitter.unlink_params('large_sld')                    # escape hatch
```

A follower is forced `vary=False` and mirrors the target's value before,
during, and after the fit; writing it directly raises. Link chains are not
supported.

**Raw string syntax (advanced).** `set_model()` accepts sasmodels' native
composite expressions directly and keeps the canonical `A_`/`B_` parameter
names — zero magic when following sasmodels documentation:

```python
fitter.set_model('dab+peak_lorentz')   # A_scale, A_cor_length, B_scale, ...
```

Every atomic name in the expression is validated before loading, with a
nearest-match suggestion for typos.

**Engine support.** Composite models and parameter links currently work with
the `bumps` engine only; `fit(engine='lmfit')` and `fit_bayesian()` raise
`NotImplementedError` when either is active.

See `examples/composite_model_example.py` for a complete runnable example.

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
