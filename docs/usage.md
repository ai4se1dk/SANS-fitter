# User Guide

This guide provides detailed instructions on how to use SANS Fitter for your data analysis.

## Basic Workflow

The typical workflow involves:
1.  Loading data
2.  Selecting a model
3.  Configuring parameters
4.  Previewing the model
5.  Fitting
6.  Visualizing and saving results

### 1. Loading Data

Use `load_data` to import your SANS data. The fitter supports various formats including CSV, XML (CanSAS), and HDF5 (NXcanSAS) via the `sasdata` library.

```python
from sans_fitter import SANSFitter

fitter = SANSFitter()
fitter.load_data('path/to/data.csv')
```

Some files contain **more than one dataset** (e.g. a CanSAS XML with several
`SASentry` blocks, or an NXcanSAS file with multiple entries). In that case
`load_data()` (and `data_ops.load()`) print a warning listing all available
datasets and load the first one. Select a specific dataset by 0-based index
or by name (title, run id, or filename):

```python
# Pick the second dataset by index
fitter.load_data('multi_dataset.xml', dataset=1)

# Or by name (title, run id, or filename)
fitter.load_data('multi_dataset.xml', dataset='beta sample')

# data_ops.load accepts the same argument
import sans_fitter.data_ops as data_ops
sample = data_ops.load('multi_dataset.xml', dataset='beta sample')
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

### 4. Previewing the Model

Check that the starting values are sane before committing to a fit. None of
these change the fitter's parameters or its fit results.

```python
import numpy as np

# Data, model at the current parameters, and residuals - no fit required
fitter.plot_model()

# The intensities themselves: on the data grid (NaN outside the fit range,
# data's own resolution applied) or on any grid, optionally smeared by dQ/Q
intensity = fitter.calculate()
smooth = fitter.calculate(q=np.geomspace(0.005, 0.5, 300), dq=0.05)

# Overlay candidate parameter sets: a one-parameter sweep, or labelled cases
fitter.compare(radius=[20, 30, 40])
fitter.compare({'current': {}, '20% polydisperse': {'radius_pd': 0.2}})
```

The goodness of fit shown by `plot_model()` is χ²/dof, the same number BUMPS
prints as "Initial χ²" at the start of a fit, and the same convention every
engine reports in `result['reduced_chisq']`. The preview is evaluated through the
active resolution mode, so it is directly comparable with the fit that follows.
It is reported as not available when the data has no `dI` column, and when the
free parameters outnumber the fitted points.

See `examples/theory_preview_example.py` for a runnable walkthrough.

### 5. Fitting

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

#### Reading the result

Both engines return the same structure, so code written against one works
against the other:

```python
result['engine']   # 'bumps' or 'lmfit'
result['method']   # the optimization method used
result['chisq']    # raw chi-squared: sum of squared weighted residuals
result['parameters']  # one entry per model parameter
```

The goodness-of-fit block means the same thing whichever engine ran:

| Field | Meaning |
|---|---|
| `chisq` | Σ((I − I_fit)/dI)² over the fitted points: **raw**, not normalized |
| `reduced_chisq` | `chisq / dof`; not a number when `dof <= 0` |
| `n_points` | Points that took part in the fit (inside the Q range, unmasked, finite) |
| `n_free` | Parameters the optimizer varied, polydispersity widths included |
| `dof` | `n_points - n_free` |
| `converged` | `True` / `False` when the optimizer reports a verdict, `None` when it does not |
| `message` | The optimizer's own termination message |
| `weighting_note` | How the residuals were weighted, e.g. `'dI'` |
| `cov`, `cov_labels`, `cov_source` | Covariance over the varied parameters, its parameter order, and where it came from |
| `on_bounds` | `(parameter, 'min' | 'max')` for each fitted parameter resting on a bound |

!!! warning "Changed in 0.4"
    `result['chisq']` from the bumps engine was χ²/dof before 0.4; use
    `result['reduced_chisq']`. The two engines previously disagreed: bumps
    normalized by the degrees of freedom while the LMFit engine stored the raw
    sum, so the same fit produced numbers a factor of `dof` apart. Both now
    report raw χ² in `chisq` and the normalized value in `reduced_chisq`.

Each entry in `result['parameters']` carries the same five fields:

| Field | Meaning |
|---|---|
| `value` | Fitted value, or the value the parameter was held at |
| `stderr` | Uncertainty on the value; `0.0` for a genuinely fixed parameter |
| `formatted` | Display string, e.g. `45.041(46)`, `2 (fixed)` |
| `fixed` | `False` for the parameters the optimizer varied, `True` otherwise |
| `linked_to` | Name of the parameter this one follows, or `None` |

```python
fitted = {
    name: info['value'] for name, info in result['parameters'].items() if not info['fixed']
}
```

A parameter that follows another one - through `link_params()` or
`radius_effective_mode='link_radius'` - reports its target's *fitted* value and
names that target in `linked_to`.

**It also reports its target's uncertainty.** An equality link makes the two
parameters one quantity under two names, so the follower's error is the
target's, not zero. `fixed=True` still separates the optimizer's coordinates
from everything else, but it no longer implies a zero uncertainty: read
`linked_to` to tell a follower from a genuinely fixed parameter, whose error
really is zero. A follower of a *fixed* target keeps the zero, because its
target never moved.

*Changed in 0.5.* Before, a follower reported `stderr = 0.0` and a `(linked)`
suffix in `formatted`. Analyses saved earlier are normalized on load wherever
the target's uncertainty is still in the file; where it cannot be recovered the
saved value stands rather than being invented.

#### The fit report

`get_fit_report()` returns the same information as an object that renders itself,
so a notebook cell shows a table instead of a wall of text:

```python
report = fitter.get_fit_report()
report                      # rich table in a notebook
print(report)               # plain text in a terminal
report.to_markdown()        # a string for a document or an issue comment
report.to_dict()            # JSON-safe data (non-finite values become None)
report.strongly_correlated(threshold=0.95)   # [(param_a, param_b, rho), ...]
```

The report carries a header line (model, engine, resolution, weighting), the
quality table, one row per parameter with its status (`fitted`, `fixed`,
`linked → target`, `fitted, on bound (max)`), the correlation matrix when more
than one parameter varied, and the posterior summary after `fit_bayesian()`.
It is a snapshot: running another fit does not change a report already returned.
`get_fit_report()` raises if no fit has been run. `plot_model()` previews the
theory without fitting and produces no result.

#### Judging a fit

- **Reduced χ² near 1** means the residuals are the size the error bars claim.
  Much greater than 1 is a model that does not describe the data (or `dI` that
  is too small); much less than 1 usually means `dI` is too large, or that the
  model has more freedom than the data supports.
- **χ² is not comparable across resolution modes**, because smearing
  redistributes residual structure. It is also not comparable across different
  Q ranges or masks, since `n_points` changes with them.
- **A parameter on a bound** raises a warning and appears in `on_bounds`. That
  can be physically correct - a non-negative background at zero, say - so the
  warning says the estimate *may be* constrained by the limit rather than that a
  better one lies outside it. When the boundary was not intentional, widen it
  and refit.
- **Uncertainties are not rescaled.** `stderr` is √diag(`cov`) with no
  √(χ²/dof) factor, which is what both bumps and SciPy's `leastsq` report. If
  your `dI` values are relative weights rather than absolute uncertainties,
  multiply by `sqrt(result['reduced_chisq'])` yourself:

    ```python
    import math

    scale = math.sqrt(result['reduced_chisq'])
    absolute = {
        name: info['stderr'] * scale
        for name, info in result['parameters'].items()
        if not info['fixed']
    }
    ```

- **Covariance is a local, symmetric estimate.** It is unreliable at an active
  bound, and for strongly non-linear or non-identifiable models. A very large
  variance is a diagnostic worth following up, not an uncertainty to quote.
  `cov_source` says where the matrix came from: `'jacobian'` for the bumps
  point estimate and LMFit's `least_squares`, `'scipy cov_x'` for `leastsq`,
  `'posterior sample'` after `fit_bayesian()`. Differential evolution supplies
  none, and `cov` is then `None`.
- **Strong correlations** mean the data does not separate those parameters.
  `report.strongly_correlated()` lists pairs at |ρ| ≥ 0.95; fix one of them, or
  reparameterize.
- **Convergence is `None` on the bumps engine.** bumps reports success for every
  fit regardless of the outcome, so there is no verdict to pass on; the message
  carries the iteration count and the configured maximum instead.

### 6. Visualization and Export

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
order `Q, I, dI, dQ`. If your file stores `dQ` in the third column, it will
be misinterpreted as `dI`. The summary printed by `load_data()` shows which
columns were detected.

### 7. Saving and sharing an analysis

`save_results()` writes the *outcome* of a fit. `save_analysis()` writes how the
fit was set up, so the analysis can be reproduced later or by someone else:

```python
fitter.save_analysis('silica_35C.json')

fitter = SANSFitter.load_analysis('silica_35C.json')
fitter.plot_results()      # the saved result, no refit needed
```

The file is JSON, so it is readable, reviewable in a pull request, and safe to
accept from a collaborator. It records the model expression and its component
names, every parameter value, bound and vary flag, polydispersity, links, the
structure factor, the resolution mode and the Q range. It does not record data
curves or posterior sample chains; the covariance matrix, being small and
useful, is kept.

**Saving and loading guarantee the configuration, not a refit.** A restored
fitter is identical in setup and produces the same theory, residuals and fit
index as the one that was saved. Calling `fit()` afterwards runs a fresh
optimization with the current API defaults, because the file records the engine
and method but not the iteration budget, random seed or starting point, and
several optimizers are not deterministic.

#### The result is saved only while it still describes the setup

Changing a parameter, the Q range, the resolution or the data does not clear
the last fit result. Saving both together would pair a χ² with a configuration
that never produced it, so `save_analysis()` compares the two and leaves the
result out when they disagree:

```
✓ Analysis saved to silica_35C.json
  Fit result NOT included: the model or its parameters changed after the fit
```

The setup is still written, so nothing is lost. Fit before you save, or save
before you experiment. The same check runs on load: an analysis opened against
different data restores the configuration and reports that the result was not
restored.

#### Finding the data again

The path to the data file is recorded twice, relative to the analysis file and
absolute, so moving both together to another machine still works. Supply
another dataset with `data=`:

```python
# A different sample, same model setup
fitter = SANSFitter.load_analysis('silica_35C.json', data='silica_45C.dat')

# An in-memory dataset (required for an analysis saved from set_data)
fitter = SANSFitter.load_analysis('difference.json', data=data_ops.subtract(a, b))
```

An analysis saved with `include_result=False` is a template: a configured model
with no sample-specific outcome, ready to apply to the next dataset.

!!! note "Custom models"
    By default only models built into sasmodels are loaded. Loading a
    `custom.<name>` expression imports a plugin module named by the file, which
    is code execution chosen by whoever wrote it, so it needs
    `load_analysis(path, allow_custom_models=True)`.

#### Reports

`report()` renders one document holding the settings, the fit tables and the
plot:

```python
fitter.report('fit.html')     # self-contained page, opens in any browser
fitter.report('fit.md')       # for an issue, a pull request or a logbook
doc = fitter.report()         # renders in a notebook; str(doc) is the Markdown
```

Before any fit this produces a configuration report, showing the settings and
the current parameter values with a theory preview in place of the fit plot.

HTML embeds the interactive Plotly figure and needs nothing beyond the standard
dependencies; `offline=True` inlines the library for a file that needs no
network. Markdown references a PNG next to the report, named after it, which
has to be rasterized: install the optional extra with
`pip install "sans-fitter[report]"` (it also needs a compatible Chrome on the
machine). Without it the report is still written, without the figure, and says
so.

## Advanced Usage

### Resolution (Smearing)

Instrument resolution changes the fitted parameters, so SANS-fitter treats it
as a stated choice rather than a property of the input file. The four modes
mirror SasView's Fit Page (*None* / *Use dQ Data* / *Custom Pinhole* /
*Custom Slit*).

```python
fitter.set_resolution('data')                      # default: the file's own columns
fitter.set_resolution('none')                      # perfect resolution
fitter.set_resolution('pinhole', dq_over_q=0.10)   # constant relative width
fitter.set_resolution('slit', slit_length=0.05)    # constant slit geometry

fitter.get_resolution()
# {'mode': 'pinhole', 'dq_over_q': 0.1, 'slit_length': None, 'slit_width': None}
```

The setting reaches `fit(engine='bumps')`, `fit(engine='lmfit')`,
`fit_bayesian()` and the post-fit curves through one shared evaluation copy of
your dataset. **`fitter.data` is never modified**: plots, CSV export, P(r)
inversion and `data_ops` all keep seeing the dataset you loaded.

`get_resolution()` works before any data is loaded, and the mode **persists**
across `load_data()` / `set_data()`, exactly as the model and the parameters
do. The summary printed on load reports the active mode, so a custom width
cannot reach a new dataset unnoticed.

#### What each mode does

| Mode | Dataset has | Result |
|---|---|---|
| `'data'` | a real `dQ` column | pinhole smearing from that column |
| `'data'` | slit columns only (`dxl`/`dxw`) | slit smearing from those columns |
| `'data'` | no resolution columns | warns, evaluates unsmeared |
| `'none'` | anything | unsmeared, file columns ignored |
| `'pinhole'` | anything | `dx = dq_over_q · q` |
| `'slit'` | anything | constant `dxl` (and optional `dxw`) |

A dataset carrying both a `dQ` column and slit columns warns: sasmodels gives
`dQ` priority and ignores the slit columns. A dataset carrying a slit *width*
with no slit *length* is refused with an explanatory error rather than being
smeared wrongly or silently. See the note on slit conventions below.

#### Units and conventions

`dq_over_q` is **σ_q/q, a Gaussian 1-σ** - the same quantity as the file's
`dQ` column and as `examples.simulate(dq=...)`. **It is not FWHM.** If your
instrument scientist quotes ΔQ/Q as a full width at half maximum, divide by
about 2.355 first.

`slit_length` (sasmodels' `dxl`, along q) and `slit_width` (`dxw`,
perpendicular) are **absolute** widths in Å⁻¹, constant across every Q point.
`slit_length` is required: sasmodels does not implement smearing from a slit
width alone, so `slit_width` refines a real slit rather than describing one on
its own. Omit it for the usual long-slit (USANS) geometry.

Out of scope: per-point custom widths, constant *absolute* σ_q, 2D/oriented
resolution and fittable resolution parameters.

#### Pairing simulated data with a mode

`examples.simulate(dq=...)` both smears the simulated intensity and attaches a
`dQ` column, so it behaves like a measurement. Fitting the wrong mode against
it double-smears or under-smears.

| Data produced by | Fit with |
|---|---|
| `simulate(...)` without `dq`, or a file with no `dQ` | `'none'`, or a `'pinhole'`/`'slit'` you supply |
| `simulate(..., dq=0.1)`, or a file with `dQ` | `'data'` (the default) |
| a file with `dQ` you want to ignore | `'none'` |

#### Things to know

- **χ² across modes is evidence, not a dial.** Changing the mode changes the
  forward model and nothing else - same points, same `dI`, same free
  parameters - so a χ² that drops really does mean the data prefers that
  smearing. Do not go looking for the mode that minimises it, though:
  resolution is a property of the instrument, and smearing is degenerate with
  real physics (polydispersity broadens a form-factor minimum much as
  resolution does), so tuning it absorbs sample physics into an instrument
  setting. Use the mode the beamline actually had, and read χ² as a check on
  it.
- **The exported `dQ` column is the file's own**, not the width the fit used.
  `save_results()` writes a `# Resolution mode:` header line recording what was
  actually applied, and `plot_results()` draws horizontal error bars from the
  file's column.
- **Smearing costs time.** Pinhole and slit resolution build a weight matrix
  and evaluate the model on an extended Q grid; this is a one-time cost per
  fit, but a noticeable one on large datasets.
- **P(r) inversion is unaffected.** `pr_inversion` reads `fitter.data`, which
  this setting deliberately leaves alone: `set_resolution('none')` will not
  change a P(r) result, and `'slit'` will not make `invert()` slit-aware.
- **Dataset arithmetic.** Under `'data'`, a background-subtracted result
  inherits the combined-`dQ` caveat noted under *Dataset Operations*.
  `'pinhole'`, `'slit'` and `'none'` are the way to override it.

See `examples/resolution_example.py` and
`notebooks/resolution_control.ipynb` for a complete walkthrough.

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
fitter.get_q_range()  # -> (qmin, qmax)
fitter.reset_q_range()  # back to the full data range
```

The restriction applies to both fitting engines. Excluded points still
appear in plots (grayed out, labelled "Excluded Data"), but the fitted
curve, residuals, χ², and the CSV export only cover the fitted range, and
`n_points` counts those points.
The range can be changed freely between fits. Each fit result remembers
the range it was fitted with.

### Dataset Operations

The `sans_fitter.data.ops` module manipulates datasets with arithmetic
operations - similar to SasView's *Data Operation* utility. Typical uses are
background subtraction, rescaling to absolute units, and transmission
correction.

```python
from sans_fitter import SANSFitter, data_ops

sample = data_ops.load('sample.csv')  # standalone loader, returns Data1D
background = data_ops.load('empty_cell.csv')

net = data_ops.subtract(sample, background)  # sample − background
net = data_ops.divide(net, 0.8)  # transmission correction

fitter = SANSFitter()
fitter.set_data(net)  # inject the in-memory dataset
fitter.set_model('sphere')
fitter.fit()
```

Available operations - each returns a new, fit-ready `Data1D`:

| Function | Result |
|---|---|
| `data_ops.add(a, b)` | `a + b` |
| `data_ops.subtract(a, b)` | `a − b` (order matters) |
| `data_ops.multiply(a, b)` | `a × b` |
| `data_ops.divide(a, b)` | `a / b` (order matters) |

The second operand can be a dataset or a scalar. For two datasets,
uncertainties are propagated (`dI = sqrt(dI_a² + dI_b²)` for add/subtract,
relative errors in quadrature for multiply/divide) and both must share the
same Q grid (x-values matching within 1% - interpolation onto a common grid
is not yet supported). For a scalar, `multiply`/`divide` scale both `I` and
`dI`, while `add`/`subtract` shift `I` and leave `dI` unchanged; the Q grid
is never altered.

Every result records its provenance: the title becomes the operation (e.g.
`"sample.csv - empty_cell.csv"`) and a `Process` entry is appended, which
survives in saved CanSAS output.

Things to know:

- **Missing dI** on an operand triggers a warning: it is treated as zero in
  error propagation. Error-free data warns again at fit time: the `lmfit`
  engine falls back to unit weights, while `bumps` refuses to fit.
- **NaN points** propagate through the arithmetic and are masked in the
  result (excluded from fits); a warning reports the masked count.
- **Resolution (dQ) propagation** through arithmetic is not validated
  upstream. A warning is emitted when any operand carries resolution data.
  Treat resolution on results with care, especially for slit-smeared data.

`SANSFitter.set_data()` accepts any sasdata `Data1D` - arithmetic results,
simulated data, or datasets built programmatically - and validates and
normalizes it so it is fit-ready.

See `examples/data_operations_example.py` and
`notebooks/data_operations_demo.ipynb` for a complete walkthrough.

### Simultaneous Fitting of Several Datasets

Contrast variation, a temperature or concentration series, and one sample
measured on several instrument configurations all need the same thing: one
model fitted to N datasets with some parameters shared and some kept separate.
`MultiFitter` does that through a single joint optimization rather than a
sequence of independent fits.

```python
from sans_fitter import MultiFitter

fit = MultiFitter()
fit.add('h2o', 'contrast_h2o.xml', model='sphere')
fit.add('d2o', 'contrast_d2o.xml', model='sphere')

for name in ('h2o', 'd2o'):
    fit[name].set_param('radius', value=45, min=10, max=100, vary=True)
    fit[name].set_param('scale', value=0.02, min=0.001, max=0.1, vary=True)
    fit[name].set_param('background', value=0.01, min=0, max=0.1, vary=True)

fit.share('radius', 'scale')             # one radius and scale for both curves
fit.constrain('h2o.sld_solvent', -0.56)  # the known solvent contrasts
fit.constrain('d2o.sld_solvent', 6.34)

fit.describe()                 # datasets, relationships, free-parameter count
result = fit.fit()
print(fit.get_fit_report())
fit.plot_results()
```

Parameters are addressed as `dataset.parameter`. Three relationships are
available — `share()` for one quantity held by several datasets,
`link_params()` for one parameter following another, and `constrain()` for a
constant or an arithmetic expression such as `'0.8 * h2o.scale'`. Each dataset
keeps its own Q range, mask and resolution; nothing is interpolated or merged,
and a shared parameter costs exactly one degree of freedom.

See the [Simultaneous Fitting](multifit.md) guide for the full picture:
choosing what to share, how expression bounds are enforced, what dataset
weights do to the reported uncertainties, and the CSV export. A runnable
walkthrough is in `examples/simultaneous_fitting_example.py` and
`notebooks/simultaneous_fitting.ipynb`.

### P(r) Inversion

The `sans_fitter.inversion` module recovers the real-space pair distance
distribution function P(r) from I(q) by indirect Fourier transform (Moore's
sine-basis expansion, as in SasView's Inversion perspective). It is
**model-free** - no sasmodels kernel is involved - and operates directly on
datasets (`fitter.data` or `data_ops` results). Typical use: monodisperse
protein solutions, where P(r) yields D_max, Rg and I(0) without assuming a
form factor.

For buffer-subtracted data (the usual protein case), pass
`fit_background=False`. The default fitted flat background can absorb I(0)
and bias Rg on already-subtracted data. Explore D_max **before** trusting an
inversion: every result is conditional on it.

```python
from sans_fitter import data_ops, pr_inversion

data = data_ops.load('protein.csv')

# 1. Find a stable D_max: look for the Rg/I(0) plateau and chi2 minimum
scan = pr_inversion.explore_dmax(data, d_max=120.0, fit_background=False)
scan.plot()  # or scan.plot(quantity='all'), scan.format_summary()

# 2. One-shot inversion with automatic selection of n_terms and alpha
result = pr_inversion.auto_invert(data, d_max=120.0, fit_background=False)
print(result.format_summary())  # Rg, I(0), oscillations, positivity, diagnostics

# 3. Plots and export
result.plot_pr()  # P(r) with its 1-sigma band
result.plot_fit(data)  # data vs fit, residuals (data passed explicitly)
result.save_csv('pr_result.csv')
```

Explicit control is available through the individual functions:

| Function | Result |
|---|---|
| `invert(data, d_max, n_terms=10, alpha=0.0, fit_background=True, background=0.0, r_points=101, regularizer='corrected')` | Core inversion → `PrResult` |
| `estimate_n_terms(data, d_max, fit_background=True, ..., background=0.0)` | `NTermsEstimate(n_terms, alpha, message)`; its `alpha` is authoritative: use it directly |
| `estimate_alpha(data, d_max, n_terms, fit_background=True, ..., background=0.0)` | `AlphaEstimate(alpha, message)` |
| `auto_invert(data, d_max, ...)` | `estimate_n_terms` → `invert`, silent |
| `explore_dmax(data, d_max, ..., refit_alpha=False, background=0.0)` | `DmaxScan` over 0.9–1.1×d_max (25 points); raises when every point fails |

When working with a known fixed background (`fit_background=False`), pass the
same `background` value to the estimators and `explore_dmax` too. The
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
  `uncertainties_fabricated` flag on the result. Chi-squared diagnostics are
  then not interpretable.
- **Q range is honoured**: the inversion uses the same accepted-point rule as
  the fit engines, so `fitter.set_q_range()` restricts it identically.
- **Shannon limits are checked**: warnings fire when `d_max > pi/q_min` or
  `n_terms` exceeds `q_max*d_max/pi` (the data cannot support either).
- **Slit smearing is not supported** (a warning fires on slit-smeared data);
  pinhole dQ resolution is ignored, as in SasView.
- **`regularizer='sasview'`** reproduces SasView's exact smoothing operator
  for comparison; the default `'corrected'` penalizes the true second
  derivative on a resolved grid (validated against SasView: identical for
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

The available structure factors are **queried from sasmodels at runtime** (not
hardcoded), so new ones added upstream are picked up automatically. List them
with `get_structure_factors()`:

```python
from sans_fitter import get_structure_factors

get_structure_factors()
# e.g. ('hardsphere', 'hayter_msa', 'squarewell', 'stickyhardsphere', 'two_yukawa')
```

As of the current sasmodels install this includes `hardsphere`, `hayter_msa`,
`squarewell`, `stickyhardsphere`, and `two_yukawa`.

### Effective Radius

When using a structure factor, you often need to define an effective radius. You can link this to the form factor's radius.

```python
# Link effective radius to the sphere radius
fitter.set_structure_factor('hardsphere', radius_effective_mode='link_radius')
```

This is an ordinary parameter link (see [Linking parameters](#combining-models-composite-models)):
`get_links()` reports it as `{'radius_effective': 'radius'}`, `radius_effective`
is held at `radius` throughout the fit, and writing to it directly raises. Pass
`radius_effective_mode='unconstrained'` (the default) to fit it independently.

### Combining Models (Composite Models)

Datasets with several distinct features - for example a low-Q diffuse
scattering contribution plus a high-Q correlation peak - are often best
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

```text
I(q) = scale · [dab_scale·I_dab(q) + peak_lorentz_scale·I_peak(q)] + background
```

The global `scale` and `background` are shared by every component natively
(sasmodels' mixture semantics), while each component carries its own
`<name>_scale`. Varying the global `scale` together with a component scale is
degenerate - only their product is fitted - so `fit()` warns when both are
free. With `operation='*'` the part intensities multiply instead.

**Friendly parameter names.** Every component parameter is prefixed with the
model name (`dab_cor_length`, `peak_lorentz_peak_pos`). Give components custom
names (monikers) with keyword arguments - useful for long model names,
duplicates, or physics labels:

```python
fitter.set_models(small='sphere', large='sphere', shared=['sld', 'sld_solvent'])
fitter.set_param('small_radius', value=20, min=5, max=100, vary=True)
fitter.set_param('large_radius', value=200, min=50, max=1000, vary=True)
fitter.set_param('sld', value=4.0, vary=True)  # one knob drives both spheres
```

**Sharing parameters.** Each name in `shared=[...]` must exist in at least
two components; it becomes a single unprefixed parameter driving all of them
(the per-component versions disappear from the parameter list). This is the
one-line answer to "share SLD across models". Note that polydispersity
configuration stays per-component: after `shared=['radius']`,
`set_pd_param('small_radius', ...)` and `set_pd_param('large_radius', ...)`
still configure the two components independently.

**One component per entry.** Each `set_models()` entry must be a single
component (optionally with `@`, see below). An entry that is itself a
composite expression - e.g. `set_models(diffuse='dab+peak_lorentz',
particle='sphere')` - raises an error, because the two entries would expand
to three kernel components and the monikers could not map 1:1. Pass each
component separately, or use the raw string path
(`set_model('dab+peak_lorentz+sphere')`) with canonical `A_`/`B_`/`C_` names.

**Structure factors on one part.** A component entry may itself contain `@`,
applying a structure factor to that part only:

```python
fitter.set_models('sphere@hardsphere', 'peak_lorentz')
```

(`@` binds tighter than `+`, so this is `(sphere@hardsphere) + peak_lorentz`.)
Applying `set_structure_factor()` to an already-composite model raises an
error: sasmodels cannot express `(A+B)@S`.

**Component curves.** After fitting a `'+'` mixture,
`plot_results(show_components=True)` overlays one dashed curve per component,
each drawn as `scale · part_scale · I_part(q)` (background excluded, shown
implicitly in the total curve). For `'*'` mixtures and atomic models the flag
is a documented no-op.

**Equality links.** For sharing that `shared=` cannot express - linking only
some components, or parameters with different names - use explicit links:

```python
fitter.link_params('large_sld', to='small_sld')  # follower mirrors target
fitter.link_params('shell_sld_core', to='small_sld')  # different names work too
fitter.unlink_params('large_sld')  # escape hatch
```

A follower is forced `vary=False` and mirrors the target's value before,
during, and after the fit; writing it directly raises. It also inherits the
target's uncertainty, because the link makes the two one quantity. Link chains
are not supported.

To relate parameters *across datasets* rather than within one model, see
[Simultaneous Fitting](multifit.md), where the same vocabulary — `share()`,
`link_params()` and `constrain()` — works on qualified `dataset.parameter`
names.

**Raw string syntax (advanced).** `set_model()` accepts sasmodels' native
composite expressions directly and keeps the canonical `A_`/`B_` parameter
names - zero magic when following sasmodels documentation:

```python
fitter.set_model('dab+peak_lorentz')  # A_scale, A_cor_length, B_scale, ...
```

Every atomic name in the expression is validated before loading, with a
nearest-match suggestion for typos.

**Engine support.** Composite models currently work with the `bumps` engine
only; `fit(engine='lmfit')` and `fit_bayesian()` raise `NotImplementedError`
when one is active. Parameter links themselves work with every engine.

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
    print(f'Polydisperse parameters: {pd_params}')
```

### Configuring Polydispersity

Configure polydispersity for a specific parameter:

```python
# Set polydispersity width (relative, 0.0 = monodisperse, 0.1 = 10% width)
fitter.set_pd_param('radius', pd_width=0.1)

# Configure all PD options
fitter.set_pd_param(
    'radius',
    pd_width=0.15,  # 15% polydispersity
    pd_n=50,  # Number of quadrature points (default: 35)
    pd_nsigma=4.0,  # Number of sigmas to include (default: 3.0)
    pd_type='gaussian',  # Distribution type
    vary=True,  # Allow pd_width to vary during fitting
)

# Get current PD configuration
pd_config = fitter.get_pd_param('radius')
print(
    pd_config
)  # {'pd': 0.15, 'pd_n': 50, 'pd_nsigma': 4.0, 'pd_type': 'gaussian', 'vary': True, 'active': True}
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
    print('Polydispersity is enabled')

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
Carlo sampler (via BUMPS, which is already a dependency - no extra
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

`fit_bayesian()` prints the usual fit report plus a posterior
table with the mean, median, standard deviation, 68%/95% credible
intervals, and convergence diagnostics (R-hat, effective sample size) for
each sampled parameter. The `cov` on a Bayesian result is the **sample**
covariance of the posterior draw rather than a Jacobian estimate
(`cov_source` reads `'posterior sample'`), and the `message` carries the
sampler settings and the largest R-hat. The reported parameter values are the best
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
fitter.plot_posterior_predictive()  # band only
fitter.plot_posterior_predictive(style='band+draws')  # band + sampled curves
fitter.plot_posterior_predictive(n_draws=100)  # more model evaluations

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

posterior.labels  # sampled parameter names (chain order)
posterior.samples  # ndarray [n_samples, n_params]
posterior.ci_95  # {name: (low, high)} 95% credible intervals
posterior.diagnostics  # {name: {'r_hat': ..., 'ess': ...}}

print(posterior.format_summary())

# Export the raw chain for external analysis (e.g. corner, arviz, pandas)
posterior.save_posterior_csv('posterior_chain.csv')
```

`save_results()` also includes the credible intervals in the CSV header
after a Bayesian fit.
