# Simultaneous Fitting

Contrast variation, a temperature series, one sample measured on three detector
distances — these all have the same shape. Most of the physics is common to
every curve, a few parameters are not, and fitting the curves one at a time
throws away the constraint that makes the common part identifiable.

`MultiFitter` fits them together: several named datasets, each with its own
model, Q range, resolution and parameters, minimised as one problem with
parameters shared or related between them.

```python
from sans_fitter import MultiFitter

fit = MultiFitter()
fit.add('h2o', 'contrast_h2o.xml', model='sphere')
fit.add('d2o', 'contrast_d2o.xml', model='sphere')

for name in ('h2o', 'd2o'):
    entry = fit[name]
    entry.set_param('radius', value=45, min=10, max=100, vary=True)
    entry.set_param('scale', value=0.02, min=0.001, max=0.1, vary=True)
    entry.set_param('background', value=0.01, min=0, max=0.1, vary=True)
    entry.set_param('sld', value=4.0, vary=False)

fit.share('radius', 'scale')             # one radius, one scale, both curves
fit.constrain('h2o.sld_solvent', -0.56)  # solvent contrast is known
fit.constrain('d2o.sld_solvent', 6.34)

fit.describe()                 # what will be fitted, and how many free parameters
fit.plot_model()               # starting point, with the constraints applied
result = fit.fit()
print(fit.get_fit_report())
fit.plot_results()
fit.save_results('contrast_results')
```

## Addressing parameters

Every parameter is `dataset.parameter`:

```python
fit.set_param('h2o.radius', value=48)
fit['h2o'].set_param('radius', value=48)   # the same thing
```

A polydispersity width is its own quantity, spelled `dataset.parameter_pd`.
An unqualified name works only where it identifies exactly one shared group,
which is the usual case after `share('radius')`:

```python
fit.share('radius')
fit.set_param('radius', value=48)          # updates every member
```

Dataset names must be plain identifiers (`h2o`, `run_2`, `cold`) because they
appear unquoted on the left of the dot.

## The three relationships

### `share()` — one quantity, several datasets

Symmetric. The members must already agree on their starting value and vary
flag, or you name the one that wins:

```python
fit.share('radius', 'radius_pd')                       # across every dataset
fit.share('scale', datasets=['h2o', 'd2o'])            # across a subset
fit.share('radius', source='h2o')                      # h2o's settings win
```

Insertion order deliberately does not decide. Which dataset's starting radius
the fit uses is a scientific choice, not a consequence of the order two files
were loaded in, so a disagreement is an error rather than a silent pick.

The shared bounds are the **intersection** of the members', so a limit set on
any member still applies. Membership is fixed when you call `share()`: a dataset
added afterwards is not enrolled.

`unshare('radius', dataset='d2o')` detaches one member. It keeps the value it
currently holds and gets its own bounds and vary flag back.

### `link_params()` — one parameter follows another

Directed, and the right tool when the parameters have different names or live
in different models:

```python
fit.link_params('cyl.radius', to='sphere.radius')
```

The follower adopts the target's value, vary flag **and bounds**; its own
settings are discarded. That is the difference from both `share()`, which treats
the members as equals and intersects their bounds, and `constrain()`, which
defines only the target's value and leaves its limits in force. Reach for
`link_params()` when the follower's own configuration is not meant to survive.

### `constrain()` — a constant or an expression

```python
fit.constrain('d2o.sld_solvent', 6.34)             # a known value
fit.constrain('d2o.background', 'h2o.background')  # equality
fit.constrain('d2o.scale', '0.8 * h2o.scale')      # a known ratio
fit.constrain('warm.length', '2 * cold.radius + 10')
```

The grammar is deliberately small: numbers, qualified references, `+ - * /`,
and integer powers. No function calls, no indexing, no bare names. Expressions
are **interpreted, never executed** — the text is parsed and walked against an
allowlist, so a constraint is data rather than code.

`unconstrain('d2o.scale')` removes it. The parameter keeps the value the
constraint last resolved to and gets its bounds and vary flag back. The same
holds for `unshare()` and `unlink_params()`: a detached parameter keeps the
value the relationship gave it rather than reverting to whatever it happened to
be configured with beforehand.

A parameter can have one definition, not two. Constraining a parameter that
already follows another through `link_params()` is refused — unlink it first —
and so is linking one that is already constrained. Constraining the same
parameter twice simply replaces the definition.

#### Why a constraint can be refused

Binding an expression replaces the target's parameter object inside bumps,
which discards the target's own limits. Rather than let a limit quietly stop
applying, `MultiFitter` proves it before the fit: a constraint is accepted only
if it cannot leave the target's bounds over the ranges its inputs are allowed to
explore.

Where the relationship is linear in one parameter that proof is *constructive* —
the input's range is narrowed so the target is guaranteed:

```python
fit['d2o'].set_param('scale', min=0.0, max=0.03)
fit.constrain('d2o.scale', '2 * h2o.scale')
# h2o.scale is now restricted to at most 0.015, whatever its own max said
```

Where the arithmetic cannot certify it — a divisor whose range straddles zero,
a nonlinear expression whose interval bound is too loose — the constraint is
refused and the message names the parameters to tighten. This is conservative:
some feasible nonlinear relationships need tighter bounds than they strictly
must. It is preferred to a bound that silently stops being enforced.

**Every form of `constrain()` enforces the target's limits**, including a bare
reference and arithmetic over literals alone (`'2 * 3.17'` is a constant and
behaves as one everywhere — in the parameter table, in the fit, and in the
export). Two spellings of one relationship always permit the same values.

Repeated references cancel exactly, so `h2o.radius - d2o.radius` is recognised
as zero once the two are shared, rather than being treated as two independently
varying quantities.

## Counting parameters

The number to watch is the free-parameter count, and `describe()` states it:

```text
MultiFitter: 2 dataset(s), 4 free parameter(s)

  h2o            sphere                      60 pts  weight 1  Q [0.005, 0.5]  data
  d2o            sphere                      45 pts  weight 1  Q [0.008, 0.3]  data

Parameters taking part in the fit:
  d2o.background          0.02  free
  d2o.radius                40  shared
  h2o.background          0.02  free
  h2o.radius                40  shared  [= d2o.radius]

Free parameters: d2o.background, d2o.radius, h2o.background, ...
```

A shared quantity is counted **once**. Its label is the alphabetically first
member, chosen that way so it does not move when datasets are added or renamed;
`share(..., source=...)` picks it explicitly if you want a particular one.

Each parameter has one of four statuses:

| Status | Meaning | Uncertainty |
|---|---|---|
| `free` | An independent coordinate the optimizer moved. | From the joint covariance. |
| `shared` | The same quantity as one or more others; one coordinate between them. | The same as its root's — an equality makes them one number. |
| `derived` | Computed from other parameters by a constraint. | Propagated through the constraint, keeping cross-parameter covariance. |
| `fixed` | Not fitted: either `vary=False` or pinned by a constant constraint. | Exactly zero. |

A standard error of `None` means the uncertainty could **not be estimated**.
That is different from a fixed parameter's zero, and the two are never
conflated.

### When the parameters are not separately identifiable

If the free parameters are not independently determined by the data — two
quantities whose sum is fitted but whose split is not — the covariance is rank
deficient, and `result.cov_note` says so in every rendering of the report. What
happens next depends on the weighting, and the difference is deliberate:

- **Unit weights** keep the estimate bumps produces, which clamps singular
  values and so returns a very large variance rather than failing. Diverging
  from `SANSFitter` here would be a worse surprise than a large error bar, so
  the number is kept, a `RuntimeWarning` is raised, and the note explains it.
- **Priority weights** have no such fallback to inherit, so there is no
  covariance at all and every standard error is `None`.

Either way the diagnosis is in `cov_note`. An implausible error bar is never
left to speak for itself.

## What gets minimised

For each dataset `d`, with `J_d` its selected points:

```text
r_dj  = [model_d(Q_dj) - I_dj] / sigma_dj
chi2_d = sum over J_d of r_dj^2
objective = sum over d of a_d * chi2_d
N = sum_d N_d,  P = number of free parameters,  dof = N - P
```

The residual sign is bumps' own, `(model - data)/sigma`, and exported residuals
say so in their header.

There is no requirement that the datasets share a Q grid. Only the residual
vectors are concatenated — nothing is interpolated, merged or rebinned, and each
dataset keeps its own resolution kernel, mask and Q interval. That is the point:
fitting the original curves is what you want, and it is why data merging is not
a prerequisite.

### Per-dataset diagnostics

The report gives each dataset its point count, its raw χ², its contribution to
the objective, its weight, Q range and resolution. It deliberately does **not**
give a per-dataset reduced χ²: the degrees of freedom belong to the joint fit
and cannot be divided between datasets. The per-point figure it does show,
χ²/N, is labelled as a mean squared normalized residual.

### Dataset weights

By default every `a_d` is 1, so every valid observation counts according to its
own uncertainty and the total is a χ². A dataset with more points or smaller
error bars legitimately carries more information, and nothing rebalances that
automatically.

`set_dataset_weight('d2o', 0.5)` states a **fitting priority**: it changes which
compromise the optimizer prefers. It does not change the error model — the
supplied `dI` is still what the uncertainties mean. Two consequences follow, and
the report labels both:

- `chisq` (the goodness of fit) and `objective` (what was minimised) become
  different numbers. The raw χ² at an artificially weighted optimum is not at
  its own minimum.
- uncertainties come from the **known-error sandwich covariance**
  `H⁻¹ (Jᵀ W² J) H⁻¹`, not from the inverse curvature of the reweighted
  objective. The inverse curvature would be the right answer only if `dI/√a`
  were the real error bars, which is not what the weights mean here. The
  sandwich is unchanged when every weight is multiplied by one constant; the
  inverse curvature is not.

Weights must be positive and finite. To leave a dataset out, remove it rather
than weighting it to zero.

### The independence assumption

The joint χ² assumes independent Gaussian point errors. Overlapping Q ranges
from separate measurements are fine. Duplicate observations, or datasets sharing
a measured subtraction background, are not independent — counting one
measurement twice makes the uncertainties optimistic. `MultiFitter` warns when
two datasets hold identical data; that case is intentional when comparing two
models on one dataset, but the comparison is better done as separate fits.

## Choosing what to share

Sharing is a physical claim, not a convenience, and same-named parameters are
not automatically the same quantity.

| Parameter | Share when |
|---|---|
| Geometry (`radius`, `length`, `thickness`) | The particles are the same. This is the usual reason for a joint fit. |
| Polydispersity width | The size distribution is the same. Requires the same distribution type on every entry. |
| `scale` | The concentration *and* the normalization are the same. Contrast variation with matched samples, yes; a concentration series, no. |
| `sld` (particle) | The material is the same and the contrast difference lives entirely in the solvent. |
| `sld_solvent` | Almost never across contrasts — that is the variable. Constrain each to its known value. |
| `background` | Rarely. Incoherent background follows the sample composition, and in a contrast series it differs by construction. |

Sharing a polydispersity width shares the width only. Quadrature count,
truncation and distribution type stay per dataset, and relating widths with
different distribution types is refused: 0.15 of a lognormal and 0.15 of a
Schulz describe different distributions.

`share()` also refuses members the models declare in **different units** —
sharing asserts they are one measurement, and no conversion is applied. A
directed `link_params()` across units is still allowed, because relating two
different quantities is exactly what it is for.

## Ownership and stale results

`add()` copies the dataset it is given, so editing the array afterwards cannot
change a configured fit. Reads give copies too — `fit['h2o'].data` is a
snapshot — and every getter that reports a *number* reads it through the
constraint graph, so `fit['h2o'].get_pd_param('radius')['pd']` and the parameter
table cannot disagree about a shared or constrained width.

A `MultiFitResult` is handed back as the same object the fitter retains, and it
is **caller-owned and mutable**: editing its dictionaries or arrays changes the
fitter's view too. Use `copy.deepcopy`, or work from `to_dict()`, if you need
the two to diverge. What it does *not* share is the analysis — the observations,
curves, selection and parameter values are all copied in at fit time.

Every configuration change recompiles the parameter graph. A change that would
leave the analysis inconsistent is rejected **and rolled back**, so the fitter is
never in a state that cannot be fitted:

```python
fit.constrain('h2o.sld_solvent', 6.34)
fit['h2o'].set_model('dab')     # dab has no sld_solvent
# ConstraintError: These references no longer exist: h2o.sld_solvent ...
# The model is still 'sphere' and the constraint is still there.
```

A result records the configuration that produced it. Changing anything
afterwards — a Q range, a weight, a parameter — makes plots, exports and the
report warn that the result no longer describes the analysis. The result object
stays available on `fit.result`; refit to bring the two back together.

## Export

`save_results(directory)` writes:

| File | Contents |
|---|---|
| `parameters.csv` | Every parameter with its value, uncertainty, status, root and constraint. |
| `datasets.csv` | Per-dataset point counts, χ², objective contribution, weight, Q range, resolution. |
| `covariance.csv` | The joint covariance over the free parameters, with labels. |
| `<dataset>_curve.csv` | Q, I, dI, fitted I, residual and objective residual, for the fitted points. |
| `manifest.txt` | What the files belong to, and which covariance convention was used. |

The exported contributions reconstruct the reported totals exactly: summing the
squared residuals across the curve files gives `chisq`, and summing the squared
objective residuals gives `objective`. Every row is internally consistent too —
`Residual` always equals `(I_fit - I_exp) / dI_exp` computed from that same row
— because the export reads the result's snapshot rather than the live datasets.

Files are rendered before anything is written and replaced atomically, so a
failure part-way cannot mix this export with the last one. A second export into
the same directory retires the files the previous one recorded in its manifest;
a `covariance.csv` beside a fit that has none would otherwise read as this
fit's. Only files a previous export listed are ever removed, so anything else in
the directory is left alone.

## Plots

`plot_results()` draws entirely from the result's own snapshot — the
observations, the selection and the model curves it was built with — so the
figure shows one coherent fit even if a dataset has since been reconfigured or
removed. `plot_model()` reads the datasets as they are now, which is the point
of a preview.

Both stack one panel per dataset, each with its own axes and its own residual
panel underneath. Stacking rather than overlaying
is the default because datasets in a joint fit routinely differ by orders of
magnitude in intensity — which is often *why* they are being fitted together.

Residuals are in the dataset's own sigma units. `objective_residuals=True` shows
`√weight · residual` instead, labelled differently so the two cannot be
confused. `sans_fitter.multi_plotting.plot_multi(..., overlay=True)` puts
comparable curves on one pair of axes.

For a `'+'` mixture model, `show_components=True` overlays one dashed curve per
component beneath the total, as the single-fit plots do. The curves are on
`result.datasets[name].component_curves`, and stack onto the fitted curve
together with the background.

## Migration and limits

`MultiFitter` is additive: `SANSFitter` keeps its API and its optimization
behaviour. One deliberate change reaches single-dataset fits — an equality
follower (`link_params`, `shared=`, `radius_effective_mode='link_radius'`) now
reports its target's uncertainty instead of zero. An equality link makes the two
parameters one quantity, so the follower's error *is* the target's. Genuinely
fixed parameters still report zero, and `fixed=True`/`linked_to` still
distinguish the two cases. Analyses saved before this change are normalized on
load where the target's error is still in the file.

Not yet supported, and intentionally so until each has its own numerical tests:

- **Engines other than bumps.** `engine='lmfit'` raises rather than silently
  running something untested.
- **Joint DREAM sampling.** The same problem would sample, but the posterior
  predictive path needs a multi-dataset evaluator.
- **Saving an analysis to JSON.** `save_results()` exports the outcome;
  `save_analysis()` for a joint analysis is planned once the result schema has
  settled. Keep the configuration in a notebook or script meanwhile.
- **Inequality constraints, fitted error scales, automatic rebalancing, 2D and
  SESANS data, batch fitting** (which is a different operation — N independent
  fits, not one).
