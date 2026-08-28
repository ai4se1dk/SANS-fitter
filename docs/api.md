# API Reference

## Module-level helpers

::: sans_fitter.fitter.get_all_models
::: sans_fitter.fitter.get_structure_factors

## SANSFitter

The main class for SANS data fitting.

::: sans_fitter.fitter.SANSFitter
    options:
      show_root_heading: true
      show_source: true
      members:
        - load_data
        - set_data
        - set_q_range
        - reset_q_range
        - get_q_range
        - set_model
        - set_models
        - link_params
        - unlink_params
        - get_links
        - get_components
        - set_structure_factor
        - get_structure_factor
        - remove_structure_factor
        - get_params
        - set_param
        - fit
        - fit_bayesian
        - get_posterior
        - plot_results
        - plot_posterior_pairs
        - plot_param_distribution
        - plot_posterior_predictive
        - plot_param_correlations
        - plot_trace
        - save_results
        - supports_polydispersity
        - get_polydisperse_parameters
        - set_pd_param
        - get_pd_param
        - enable_polydispersity
        - is_polydispersity_enabled
        - get_pd_params
        - get_varying_pd_params

### Composite model naming

`set_models()` exposes parameters under friendly alias names; `set_model()`
with a composite expression keeps sasmodels' canonical names. Both spellings
are accepted by `set_param()`, `link_params()`, `set_pd_param()` and
`get_pd_param()`. For `set_models('dab', 'peak_lorentz')`:

| Friendly alias (`set_models`) | Canonical name (`set_model`) |
|---|---|
| `scale`, `background` | `scale`, `background` (shared natively) |
| `dab_scale` | `A_scale` |
| `dab_cor_length` | `A_cor_length` |
| `peak_lorentz_scale` | `B_scale` |
| `peak_lorentz_peak_pos` | `B_peak_pos` |
| `peak_lorentz_peak_hwhm` | `B_peak_hwhm` |

With keyword monikers (`set_models(small='sphere', large='sphere')`) the
prefix is the moniker (`small_radius` → `A_radius`). Parameters listed in
`shared=` collapse to a single unprefixed name (`sld` → `A_sld` + `B_sld`);
their prefixed aliases remain addressable for polydispersity configuration.

## PosteriorSummary

Posterior sample chain and per-parameter statistics returned by `fit_bayesian()`.

::: sans_fitter.results.PosteriorSummary
    options:
      show_root_heading: true
      show_source: true
      members:
        - index_of
        - format_summary
        - save_posterior_csv

## data_ops

Dataset arithmetic: add, subtract, multiply and divide datasets (or a dataset
and a scalar) with propagated uncertainties, returning fit-ready `Data1D`
objects.

::: sans_fitter.data.ops
    options:
      show_root_heading: true
      show_source: true
      members:
        - load
        - add
        - subtract
        - multiply
        - divide


## pr_inversion

Model-free P(r) inversion (indirect Fourier transform): recover the pair
distance distribution function from I(q), with automatic parameter
estimation and D_max exploration.

::: sans_fitter.inversion
    options:
      show_root_heading: true
      show_source: true
      members:
        - invert
        - auto_invert
        - estimate_alpha
        - estimate_n_terms
        - explore_dmax
        - PrResult
        - DmaxScan
        - AlphaEstimate
        - NTermsEstimate
        - InsufficientDataError
        - PrEstimationError
        - DEFAULT_N_TERMS
        - DEFAULT_R_POINTS
        - REGULARIZERS
        - SASVIEW_N_REG

## examples

Curated example datasets and a simulator for generating data.
See the [Example Data](examples.md) guide for the full collection.

::: sans_fitter.examples
    options:
      show_root_heading: true
      show_source: true
      members:
        - list_examples
        - describe
        - get_example
        - example_path
        - load
        - load_fitter
        - simulate
        - simulate_pair
        - Example


## ParameterManager

Internal class for managing model parameters and polydispersity settings.

::: sans_fitter.modeling.parameters.ParameterManager
    options:
      show_root_heading: true
      show_source: true
      members:
        - initialize_parameters
        - get_parameter_names
        - get_parameter
        - set_parameter
        - get_polydisperse_parameters
        - has_polydisperse_parameters
        - set_pd_param
        - get_pd_param
        - toggle_pd_visibility
        - is_pd_enabled
        - get_pd_params_for_fitting
        - display_pd_params

## PolydispersityManager

Internal class for managing per-parameter polydispersity state.

::: sans_fitter.modeling.polydispersity.PolydispersityManager
    options:
      show_root_heading: true
      show_source: true
      members:
        - initialize
        - get_parameters
        - has_parameters
        - set_param
        - get_param
        - set_enabled
        - is_enabled
        - get_fitting_params
        - get_varying_params
        - display
        - backup
        - restore
        - has_backup
        - clear

## StructureFactorManager

Internal class for managing structure factor selection and parameter backup/restore.

::: sans_fitter.modeling.structure_factor.StructureFactorManager
    options:
      show_root_heading: true
      show_source: true
      members:
        - apply
        - remove
        - backup_params
        - restore_params
        - has_backup
        - clear

::: sans_fitter.modeling.structure_factor.default_parameter_bounds
    options:
      show_root_heading: true
      show_source: true

## Polydispersity Constants

The following constants are available in `sans_fitter.modeling.parameters`:

### PD_DISTRIBUTION_TYPES

```python
PD_DISTRIBUTION_TYPES = ['gaussian', 'rectangle', 'lognormal', 'schulz', 'boltzmann']
```

Supported polydispersity distribution types:

| Type | Description |
|------|-------------|
| `gaussian` | Gaussian/normal distribution (default) |
| `rectangle` | Uniform/rectangular distribution |
| `lognormal` | Log-normal distribution |
| `schulz` | Schulz distribution (common for polymers) |
| `boltzmann` | Boltzmann distribution |

### Default Values

| Constant | Default Value | Description |
|----------|---------------|-------------|
| `DEFAULT_PD_WIDTH` | `0.0` | Default polydispersity width (monodisperse) |
| `DEFAULT_PD_N` | `35` | Default number of quadrature points |
| `DEFAULT_PD_NSIGMA` | `3.0` | Default number of sigmas to include |
| `DEFAULT_PD_TYPE` | `'gaussian'` | Default distribution type |
