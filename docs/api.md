# API Reference

## SANSFitter

The main class for SANS data fitting.

::: sans_fitter.sans_fitter.SANSFitter
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
        - set_structure_factor
        - get_params
        - set_param
        - fit
        - fit_bayesian
        - get_posteriors
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

::: sans_fitter.data_ops
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

::: sans_fitter.pr_inversion
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


## ParameterManager

Internal class for managing model parameters and polydispersity settings.

::: sans_fitter.parameter_manager.ParameterManager
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

## Polydispersity Constants

The following constants are available in `sans_fitter.parameter_manager`:

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
