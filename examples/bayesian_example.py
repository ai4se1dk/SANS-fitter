"""
Example: Bayesian (MCMC) analysis with SANSFitter

This script demonstrates how to:
1. Load SANS data and set up a model
2. Sample the posterior with fit_bayesian() (bumps DREAM)
3. Render the five posterior displays
4. Access and export the raw posterior chain
"""

from sans_fitter import SANSFitter

# Initialize the fitter and load data
fitter = SANSFitter()
fitter.load_data('simulated_sans_data.csv')

# Set up a cylinder model
fitter.set_model('cylinder')
fitter.set_param('radius', value=20, min=1, max=100, vary=True)
fitter.set_param('length', value=400, min=10, max=1000, vary=True)
fitter.set_param('sld', value=4.0, vary=False)
fitter.set_param('sld_solvent', value=1.0, vary=False)
fitter.set_param('scale', value=1.0, min=0.1, max=10, vary=True)
fitter.set_param('background', value=0.001, min=0, max=1, vary=True)

# ============================================================================
# Bayesian fit: sample the posterior with DREAM
# ============================================================================
# Note: this evaluates the model tens of thousands of times. Reduce
# samples/burn for a quick look; increase them for production-quality
# posteriors (check the R-hat/ESS diagnostics printed with the summary).
print('\n' + '=' * 80)
print('Sampling the posterior with BUMPS DREAM...')
print('=' * 80)
result = fitter.fit_bayesian(samples=5000, burn=100)

# The regular point-estimate views still work: the reported curve is the
# best (maximum-likelihood) posterior sample.
fitter.plot_results(show_residuals=True, log_scale=True)

# ============================================================================
# Posterior displays
# ============================================================================

# Corner plot: marginal densities on the diagonal, sample clouds below
print('\nGenerating posterior pair (corner) plot...')
fitter.plot_posterior_pairs()

# Marginal posterior for a single parameter
print('\nGenerating marginal posterior for radius...')
fitter.plot_param_distribution('radius')

# Posterior predictive check: 95% credible band over the measured data
print('\nGenerating posterior predictive band...')
fitter.plot_posterior_predictive(style='band')

# ... or with individual posterior draws overlaid
print('\nGenerating posterior predictive band + draws...')
fitter.plot_posterior_predictive(style='band+draws', n_draws=30)

# Correlation heatmap of the sampled parameters
print('\nGenerating parameter correlation heatmap...')
fitter.plot_param_correlations()

# MCMC chain traces (convergence check)
print('\nGenerating MCMC trace plot...')
fitter.plot_trace()

# ============================================================================
# Programmatic access to the posterior
# ============================================================================
posterior = fitter.get_posterior()
print('\nSampled parameters:', posterior.labels)
print('Chain shape:', posterior.samples.shape)
print('95% credible intervals:')
for name in posterior.labels:
    low, high = posterior.ci_95[name]
    print(f'  {name}: [{low:.6g}, {high:.6g}]')

# Export the raw chain for external analysis (pandas, corner, arviz, ...)
posterior.save_posterior_csv('posterior_chain.csv')
print('\nRaw posterior chain saved to posterior_chain.csv')

# The saved fit results include the credible intervals in the header
fitter.save_results('bayesian_fit_results.csv')

print('\n✓ Example completed successfully!')
