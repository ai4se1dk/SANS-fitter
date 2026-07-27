"""
Example: Using SANSFitter with plotting functionality

This script demonstrates how to:
1. Load SANS data
2. Set up a model with parameters
3. Perform fitting
4. Visualize results with plot_results()
5. Load data with Q-resolution and see horizontal error bars
"""

from sans_fitter import SANSFitter

# ============================================================================
# Part 1: Standard Workflow (data without Q-resolution)
# ============================================================================

print("=" * 80)
print("Part 1: Standard Workflow (no Q-resolution)")
print("=" * 80)

# Initialize the fitter
fitter = SANSFitter()

# Load SANS data
# Note the output: "Resolution (dQ) column: no" — this file has no Q-resolution
fitter.load_data('simulated_sans_data.csv')

# Set up a cylinder model
fitter.set_model('cylinder')

# Configure parameters for fitting
fitter.set_param('radius', value=20, min=1, max=100, vary=True)
fitter.set_param('length', value=400, min=10, max=1000, vary=True)
fitter.set_param('sld', value=4.0, vary=False)
fitter.set_param('sld_solvent', value=1.0, vary=False)
fitter.set_param('scale', value=1.0, min=0.1, max=10, vary=True)
fitter.set_param('background', value=0.001, min=0, max=1, vary=True)

# Display current parameters
fitter.get_params()

# Perform the fit using BUMPS
print("\n" + "="*80)
print("Fitting with BUMPS engine...")
print("="*80)
result = fitter.fit(engine='bumps', method='amoeba')

# Plot results with residuals in log scale (default)
print("\nGenerating plot with residuals (log scale)...")
fitter.plot_results(show_residuals=True, log_scale=True)

# Alternative: Plot without residuals
print("\nGenerating plot without residuals...")
fitter.plot_results(show_residuals=False, log_scale=True)

# Alternative: Plot in linear scale
print("\nGenerating plot in linear scale...")
fitter.plot_results(show_residuals=True, log_scale=False)

# Save the results
fitter.save_results('fit_results.csv')

# ============================================================================
# Part 2: Data with Q-Resolution (horizontal error bars)
# ============================================================================

print("\n" + "=" * 80)
print("Part 2: Data with Q-Resolution")
print("=" * 80)

# Load a file that includes a dQ (resolution) column
# Note the output: "Resolution (dQ) column: yes"
fitter_res = SANSFitter()
fitter_res.load_data('simulated_sans_data_with_resolution.csv')

# Set up the same cylinder model
fitter_res.set_model('cylinder')
fitter_res.set_param('radius', value=20, min=1, max=100, vary=True)
fitter_res.set_param('length', value=400, min=10, max=1000, vary=True)
fitter_res.set_param('sld', value=4.0, vary=False)
fitter_res.set_param('sld_solvent', value=1.0, vary=False)
fitter_res.set_param('scale', value=1.0, min=0.1, max=10, vary=True)
fitter_res.set_param('background', value=0.001, min=0, max=1, vary=True)

# Fit
print("\nFitting data with Q-resolution...")
result_res = fitter_res.fit(engine='bumps', method='amoeba')

# Plot — notice horizontal error bars now appear on data points
print("\nPlot with horizontal error bars (Q-resolution)...")
fitter_res.plot_results(show_residuals=True, log_scale=True)

# Save — the CSV now includes a dQ column
fitter_res.save_results('fit_results_with_resolution.csv')
print("\nCSV output includes dQ column when resolution data is present.")

print("\n✓ Example completed successfully!")
