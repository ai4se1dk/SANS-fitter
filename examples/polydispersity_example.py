"""
Example: Using SANSFitter with Polydispersity

This script demonstrates how to:
1. Check if a model supports polydispersity
2. Enable and configure polydispersity parameters
3. Set polydispersity width, distribution type, and other settings
4. Compare fits with and without polydispersity
5. Use different polydispersity distribution types

Polydispersity accounts for the size distribution of particles in a sample,
which is common in real experimental systems where not all particles are
exactly the same size.
"""

from sans_fitter import SANSFitter, PD_DEFAULTS, PD_DISTRIBUTION_TYPES

# ============================================================================
# Part 1: Checking Polydispersity Support
# ============================================================================

print("=" * 80)
print("Part 1: Checking Polydispersity Support")
print("=" * 80)

# Initialize the fitter
fitter = SANSFitter()

# Load SANS data
fitter.load_data('simulated_sans_data.csv')

# Set up a sphere model
fitter.set_model('sphere')

# Check if the model supports polydispersity
print(f"\nModel supports polydispersity: {fitter.supports_polydispersity()}")

# Get the list of polydisperse parameters
pd_params = fitter.get_polydisperse_parameters()
print(f"Polydisperse parameters: {pd_params}")

# Show available distribution types
print(f"\nAvailable distribution types: {PD_DISTRIBUTION_TYPES}")

# Show default PD values
print(f"\nDefault PD values: {PD_DEFAULTS}")

# ============================================================================
# Part 2: Fitting WITHOUT Polydispersity (Baseline)
# ============================================================================

print("\n" + "=" * 80)
print("Part 2: Fitting WITHOUT Polydispersity (Baseline)")
print("=" * 80)

# Configure form factor parameters
fitter.set_param('radius', value=50, min=10, max=100, vary=True)
fitter.set_param('sld', value=4.0, vary=False)
fitter.set_param('sld_solvent', value=1.0, vary=False)
fitter.set_param('scale', value=0.01, min=0.001, max=1, vary=True)
fitter.set_param('background', value=0.001, min=0, max=0.1, vary=True)

# Verify polydispersity is disabled
print(f"\nPolydispersity enabled: {fitter.is_polydispersity_enabled()}")

# Fit without polydispersity
print("\nFitting without polydispersity...")
result_mono = fitter.fit(engine='bumps', method='amoeba')
print(f"Chi-squared (monodisperse): {result_mono['chisq']:.4f}")

# ============================================================================
# Part 3: Enabling and Configuring Polydispersity
# ============================================================================

print("\n" + "=" * 80)
print("Part 3: Enabling and Configuring Polydispersity")
print("=" * 80)

# Create a new fitter for polydisperse fitting
fitter_pd = SANSFitter()
fitter_pd.load_data('simulated_sans_data.csv')
fitter_pd.set_model('sphere')

# Configure form factor parameters (same as before)
fitter_pd.set_param('radius', value=50, min=10, max=100, vary=True)
fitter_pd.set_param('sld', value=4.0, vary=False)
fitter_pd.set_param('sld_solvent', value=1.0, vary=False)
fitter_pd.set_param('scale', value=0.01, min=0.001, max=1, vary=True)
fitter_pd.set_param('background', value=0.001, min=0, max=0.1, vary=True)

# Configure polydispersity for the radius parameter
# pd_width: relative width of the distribution (0.1 = 10% polydispersity)
# pd_type: distribution shape ('gaussian', 'lognormal', 'schulz', etc.)
# pd_n: number of quadrature points (higher = more accurate, slower)
# pd_nsigma: number of standard deviations to include
# vary: whether to fit the pd_width during optimization

fitter_pd.set_pd_param(
    'radius',
    pd_width=0.1,       # 10% size polydispersity
    pd_type='gaussian', # Gaussian size distribution
    pd_n=35,            # Default quadrature points
    pd_nsigma=3.0,      # Include 3 sigma range
    vary=False          # Keep PD width fixed during fit
)

# View the PD configuration
print("\nPolydispersity configuration for 'radius':")
pd_config = fitter_pd.get_pd_param('radius')
for key, value in pd_config.items():
    print(f"  {key}: {value}")

# Enable polydispersity globally
fitter_pd.enable_polydispersity(True)
print(f"\nPolydispersity enabled: {fitter_pd.is_polydispersity_enabled()}")

# Display all PD parameters
fitter_pd.get_pd_params()

# ============================================================================
# Part 4: Fitting WITH Polydispersity
# ============================================================================

print("\n" + "=" * 80)
print("Part 4: Fitting WITH Polydispersity")
print("=" * 80)

print("\nFitting with 10% Gaussian polydispersity on radius...")
result_pd = fitter_pd.fit(engine='bumps', method='amoeba')
print(f"Chi-squared (polydisperse): {result_pd['chisq']:.4f}")

# Compare chi-squared values
print(f"\n--- Comparison ---")
print(f"Monodisperse chi-squared: {result_mono['chisq']:.4f}")
print(f"Polydisperse chi-squared: {result_pd['chisq']:.4f}")

# ============================================================================
# Part 5: Using Different Distribution Types
# ============================================================================

print("\n" + "=" * 80)
print("Part 5: Using Different Distribution Types")
print("=" * 80)

# Lognormal distribution - often more realistic for particle sizes
# as it prevents negative sizes and naturally skews toward smaller particles
fitter_lognorm = SANSFitter()
fitter_lognorm.load_data('simulated_sans_data.csv')
fitter_lognorm.set_model('sphere')

fitter_lognorm.set_param('radius', value=50, min=10, max=100, vary=True)
fitter_lognorm.set_param('scale', value=0.01, min=0.001, max=1, vary=True)
fitter_lognorm.set_param('background', value=0.001, min=0, max=0.1, vary=True)
fitter_lognorm.set_param('sld', value=4.0, vary=False)
fitter_lognorm.set_param('sld_solvent', value=1.0, vary=False)

# Use lognormal distribution
fitter_lognorm.set_pd_param('radius', pd_width=0.1, pd_type='lognormal')
fitter_lognorm.enable_polydispersity(True)

print("Fitting with lognormal distribution...")
result_lognorm = fitter_lognorm.fit(engine='bumps', method='amoeba')
print(f"Chi-squared (lognormal): {result_lognorm['chisq']:.4f}")

# Schulz distribution - commonly used for polymers and colloids
fitter_schulz = SANSFitter()
fitter_schulz.load_data('simulated_sans_data.csv')
fitter_schulz.set_model('sphere')

fitter_schulz.set_param('radius', value=50, min=10, max=100, vary=True)
fitter_schulz.set_param('scale', value=0.01, min=0.001, max=1, vary=True)
fitter_schulz.set_param('background', value=0.001, min=0, max=0.1, vary=True)
fitter_schulz.set_param('sld', value=4.0, vary=False)
fitter_schulz.set_param('sld_solvent', value=1.0, vary=False)

# Use Schulz distribution
fitter_schulz.set_pd_param('radius', pd_width=0.1, pd_type='schulz')
fitter_schulz.enable_polydispersity(True)

print("Fitting with Schulz distribution...")
result_schulz = fitter_schulz.fit(engine='bumps', method='amoeba')
print(f"Chi-squared (Schulz): {result_schulz['chisq']:.4f}")

# ============================================================================
# Part 6: Fitting the Polydispersity Width
# ============================================================================

print("\n" + "=" * 80)
print("Part 6: Fitting the Polydispersity Width")
print("=" * 80)

# You can also vary the polydispersity width during fitting
fitter_vary_pd = SANSFitter()
fitter_vary_pd.load_data('simulated_sans_data.csv')
fitter_vary_pd.set_model('sphere')

fitter_vary_pd.set_param('radius', value=50, min=10, max=100, vary=True)
fitter_vary_pd.set_param('scale', value=0.01, min=0.001, max=1, vary=True)
fitter_vary_pd.set_param('background', value=0.001, min=0, max=0.1, vary=True)
fitter_vary_pd.set_param('sld', value=4.0, vary=False)
fitter_vary_pd.set_param('sld_solvent', value=1.0, vary=False)

# Set PD with vary=True to fit the polydispersity width
fitter_vary_pd.set_pd_param(
    'radius',
    pd_width=0.05,      # Initial guess of 5% polydispersity
    pd_type='gaussian',
    vary=True           # Fit the PD width!
)
fitter_vary_pd.enable_polydispersity(True)

print("Fitting with varying polydispersity width...")
print("(Note: This adds an extra fitting parameter)")

# Get list of parameters that will vary (including PD)
varying_pd = fitter_vary_pd.get_varying_pd_params()
print(f"Polydispersity parameters that will vary: {varying_pd}")

# ============================================================================
# Part 7: Multiple Polydisperse Parameters (Cylinder Example)
# ============================================================================

print("\n" + "=" * 80)
print("Part 7: Multiple Polydisperse Parameters (Cylinder)")
print("=" * 80)

fitter_cyl = SANSFitter()
fitter_cyl.load_data('simulated_sans_data.csv')
fitter_cyl.set_model('cylinder')

# Check polydisperse parameters for cylinder
pd_params_cyl = fitter_cyl.get_polydisperse_parameters()
print(f"Polydisperse parameters for cylinder: {pd_params_cyl}")

# Configure form factor parameters
fitter_cyl.set_param('radius', value=20, min=5, max=50, vary=True)
fitter_cyl.set_param('length', value=400, min=100, max=1000, vary=True)
fitter_cyl.set_param('scale', value=0.01, vary=True)
fitter_cyl.set_param('background', value=0.001, vary=True)

# Configure polydispersity for BOTH radius and length
fitter_cyl.set_pd_param('radius', pd_width=0.1, pd_type='gaussian')
fitter_cyl.set_pd_param('length', pd_width=0.15, pd_type='gaussian')

# Enable polydispersity
fitter_cyl.enable_polydispersity(True)

# Display all PD parameters
print("\nPolydispersity configuration for cylinder:")
fitter_cyl.get_pd_params()

# ============================================================================
# Part 8: Toggling Polydispersity On/Off
# ============================================================================

print("\n" + "=" * 80)
print("Part 8: Toggling Polydispersity On/Off")
print("=" * 80)

# Values are preserved when you toggle polydispersity off and on
print("Current PD configuration:")
print(f"  PD enabled: {fitter_cyl.is_polydispersity_enabled()}")
print(f"  radius pd_width: {fitter_cyl.get_pd_param('radius')['pd']}")

# Disable polydispersity
fitter_cyl.enable_polydispersity(False)
print("\nAfter disabling PD:")
print(f"  PD enabled: {fitter_cyl.is_polydispersity_enabled()}")

# Re-enable - values are still there!
fitter_cyl.enable_polydispersity(True)
print("\nAfter re-enabling PD:")
print(f"  PD enabled: {fitter_cyl.is_polydispersity_enabled()}")
print(f"  radius pd_width: {fitter_cyl.get_pd_param('radius')['pd']}")
print("  (Values were preserved!)")

# ============================================================================
# Part 9: Complete Workflow with Plotting
# ============================================================================

print("\n" + "=" * 80)
print("Part 9: Complete Workflow with Plotting")
print("=" * 80)

# Full workflow example
fitter_full = SANSFitter()
fitter_full.load_data('simulated_sans_data.csv')
fitter_full.set_model('sphere')

# Set form factor parameters
fitter_full.set_param('radius', value=50, min=10, max=100, vary=True)
fitter_full.set_param('sld', value=4.0, vary=False)
fitter_full.set_param('sld_solvent', value=1.0, vary=False)
fitter_full.set_param('scale', value=0.01, min=0.001, max=1, vary=True)
fitter_full.set_param('background', value=0.001, min=0, max=0.1, vary=True)

# Configure 10% Gaussian polydispersity on radius
fitter_full.set_pd_param('radius', pd_width=0.1, pd_type='gaussian')
fitter_full.enable_polydispersity(True)

# Fit
result_full = fitter_full.fit(engine='bumps', method='amoeba')

# Plot results
print("\nGenerating plot...")
fitter_full.plot_results(show_residuals=True, log_scale=True)

# Save results
fitter_full.save_results('sphere_polydisperse_fit_results.csv')

print("\n" + "=" * 80)
print("Summary: Polydispersity Key Points")
print("=" * 80)
print("""
✓ Use supports_polydispersity() to check if a model supports PD
✓ Use get_polydisperse_parameters() to see which parameters can be polydisperse
✓ Use set_pd_param() to configure pd_width, pd_type, pd_n, pd_nsigma, vary
✓ Use enable_polydispersity(True/False) to toggle PD globally
✓ PD values are preserved when toggling - no need to reconfigure
✓ Available distribution types: gaussian, lognormal, schulz, rectangle, boltzmann
✓ pd_width is relative: 0.1 = 10% polydispersity
✓ Set vary=True on pd_param to fit the polydispersity width itself
""")

print("\n✓ Polydispersity example completed successfully!")
