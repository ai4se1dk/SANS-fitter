"""
Example: Previewing the Theory Before Fitting

This script demonstrates how to:
1. Plot the model at the current parameters, before any fit (plot_model)
2. Overlay several candidate parameter sets on one plot (compare)
3. Read the intensities out as an array, on the data grid or on any Q grid (calculate)
4. Confirm that none of it touches the fitter's parameters or its fit results

The preview answers the question SasView answers by redrawing the theory as
you drag a slider: "are my starting values sane?" An optimiser started in the
wrong basin fails slowly and quietly, so ten seconds of looking first is
cheaper than debugging the fit afterwards.
"""

import numpy as np

from sans_fitter import SANSFitter, examples

# ============================================================================
# Part 1: A Dataset with a Known Answer
# ============================================================================

print('=' * 80)
print('Part 1: A Dataset with a Known Answer')
print('=' * 80)

# Simulated data carries its generating parameters, so every claim the preview
# makes below can be checked against the truth.
data = examples.simulate(
    'sphere',
    radius=60.0,
    sld=4.0,
    sld_solvent=1.0,
    scale=0.05,
    background=0.001,
    npoints=90,
    noise=0.02,
    seed=7,
)
print(f'\nSimulated with radius = {data.truth["radius"]} Å')

fitter = SANSFitter()
fitter.set_data(data)
fitter.set_model('sphere')

# Start deliberately off: 25 Å instead of the true 60 Å.
fitter.set_param('radius', value=25, min=5, max=300, vary=True)
fitter.set_param('scale', value=0.05, min=1e-4, max=10, vary=True)
fitter.set_param('background', value=0.001, min=0, max=1, vary=True)
fitter.set_param('sld', value=4.0, vary=False)
fitter.set_param('sld_solvent', value=1.0, vary=False)

# ============================================================================
# Part 2: Looking at the Starting Values (plot_model)
# ============================================================================

print('\n' + '=' * 80)
print('Part 2: Looking at the Starting Values')
print('=' * 80)

# Data, the model at the current parameters, and residuals — no fit required.
# The printed chi-squared is chi2/dof, the same number BUMPS prints as
# "Initial chi2" at the start of a fit.
print('\nPlotting the model at radius = 25 Å...')
fitter.plot_model(show_residuals=True, log_scale=True)

# The residual panel is the useful half: a form-factor minimum in the wrong
# place shows up there long before it shows up in a converged fit.

# ============================================================================
# Part 3: Trying Candidate Values (compare)
# ============================================================================

print('\n' + '=' * 80)
print('Part 3: Trying Candidate Values')
print('=' * 80)

# The quick form: sweep one parameter over a list of values.
print('\nSweeping radius over 25, 40, 60 and 90 Å...')
fitter.compare(radius=[25, 40, 60, 90])

# The general form: labelled cases, each a set of overrides applied on top of
# the current parameters. An empty dict means "the current parameters".
print('Overlaying three labelled cases...')
fitter.compare(
    {
        'current (25 Å)': {},
        'truth (60 Å)': {'radius': 60},
        '60 Å, 20% polydisperse': {'radius': 60, 'radius_pd': 0.2},
    }
)

# Note that 'radius_pd' works without enable_polydispersity(True): a compare
# case is a what-if, so it brings the polydispersity settings it needs with it
# instead of demanding that the fitter be reconfigured first.

# ============================================================================
# Part 4: The Intensities Themselves (calculate)
# ============================================================================

print('\n' + '=' * 80)
print('Part 4: The Intensities Themselves')
print('=' * 80)

# On the data grid: one value per data point, with the dataset's own
# resolution applied, so it lines up with fitter.data.x point for point.
intensity = fitter.calculate()
print(f'\nOn the data grid: {len(intensity)} points for {len(fitter.data.x)} data points')
print(f'I(Q) at Q = {fitter.data.x[0]:.4f} Å⁻¹: {intensity[0]:.5g}')

# Points excluded from the fit come back as NaN rather than silently shifting
# the array, so the result can always be plotted against fitter.data.x.
fitter.set_q_range(qmin=0.02)
trimmed = fitter.calculate()
print(f'After set_q_range(qmin=0.02): {np.isnan(trimmed).sum()} of {len(trimmed)} points are NaN')
fitter.reset_q_range()

# On an arbitrary grid: a smooth curve for publication plots, or a resolution
# study. dq is a relative width (ΔQ/Q) and applies to the explicit grid only.
q = np.geomspace(0.005, 0.5, 400)
sharp = fitter.calculate(q=q)
smeared = fitter.calculate(q=q, dq=0.10)

# Smearing fills in the form-factor minima, so the sharp curve dips deeper.
print(f'Deepest minimum, unsmeared:      {sharp.min():.4g}')
print(f'Deepest minimum, 10% ΔQ/Q:       {smeared.min():.4g}')

# ============================================================================
# Part 5: The Preview Changes Nothing
# ============================================================================

print('\n' + '=' * 80)
print('Part 5: The Preview Changes Nothing')
print('=' * 80)

# Every call above evaluated the model without writing anything back.
print(f'\nradius is still {fitter.params["radius"]["value"]} Å')
print(f'fit_result is still {fitter.fit_result}')

# Now do the real fit and see where the starting point actually got us.
result = fitter.fit(engine='bumps', method='amoeba')
print(f'\nradius after fitting: {result["parameters"]["radius"]["value"]:.2f} Å')
print(f'truth was:            {data.truth["radius"]} Å')

# plot_results() shows the fit; plot_model() would still show the model at the
# current parameters — which, after a fit, are the fitted ones.
fitter.plot_results(show_residuals=True, log_scale=True)

print('\n' + '=' * 80)
print('Summary: Theory Preview Key Points')
print('=' * 80)
print("""
✓ plot_model() draws data, model and residuals at the current parameters
✓ Its χ² is χ²/dof, matching the "Initial χ²" a BUMPS fit prints
✓ compare(radius=[...]) sweeps one parameter; compare({'label': {...}}) is general
✓ Overrides accept aliases, shared names and polydispersity widths (radius_pd)
✓ calculate() returns intensities: on the data grid, or on any q= grid
✓ On the data grid, excluded points are NaN so the array stays aligned
✓ dq= smears an explicit q grid; the dataset's own resolution is used otherwise
✓ None of it changes parameters or fit results — plot_results() is unaffected
""")

print('\n✓ Theory preview example completed successfully!')
