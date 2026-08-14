"""
Example: model-free P(r) inversion with sans_fitter.pr_inversion

This script demonstrates how to:
1. Simulate a protein-like sphere dataset (analytic form factor — P(r)
   inversion is model-free, so no model setup is involved anywhere)
2. Load it with data_ops.load(), the standard SANS-fitter loading path
3. Explore D_max before trusting any inversion
4. Run auto_invert() and read the diagnostics
5. Control n_terms/alpha explicitly and export the result
"""

import numpy as np

from sans_fitter import data_ops, pr_inversion

# ============================================================================
# Part 1: simulate a protein-like sphere dataset (R = 60 Å, 2% noise)
# ============================================================================
# The sphere form factor is analytic: I(q) = I(0)·[3(sin(u) − u·cos(u))/u³]²
# with u = qR — so the simulation needs nothing beyond numpy. With measured
# data you would skip this part entirely and start at data_ops.load().
radius = 60.0
i0_true = 100.0
q = np.geomspace(0.005, 0.25, 80)
u = q * radius
i_clean = i0_true * (3.0 * (np.sin(u) - u * np.cos(u)) / u**3) ** 2

rng = np.random.default_rng(42)
sigma = 0.02 * i_clean
i_noisy = i_clean + rng.normal(0.0, sigma)

with open('simulated_protein.csv', 'w') as f:
    f.write('Q,I,dI\n')
    for q_value, i_value, di_value in zip(q, i_noisy, sigma):
        f.write(f'{q_value},{i_value},{di_value}\n')

data = data_ops.load('simulated_protein.csv')

# The data is simulated background-free, so we skip the fitted background —
# the normal workflow for buffer-subtracted protein data.
FIT_BACKGROUND = False

# ============================================================================
# Part 2: explore D_max first — every P(r) result is conditional on it
# ============================================================================
# Start from a deliberately wrong guess to see the scan earn its keep: the
# Rg/I(0) plateau and the chi-squared minimum locate the true D_max (= 2R).
print('Scanning D_max around a deliberately wrong initial guess (100 Å)...')
scan = pr_inversion.explore_dmax(
    data, d_max=100.0, dmin=80.0, dmax=160.0, n_points=17,
    fit_background=FIT_BACKGROUND,
)
print(scan.format_summary())
scan.plot(quantity='all')

best_dmax = float(scan.d_max_values[np.argmin(scan.data_chisq)])
print(f'\nChi-squared minimum at D_max = {best_dmax:.1f} Å (true value: {2 * radius:.0f} Å)')

# ============================================================================
# Part 3: automatic inversion at the chosen D_max
# ============================================================================
result = pr_inversion.auto_invert(data, d_max=best_dmax, fit_background=FIT_BACKGROUND)
print('\n' + result.format_summary())
print(f'\nExpected sphere Rg = sqrt(3/5)*R = {np.sqrt(3 / 5) * radius:.2f} Å')
print(f'Expected I(0) = {i0_true:.1f}')

result.plot_pr()
result.plot_fit(data)

# ============================================================================
# Part 4: explicit control and export
# ============================================================================
estimate = pr_inversion.estimate_n_terms(data, d_max=best_dmax,
                                         fit_background=FIT_BACKGROUND)
print(f'\nEstimator message: {estimate.message}')

result = pr_inversion.invert(
    data, d_max=best_dmax,
    n_terms=estimate.n_terms, alpha=estimate.alpha,
    fit_background=FIT_BACKGROUND,
)
result.save_csv('pr_result.csv')
print("Saved P(r) curve and diagnostics to 'pr_result.csv'")

print('\n✓ Example completed successfully!')
