"""
Example: Combining Multiple Models (Composite Models)

This script demonstrates how to:
1. Simulate data with a low-Q diffuse feature plus a high-Q peak
2. Combine two models (dab + peak_lorentz) against the single dataset
3. Configure per-component parameters with friendly names
4. Fit with the bumps engine
5. Visualize per-component curves
6. Share parameters across components and use equality links

Composite models let you describe a dataset that needs more than one physical
model — the motivating case from issue #46: a low-Q diffuse contribution
(dab) together with a high-Q correlation peak (peak_lorentz), sharing a common
background.
"""

import numpy as np
from sasdata.dataloader.data_info import Data1D
from sasmodels.core import load_model
from sasmodels.direct_model import DirectModel

from sans_fitter import SANSFitter


def simulate_composite_data():
    """Generate synthetic dab + peak_lorentz data on a realistic Q grid."""
    q = np.linspace(0.005, 0.3, 80)
    data = Data1D(x=q, y=np.ones_like(q), dy=np.full_like(q, 0.05))
    data.qmin, data.qmax = q.min(), q.max()

    kernel = load_model('dab+peak_lorentz', dtype='single', platform='dll')
    truth = dict(
        scale=1.0,
        background=0.01,
        A_scale=10.0,
        A_cor_length=50.0,
        B_scale=5.0,
        B_peak_pos=0.1,
        B_peak_hwhm=0.01,
    )
    y = np.asarray(DirectModel(data, kernel)(**truth))
    rng = np.random.default_rng(42)
    data.y = y + rng.normal(0, 0.02 * y, size=y.shape)
    return data


# ============================================================================
# Part 1: Combine two models with friendly names
# ============================================================================

print('=' * 80)
print('Part 1: Combine dab + peak_lorentz against one dataset')
print('=' * 80)

fitter = SANSFitter()
fitter.set_data(simulate_composite_data())

# Combine the models; parameters get friendly per-model prefixes
fitter.set_models('dab', 'peak_lorentz')

print('\nCombined-model parameters:')
fitter.get_params()

# Configure the parameters (global scale/background are shared natively)
fitter.set_param('dab_cor_length', value=40, min=1, max=500, vary=True)
fitter.set_param('dab_scale', value=8, min=0.1, max=100, vary=True)
fitter.set_param('peak_lorentz_peak_pos', value=0.08, min=0.01, max=0.5, vary=True)
fitter.set_param('peak_lorentz_peak_hwhm', value=0.008, min=0.001, max=0.1, vary=True)
fitter.set_param('peak_lorentz_scale', value=4, min=0.1, max=100, vary=True)
fitter.set_param('background', value=0.005, min=0, max=0.1, vary=True)

# Fit with the bumps engine (composite models require bumps)
result = fitter.fit(engine='bumps', method='amoeba')

# Overlay one dashed curve per component to see which feature each model fits
fitter.plot_results(show_components=True, show=False)

# ============================================================================
# Part 2: Custom monikers and shared parameters
# ============================================================================

print('\n' + '=' * 80)
print('Part 2: Monikers and shared parameters (two sphere populations)')
print('=' * 80)

# Build two-sphere data sharing a contrast
q = np.linspace(0.005, 0.3, 80)
data = Data1D(x=q, y=np.ones_like(q), dy=np.full_like(q, 0.05))
data.qmin, data.qmax = q.min(), q.max()
kernel = load_model('sphere+sphere', dtype='single', platform='dll')
truth = dict(
    scale=0.1, background=0.001,
    A_sld=4.0, A_sld_solvent=6.4, A_radius=20.0, A_scale=1.0,
    B_sld=4.0, B_sld_solvent=6.4, B_radius=200.0, B_scale=1.0,
)
data.y = np.asarray(DirectModel(data, kernel)(**truth))

fitter2 = SANSFitter()
fitter2.set_data(data)

# Monikers label the components; shared=['sld', 'sld_solvent'] collapses the
# contrast into single unprefixed parameters driving both spheres.
fitter2.set_models(small='sphere', large='sphere', shared=['sld', 'sld_solvent'])

print('\nTwo-sphere parameters (contrast shared):')
fitter2.get_params()

fitter2.set_param('sld', value=4.0, min=0, max=8, vary=True)
fitter2.set_param('sld_solvent', value=6.4, vary=False)
fitter2.set_param('small_radius', value=15, min=5, max=100, vary=True)
fitter2.set_param('large_radius', value=150, min=50, max=1000, vary=True)
fitter2.set_param('scale', value=0.1, min=0.001, max=1, vary=True)
fitter2.set_param('background', value=0.001, min=0, max=0.1, vary=True)

result2 = fitter2.fit(engine='bumps', method='amoeba')
fitter2.plot_results(show_components=True, show=False)

# ============================================================================
# Part 3: Equality links for asymmetric sharing
# ============================================================================

print('\n' + '=' * 80)
print('Part 3: Equality links (link_params)')
print('=' * 80)

# link_params handles sharing that shared= cannot express: linking only some
# components, or parameters with different names.
fitter3 = SANSFitter()
fitter3.set_data(data)
fitter3.set_models(small='sphere', large='sphere')

# Force the large sphere to share the small sphere's SLD
fitter3.link_params('large_sld', to='small_sld')
print('\nActive links:', fitter3.get_links())

fitter3.set_param('small_sld', value=4.0, min=0, max=8, vary=True)
# Match the solvent SLD the data was simulated with (the sphere default is
# 6.0); otherwise the fit compensates through small_sld and recovers ~3.6.
fitter3.set_param('small_sld_solvent', value=6.4, vary=False)
fitter3.set_param('large_sld_solvent', value=6.4, vary=False)
fitter3.set_param('small_radius', value=15, min=5, max=100, vary=True)
fitter3.set_param('large_radius', value=150, min=50, max=1000, vary=True)
fitter3.set_param('scale', value=0.1, min=0.001, max=1, vary=True)
fitter3.set_param('background', value=0.001, min=0, max=0.1, vary=True)

result3 = fitter3.fit(engine='bumps', method='amoeba')

# The follower mirrors the target after the fit
print(f"\nsmall_sld = {fitter3.params['small_sld']['value']:.4f}")
print(f"large_sld = {fitter3.params['large_sld']['value']:.4f} (linked)")

# Remove the link to restore independence
fitter3.unlink_params('large_sld')
print('Links after unlink:', fitter3.get_links())

print('\n' + '=' * 80)
print('Done. See docs/usage.md "Combining Models" for details.')
print('=' * 80)
