"""
Example: Four form factors against one dataset

A sample holding four non-interacting populations (dilute, so S(q) = 1)
scatters as the sum of their form factors:

    I(q) = sum_k scale_k * P_k(q) + background

    spheres   sphere             large colloids, dominate the lowest Q
    shells    core_shell_sphere  vesicle-like shells, mid-Q oscillations
    rods      cylinder           rod-like aggregates, q^-1 region
    coils     mono_gauss_coil    free polymer, dominates the highest Q

This script:
1. Simulates such a sample with known truth
2. Builds the 4-component sum with set_models()
3. Fixes the SLDs (scale and contrast are degenerate in a sum; only
   scale * contrast^2 is measured) and fits sizes plus amplitudes
4. Fits with a global optimizer, compares with the truth, and plots the
   per-component curves

A sum of four form factors is badly conditioned when the components overlap
in Q: the fit can trade one against another and still match the data. The
populations here are chosen to dominate different Q ranges; with real data,
fix whatever you know independently (sizes from microscopy, SLDs from
composition) and check the correlation matrix in the fit report.
"""

import logging

from sans_fitter import SANSFitter
from sans_fitter.examples import simulate

logging.basicConfig(level=logging.INFO, format='%(message)s')

MODELS = {
    'spheres': 'sphere',
    'shells': 'core_shell_sphere',
    'rods': 'cylinder',
    'coils': 'mono_gauss_coil',
}

# Contrast: known from composition, held fixed. mono_gauss_coil carries its
# amplitude in i_zero, so its scale is fixed at 1 to avoid a redundant pair.
FIXED = {
    'spheres_sld': 1.0,
    'spheres_sld_solvent': 6.0,
    'shells_sld_core': 3.0,
    'shells_sld_shell': 1.0,
    'shells_sld_solvent': 6.0,
    'rods_sld': 4.0,
    'rods_sld_solvent': 1.0,
    'coils_scale': 1.0,
}

# What the fit should recover: every size and every amplitude.
TRUTH = {
    'spheres_scale': 0.002,
    'spheres_radius': 300.0,
    'shells_scale': 0.005,
    'shells_radius': 40.0,
    'shells_thickness': 15.0,
    'rods_scale': 0.01,
    'rods_radius': 15.0,
    'rods_length': 500.0,
    'coils_i_zero': 1.0,
    'coils_rg': 30.0,
    'background': 0.001,
}

# Starting guesses and bounds: (value, min, max)
START = {
    'spheres_scale': (0.005, 1e-5, 0.1),
    'spheres_radius': (200.0, 100.0, 600.0),
    'shells_scale': (0.01, 1e-5, 0.1),
    'shells_radius': (30.0, 10.0, 100.0),
    'shells_thickness': (10.0, 2.0, 40.0),
    'rods_scale': (0.005, 1e-5, 0.1),
    'rods_radius': (10.0, 3.0, 50.0),
    'rods_length': (300.0, 100.0, 2000.0),
    'coils_i_zero': (0.5, 0.01, 5.0),
    'coils_rg': (20.0, 5.0, 100.0),
    'background': (0.002, 0.0, 0.01),
}


def to_raw(name):
    """Friendly name -> sasmodels A_/B_/C_/D_ name, for simulate()."""
    if name in ('scale', 'background'):
        return name
    for prefix, moniker in zip('ABCD', MODELS, strict=True):
        if name.startswith(moniker + '_'):
            return f'{prefix}_{name[len(moniker) + 1 :]}'
    raise KeyError(name)


# 1. Simulate the sample
expression = '+'.join(MODELS.values())
truth_raw = {to_raw(k): v for k, v in {**FIXED, **TRUTH}.items()}
data = simulate(expression, qmin=0.003, qmax=0.4, npoints=150, noise=0.02, seed=7, **truth_raw)

# 2. Four-component sum with friendly parameter names
fitter = SANSFitter()
fitter.set_data(data)
fitter.set_resolution('none')  # simulated data carries no dQ
fitter.set_models(**MODELS)

# 3. Fix contrast, free sizes and amplitudes
for name, value in FIXED.items():
    fitter.set_param(name, value=value, vary=False)
for name, (value, lo, hi) in START.items():
    fitter.set_param(name, value=value, min=lo, max=hi, vary=True)

# Composite models fit with the bumps engine only. With eleven free
# parameters and overlapping components, a global optimizer is the safe
# default; a local one can settle in a minimum where components swap roles.
fitter.fit(engine='bumps', method='de', steps=1000)

# 4. Compare with the truth
print(f'\n{"parameter":<24}{"truth":>10}{"fit":>10}')
for name, true in TRUTH.items():
    print(f'{name:<24}{true:>10.4g}{fitter.params[name]["value"]:>10.4g}')

# What to expect in the fit report's correlation matrix:
# - shells_radius / shells_thickness near -1: the data pin the outer radius
#   (radius + thickness) far better than how it splits between core and shell.
# - rods_length is loose: the rods' low-Q Guinier bend sits under the far
#   stronger sphere signal, so only the q^-1 region constrains them.
# - coils_i_zero / coils_rg: the coil is visible only where it outlasts the
#   other components at high Q.
# Fix any of these from independent knowledge if it matters to the result.

fitter.plot_results(show_components=True)
