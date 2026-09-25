"""
Example: Four particle populations, each with its own structure factor

sasmodels allows one structure factor per form factor (``P@S``), so "four
structure factors" means four populations in one sample, each interacting
through its own S(q):

    I(q) = scale * sum_k [ scale_k * P_k(q) * S_k(q) ] + background

    spheres   sphere            @ hardsphere        large colloids, excluded volume
    micelles  core_shell_sphere @ hayter_msa        charged micelles, screened Coulomb
    rods      cylinder          @ squarewell        short rods, short-range attraction
    drops     ellipsoid         @ stickyhardsphere  emulsion droplets, adhesive contact

This script:
1. Simulates such a sample with known truth
2. Builds the 4-component composite with set_models()
3. Fixes the form-factor geometry (in practice known from dilute measurements),
   links each population's scale to its S(q) volume fraction, and fits the
   volume fractions plus the interaction parameter of each S(q)
4. Fits with a global optimizer, compares with the truth, and plots the
   per-component curves

Fitting four P*S products at once is badly conditioned: the components
overlap in Q and the fit can trade one against another. Fixing what you know
independently (sizes, SLDs) and tying scale to volume fraction is what
makes it tractable.
"""

import logging

from sans_fitter import SANSFitter
from sans_fitter.examples import simulate

logging.basicConfig(level=logging.INFO, format='%(message)s')

MODELS = {
    'spheres': 'sphere@hardsphere',
    'micelles': 'core_shell_sphere@hayter_msa',
    'rods': 'cylinder@squarewell',
    'drops': 'ellipsoid@stickyhardsphere',
}

# Form-factor geometry and contrast: known, held fixed during the fit.
# radius_effective_mode=1 derives each S(q) radius from its form factor
# (equivalent-volume sphere), so no free radius_effective is left over.
FIXED = {
    'spheres_radius': 250.0,
    'spheres_sld': 5.0,
    'spheres_sld_solvent': 6.0,
    'spheres_radius_effective_mode': 1,
    'micelles_radius': 20.0,
    'micelles_thickness': 10.0,
    'micelles_sld_core': 0.0,
    'micelles_sld_shell': 1.0,
    'micelles_sld_solvent': 6.0,
    'micelles_concentration_salt': 0.01,
    'micelles_radius_effective_mode': 1,
    'rods_radius': 10.0,
    'rods_length': 200.0,
    'rods_sld': 4.0,
    'rods_sld_solvent': 1.0,
    'rods_wellwidth': 1.2,
    'rods_radius_effective_mode': 1,
    'drops_radius_polar': 30.0,
    'drops_radius_equatorial': 60.0,
    'drops_sld': 2.0,
    'drops_sld_solvent': 1.0,
    'drops_perturb': 0.05,
    'drops_radius_effective_mode': 1,
}

# What the fit should recover: each population's volume fraction plus the
# interaction strength of its S(q). The per-component scale is linked to the
# volume fraction below, so it is not a separate unknown.
TRUTH = {
    'spheres_volfraction': 0.25,
    'micelles_volfraction': 0.05,
    'micelles_charge': 20.0,
    'rods_volfraction': 0.05,
    'rods_welldepth': 1.0,
    'drops_volfraction': 0.10,
    'drops_stickiness': 0.20,
    'background': 0.001,
}

# Starting guesses and bounds: (value, min, max)
START = {
    'spheres_volfraction': (0.15, 0.01, 0.45),
    'micelles_volfraction': (0.08, 0.005, 0.3),
    'micelles_charge': (10.0, 1.0, 60.0),
    'rods_volfraction': (0.08, 0.005, 0.3),
    'rods_welldepth': (0.5, 0.0, 3.0),
    'drops_volfraction': (0.15, 0.01, 0.4),
    'drops_stickiness': (0.3, 0.05, 2.0),
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
truth = {**FIXED, **TRUTH}
for moniker in MODELS:  # scale_k == volfraction_k in the simulated sample too
    truth[f'{moniker}_scale'] = truth[f'{moniker}_volfraction']
truth_raw = {to_raw(k): v for k, v in truth.items()}
data = simulate(expression, qmin=0.003, qmax=0.4, npoints=150, noise=0.02, seed=7, **truth_raw)

# 2. Four-component composite with friendly parameter names
fitter = SANSFitter()
fitter.set_data(data)
fitter.set_resolution('none')  # simulated data carries no dQ
fitter.set_models(**MODELS)

# 3. Fix geometry/contrast, free the amplitudes and S(q) interactions
for name, value in FIXED.items():
    fitter.set_param(name, value=value, vary=False)
for name, (value, lo, hi) in START.items():
    fitter.set_param(name, value=value, min=lo, max=hi, vary=True)

# Without this, scale_k and volfraction_k are fully correlated: scale_k
# follows volfraction_k, so each population has one amplitude that also
# sets the strength of its interactions.
for moniker in MODELS:
    fitter.link_params(f'{moniker}_scale', to=f'{moniker}_volfraction')

# Composite models fit with the bumps engine only. Use the global
# differential-evolution optimizer: a local one (amoeba, lm) started from
# these guesses stalls in a false minimum at chi2/dof ~ 2.5, trading rod
# attraction against micelle charge.
fitter.fit(engine='bumps', method='de', steps=300)

# 4. Compare with the truth
print(f'\n{"parameter":<24}{"truth":>10}{"fit":>10}')
for name, true in TRUTH.items():
    print(f'{name:<24}{true:>10.4g}{fitter.params[name]["value"]:>10.4g}')

# rods_welldepth is the weakest parameter: at 5% volume fraction the rods'
# S(q) barely departs from 1, so expect an uncertainty comparable to the value.
fitter.plot_results(show_components=True)
