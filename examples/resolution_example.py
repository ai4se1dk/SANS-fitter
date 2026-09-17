"""
Example: explicit resolution (smearing) control

Instrument resolution changes the parameters a fit returns, so SANS-fitter
treats it as a stated choice rather than a property of the input file. The four
modes mirror SasView's Fit Page: *None* / *Use dQ Data* / *Custom Pinhole* /
*Custom Slit*.

This script demonstrates how to:
1. Read the active setting, and what the default means
2. Measure what ignoring resolution costs, against known truth
3. State a pinhole width for a dataset that carries no dQ column
4. Smear with slit geometry (the USANS case)
5. Confirm your dataset is never modified, and read the setting back off a
   saved fit
6. Meet the guardrails that stop a wrong width reaching a fit

Run from the repository root:  python examples/resolution_example.py
"""

import os
import tempfile
import warnings

import numpy as np

from sans_fitter import SANSFitter, examples

# The package picks ASCII spellings for these when the console cannot encode
# them, which is what keeps this script runnable on a legacy Windows terminal.
from sans_fitter.console import INVERSE_ANGSTROM, OK, SIGMA

TRUTH = {'radius': 60.0, 'scale': 0.02, 'background': 0.001, 'sld': 4.0, 'sld_solvent': 1.0}


def sphere_fitter(data):
    """A sphere fitter on *data*, started well away from the truth."""
    fitter = SANSFitter()
    fitter.set_data(data)
    fitter.set_model('sphere')
    fitter.set_param('radius', value=40.0, min=10.0, max=200.0, vary=True)
    fitter.set_param('scale', value=0.01, min=1e-4, max=1.0, vary=True)
    fitter.set_param('background', value=0.0, min=0.0, max=0.1, vary=True)
    fitter.set_param('sld', value=4.0, vary=False)
    fitter.set_param('sld_solvent', value=1.0, vary=False)
    return fitter


def banner(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


# ============================================================================
# Part 1: what is the current setting
# ============================================================================
banner('Part 1: the default setting')

fresh = SANSFitter()
print(f'A new fitter starts at: {fresh.get_resolution()}')

# 'data' means "use whatever this dataset carries": a dQ column smears as a
# pinhole, slit columns (dxl/dxw) smear as a slit, and a file with neither is
# evaluated unsmeared with a warning. Nothing is assumed on your behalf.

# ============================================================================
# Part 2: what ignoring resolution costs
# ============================================================================
banner('Part 2: the same data, fitted with and without its resolution')

# simulate(dq=...) both smears the intensity and attaches the matching dQ
# column, so it behaves exactly like a measurement from a real instrument.
measured = examples.simulate('sphere', dq=0.10, noise=0.02, seed=7, npoints=120, **TRUTH)
print(f'Simulated 120 points with {SIGMA}_q/q = 0.10 and radius = {TRUTH["radius"]} A')

for mode in ('data', 'none'):
    fitter = sphere_fitter(measured)
    fitter.set_resolution(mode)
    result = fitter.fit(engine='bumps', method='amoeba')
    radius = result['parameters']['radius']
    error = radius['value'] - TRUTH['radius']
    print(
        f"\n  mode '{mode}': radius = {radius['value']:.3f} +/- {radius['stderr']:.3f} A"
        f' ({error:+.3f} A from truth), chi^2/dof = {result["reduced_chisq"]:.3f}'
    )

# The bias in the radius is modest; the goodness of fit is not. Fitting a
# smeared measurement with a sharp model leaves structure in the residuals
# that no parameter can absorb, and chi^2/dof is where you see it first.
#
# Resolution is a property of the instrument, and smearing is degenerate
# with real physics.
# Polydispersity broadens a form-factor minimum much as resolution does
# so the mode to use is the one the beamline actually had, not the one
# that minimises chi^2.

# A pinhole width you state yourself is the same quantity as the file's dQ
# column, so stating the width the data was made with reproduces mode 'data'.
fitter = sphere_fitter(measured)
fitter.set_resolution('pinhole', dq_over_q=0.10)
stated = fitter.fit(engine='bumps', method='amoeba')
print(f"\n  mode 'pinhole' at 0.10: radius = {stated['parameters']['radius']['value']:.3f} A")
print("  (identical to mode 'data': dq_over_q and the dQ column are both sigma_q/q)")

# ============================================================================
# Part 3: a dataset with no resolution information
# ============================================================================
banner('Part 3: stating a width for a file with no dQ column')

# The same sphere as Part 2, simulated without dq. Data with no smearing.
plain = examples.simulate('sphere', noise=0.02, seed=7, npoints=120, **TRUTH)
print(f"This dataset has dx = {plain.dx}, so mode 'data' has nothing to use.")

# Under 'data' that is not an error, it is a warning, because evaluating
# unsmeared is a thing to do, just not something to do unnoticed.
fitter = sphere_fitter(plain)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    fitter.fit(engine='bumps', method='amoeba')
for warning in caught:
    print(f'\n  warning: {warning.message}')

# Two ways to address it:
#   set_resolution('none')                    -- "this measurement is sharp"
#   set_resolution('pinhole', dq_over_q=...)  -- "the instrument smears by this"
#
# Here the first one is the truth, and the second is a guess. Smearing data
# that was never smeared biases the radius as surely as ignoring real
# smearing did in Part 2, in the opposite direction.
for label, kwargs in (("mode 'none'", {}), ('a guessed 10% pinhole', {'dq_over_q': 0.10})):
    fitter = sphere_fitter(plain)
    fitter.set_resolution('pinhole' if kwargs else 'none', **kwargs)
    result = fitter.fit(engine='bumps', method='amoeba')
    radius = result['parameters']['radius']
    error = radius['value'] - TRUTH['radius']
    print(
        f'\n  {label}: radius = {radius["value"]:.3f} A '
        f'({error:+.3f} A from truth), chi^2/dof = {result["reduced_chisq"]:.3f}'
    )


# ============================================================================
# Part 4: slit geometry (USANS)
# ============================================================================
banner('Part 4: slit smearing')

# slit_length is the slit dimension along q, an absolute width in 1/Ang -- not
# a relative one. It is required: sasmodels smears along the slit length, so a
# width on its own has nothing to integrate over. Omit slit_width for the usual
# long-slit geometry.
fitter = sphere_fitter(measured)
fitter.set_resolution('slit', slit_length=0.05)
print(f'Active setting: {fitter.get_resolution()}')
result = fitter.fit(engine='bumps', method='amoeba')
print(f'  radius = {result["parameters"]["radius"]["value"]:.3f} A')
print(f'  chi^2/dof = {result["reduced_chisq"]:.3f} -- slit geometry is wrong for this data,')
print('  which is exactly what a bad goodness of fit is for.')

# ============================================================================
# Part 5: your dataset is untouched, and the choice is recorded
# ============================================================================
banner('Part 5: provenance')

# The setting is applied to a copy made for evaluation. Plots, CSV export,
# P(r) inversion and data_ops all keep seeing the dataset you loaded.
fitter = sphere_fitter(measured)
fitter.set_resolution('pinhole', dq_over_q=0.25)
fitter.fit(engine='bumps', method='amoeba')
print(f"fitter.data.dx still the file's own column: {np.allclose(fitter.data.dx, measured.dx)}")
print(f'  the fit ran with a 0.25 pinhole; the data still says {SIGMA}_q/q = 0.10')

# A fitted parameter set only means something alongside the smearing that
# produced it, so the saved CSV records the mode in its header.
with tempfile.TemporaryDirectory() as folder:
    saved = os.path.join(folder, 'fit.csv')
    fitter.save_results(saved)
    with open(saved, encoding='utf-8') as handle:
        for line in handle:
            if line.startswith('# Resolution'):
                print(f'  saved CSV header: {line.strip()}')
                break

# ============================================================================
# Part 6: the guardrails
# ============================================================================
banner('Part 6: what gets refused')

fitter = sphere_fitter(measured)
fitter.set_resolution('pinhole', dq_over_q=0.10)

bad_calls = [
    ('gaussian', {}),  # not one of the four modes
    ('pinhole', {}),  # a pinhole needs a width
    ('slit', {'slit_width': 0.01}),  # a slit needs a length
    ('data', {'dq_over_q': 0.1}),  # 'data' takes no width at all
    ('pinhole', {'dq_over_q': 0}),  # a zero width is 'none', not a pinhole
]
for mode, kwargs in bad_calls:
    arguments = ''.join(f', {name}={value!r}' for name, value in kwargs.items())
    try:
        fitter.set_resolution(mode, **kwargs)
    except ValueError as error:
        print(f"\n  set_resolution('{mode}'{arguments})\n    -> {error}")

# Every check runs before any state is touched, so a rejected call leaves the
# fitter exactly as it was.
print(f'\nStill active after all of those: {fitter.get_resolution()}')

# ============================================================================
# One convention worth repeating
# ============================================================================
banner('Units')

print(f'dq_over_q is {SIGMA}_q/q, a Gaussian 1-{SIGMA}. It is NOT FWHM.')
print('  If your instrument scientist quotes dQ/Q as a full width at half')
print('  maximum, divide by about 2.355 before passing it here.')
print(f'slit_length and slit_width are absolute widths in {INVERSE_ANGSTROM}, constant in Q.')

print(f'\n{OK} Resolution example completed successfully!')
