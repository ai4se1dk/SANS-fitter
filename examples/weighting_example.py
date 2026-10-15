"""
Example: data weighting options

Every engine minimises chi^2 = sum(((I - I_model) / sigma) ** 2). Which sigma
divides each residual is a stated choice, made with set_weighting(). The four
modes mirror SasView's Fit Page: *Use dI Data* / *Use |sqrt(I Data)|* /
*Use |I Data|* / *No Weighting*.

This script demonstrates how to:
1. Fit the same dataset under all four modes and compare what comes back
2. Fit noise-free data, which has no dI, with the bumps engine
3. Read the weighting back off a fit result and a saved CSV

Run from the repository root:  python examples/weighting_example.py
"""

import os
import tempfile
import warnings

from sans_fitter import SANSFitter, examples, set_verbosity

# The package picks ASCII spellings for these when the console cannot encode
# them, which is what keeps this script runnable on a legacy Windows terminal.
from sans_fitter.console import CHI_SQUARED

TRUTH = {'radius': 50.0, 'scale': 1.0, 'background': 0.001, 'sld': 1.0, 'sld_solvent': 6.0}


def sphere_fitter(data, mode):
    """A sphere fitter on *data* with only the radius free, under *mode*."""
    fitter = SANSFitter()
    fitter.set_data(data)
    fitter.set_model('sphere')
    fitter.set_resolution('none')  # simulated without dQ
    for name, value in TRUTH.items():
        fitter.set_param(name, value=value, vary=False)
    fitter.set_param('radius', value=40, min=10, max=100, vary=True)
    fitter.set_weighting(mode)
    return fitter


def compare_modes():
    print('\n1. One dataset, four weightings')
    print('-' * 60)
    data = examples.simulate('sphere', noise=0.03, seed=1, **TRUTH)
    print(f'{"mode":<6} {"radius":>18} {CHI_SQUARED + "/dof":>12}  weighting_note')
    for mode in ('dI', 'sqrt', 'abs', 'none'):
        result = sphere_fitter(data, mode).fit()
        radius = result['parameters']['radius']['formatted']
        print(
            f'{mode:<6} {radius:>18} {result["reduced_chisq"]:>12.4g}  {result["weighting_note"]}'
        )
    print(
        f'Only dI gives a {CHI_SQUARED}/dof near 1: it is the only mode whose sigma is the\n'
        'real measurement uncertainty. The others are for comparing fits of the\n'
        'same data under the same mode - and their stderr inherits that caveat.'
    )


def noise_free():
    print('\n2. Noise-free data has no dI')
    print('-' * 60)
    data = examples.simulate('sphere', noise=0, **TRUTH)
    fitter = sphere_fitter(data, 'dI')
    try:
        fitter.fit()
    except ValueError as error:
        print(f'Under dI the bumps engine refuses it:\n  {error}')

    fitter.set_weighting('none')
    result = fitter.fit()
    print(
        f"Under 'none' it fits: radius = {result['parameters']['radius']['value']:.4f} "
        f'(truth {TRUTH["radius"]}), {CHI_SQUARED} = {result["chisq"]:.3g}'
    )


def read_back():
    print('\n3. The weighting travels with the result')
    print('-' * 60)
    data = examples.simulate('sphere', noise=0.03, seed=1, **TRUTH)
    fitter = sphere_fitter(data, 'sqrt')
    fitter.fit()
    print(f'get_weighting():       {fitter.get_weighting()}')
    print(f'fit result note:       {fitter.fit_result["weighting_note"]}')

    path = os.path.join(tempfile.mkdtemp(), 'weighted_fit.csv')
    fitter.save_results(path)
    with open(path, encoding='utf-8') as handle:
        header = [line.rstrip() for line in handle if line.startswith('# Weighting')]
    print('CSV header lines:')
    for line in header:
        print(f'  {line}')
    print("The dI_exp column is still the file's own dI; fitter.data is never modified.")


if __name__ == '__main__':
    set_verbosity('quiet')
    # simulate() without dq carries no resolution columns; 'none' is stated
    # explicitly above, so nothing here is worth a warning.
    warnings.simplefilter('ignore')
    compare_modes()
    noise_free()
    read_back()
