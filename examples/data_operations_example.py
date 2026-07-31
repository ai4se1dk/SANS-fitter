"""
Example: Dataset arithmetic with sans_fitter.data_ops

This script demonstrates how to:
1. Simulate a "sample" measurement (sphere scattering + flat instrument
   background) and a separate "background" measurement
2. Load both files with data_ops.load()
3. Subtract the background run and apply a transmission correction
4. Feed the result into SANSFitter with set_data() and fit it
5. Plot and save the results

The arithmetic functions (add, subtract, multiply, divide) accept either two
datasets on the same Q grid or a dataset and a scalar, and return a new,
fit-ready Data1D with propagated uncertainties.
"""

import numpy as np
from sasmodels.core import load_model
from sasmodels.data import empty_data1D
from sasmodels.direct_model import DirectModel

from sans_fitter import SANSFitter, data_ops

# ============================================================================
# Part 1: Simulate the two measurements
# ============================================================================

print('=' * 80)
print('Part 1: Simulating sample and background measurements')
print('=' * 80)

rng = np.random.default_rng(seed=42)
q = np.logspace(np.log10(0.008), np.log10(0.35), 80)

# Ideal sphere scattering (radius 60 Å) computed with sasmodels itself
kernel = load_model('sphere', dtype='single', platform='dll')
calculator = DirectModel(empty_data1D(q), kernel)
i_sphere = calculator(radius=60.0, scale=0.005, background=0.0, sld=4.0, sld_solvent=1.0)

BACKGROUND_LEVEL = 0.08  # flat instrument/solvent background
TRANSMISSION = 0.8  # sample transmission factor

# "Sample" run: sphere signal + background, with 3% counting noise
i_sample = i_sphere + BACKGROUND_LEVEL
di_sample = 0.03 * i_sample
i_sample = i_sample + rng.normal(0.0, di_sample)

# "Background" run: background only, with its own noise
i_bkg = np.full_like(q, BACKGROUND_LEVEL)
di_bkg = 0.03 * i_bkg
i_bkg = i_bkg + rng.normal(0.0, di_bkg)

for filename, intensity, error in [
    ('example_sample.csv', i_sample, di_sample),
    ('example_background.csv', i_bkg, di_bkg),
]:
    with open(filename, 'w') as f:
        f.write('Q,I,dI\n')
        for row in zip(q, intensity, error):
            f.write(f'{row[0]},{row[1]},{row[2]}\n')
    print(f'  wrote {filename}')

# ============================================================================
# Part 2: Load and combine the datasets
# ============================================================================

print('\n' + '=' * 80)
print('Part 2: Background subtraction and transmission correction')
print('=' * 80)

# data_ops.load() is the standalone equivalent of SANSFitter.load_data():
# it returns a fit-ready Data1D instead of storing it on a fitter.
sample = data_ops.load('example_sample.csv')
background = data_ops.load('example_background.csv')

# subtract(a, b) returns a - b with dy = sqrt(dy_a^2 + dy_b^2).
# Both datasets must share the same Q grid (within 1%); mismatched grids
# raise a ValueError naming both datasets and their Q ranges.
net = data_ops.subtract(sample, background)

# Scalar operations work too: divide by the transmission to correct the
# intensity scale (y and dy are divided, the Q grid is untouched).
net = data_ops.divide(net, TRANSMISSION)

print(f'\nResult title: {net.title}')
print(f'Process history: {[p.description for p in net.process]}')

# ============================================================================
# Part 3: Fit the corrected dataset
# ============================================================================

print('\n' + '=' * 80)
print('Part 3: Fitting the background-subtracted data')
print('=' * 80)

fitter = SANSFitter()
fitter.set_data(net)  # injection point for in-memory datasets
fitter.set_model('sphere')

fitter.set_param('radius', value=40.0, min=10.0, max=150.0, vary=True)
fitter.set_param('scale', value=0.001, min=1e-4, max=1.0, vary=True)
fitter.set_param('background', value=0.001, min=0.0, max=1.0, vary=True)
fitter.set_param('sld', value=4.0, vary=False)
fitter.set_param('sld_solvent', value=1.0, vary=False)

result = fitter.fit(engine='bumps', method='amoeba')

print('\nGround truth vs fit:')
print(f'  radius: 60.0 Å   -> fitted {result["parameters"]["radius"]["value"]:.1f} Å')
print(f'  scale:  {0.005 / TRANSMISSION:.5f}  -> fitted {result["parameters"]["scale"]["value"]:.5f}')

# ============================================================================
# Part 4: Plot and save
# ============================================================================

print('\nGenerating plot...')
fitter.plot_results(show_residuals=True, log_scale=True)

fitter.save_results('background_subtracted_fit.csv')

print('\n✓ Example completed successfully!')
