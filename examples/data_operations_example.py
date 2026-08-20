"""
Example: Dataset arithmetic with sans_fitter.data_ops

This script demonstrates how to:
1. Generate a matched "sample" and "background" pair with examples.simulate_pair()
2. Subtract the background run and apply a transmission correction
3. Feed the result into SANSFitter with set_data() and fit it
4. Plot and save the results

The arithmetic functions (add, subtract, multiply, divide) accept either two
datasets on the same Q grid or a dataset and a scalar, and return a new,
fit-ready Data1D with propagated uncertainties.
"""

from sans_fitter import SANSFitter, data_ops, examples

# ============================================================================
# Part 1: Simulate the two measurements
# ============================================================================

print('=' * 80)
print('Part 1: Simulating sample and background measurements')
print('=' * 80)

RADIUS = 60.0  # Å
SCALE = 0.005
BACKGROUND_LEVEL = 0.08  # flat instrument/solvent background
TRANSMISSION = 0.8  # sample transmission factor

# simulate_pair() puts both runs on an identical Q grid, which is what the
# arithmetic below requires. The sample is sphere signal + flat background;
# the background run is that flat level alone, with its own independent noise.
# Nothing is written to disk.
sample, background = examples.simulate_pair(
    'sphere',
    background_level=BACKGROUND_LEVEL,
    qmin=0.008,
    qmax=0.35,
    npoints=80,
    noise=0.03,
    seed=42,
    radius=RADIUS,
    scale=SCALE,
    sld=4.0,
    sld_solvent=1.0,
)

print(f'  sample:     {len(sample.x)} points, generated with radius = {RADIUS} Å')
print(f'  background: {len(background.x)} points, flat at {BACKGROUND_LEVEL}')

# ============================================================================
# Part 2: Combine the datasets
# ============================================================================

print('\n' + '=' * 80)
print('Part 2: Background subtraction and transmission correction')
print('=' * 80)

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

# simulate_pair() records what generated the sample on sample.truth, so the
# comparison below reads the ground truth rather than repeating a literal.
print('\nGround truth vs fit:')
print(
    f'  radius: {sample.truth["radius"]:.1f} Å   -> '
    f'fitted {result["parameters"]["radius"]["value"]:.1f} Å'
)
print(
    f'  scale:  {sample.truth["scale"] / TRANSMISSION:.5f}  -> '
    f'fitted {result["parameters"]["scale"]["value"]:.5f}'
)

# ============================================================================
# Part 4: Plot and save
# ============================================================================

print('\nGenerating plot...')
fitter.plot_results(show_residuals=True, log_scale=True)

fitter.save_results('background_subtracted_fit.csv')

print('\n✓ Example completed successfully!')
