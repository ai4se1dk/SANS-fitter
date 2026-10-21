"""
Example: scattering from a real-space structure (Generic Scattering Calculator)

GenericScatteringCalculator is the nuclear, 1D part of SasView's Generic
Scattering Calculator. It reads a structure from a PDB, SLD or VTK file and
computes the orientationally averaged I(q) with the exact Debye formula.

This script demonstrates how to:
1. Write a sphere of pixels in a box of solvent as an SLD file
2. Load it, look at its points, and compute I(q) in D2O
3. Check the curve against sasmodels' analytic sphere
4. Fit the computed curve with SANSFitter
5. Compute a PDB structure (C60) in vacuum
6. Load a VTK mesh, whose cells become points

Run from the repository root:  python examples/structure_calculator_example.py
"""

import os
import tempfile

import numpy as np
from sasmodels.core import load_model
from sasmodels.direct_model import call_kernel

from sans_fitter import GenericScatteringCalculator, SANSFitter

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
RADIUS, STEP = 30.0, 3.0  # Å
SLD_PARTICLE, SLD_D2O = 1.0, 6.34  # 10^-6 Å^-2, the unit every sasmodels SLD uses

# ============================================================================
# Part 1: write a sphere of pixels as an SLD file
# ============================================================================
# SLD files store x, y, z in Å and the SLD in Å^-2, after one header line.
# Pixels outside the sphere carry the solvent's SLD: at that solvent they do
# not scatter, and they do not count towards the normalisation volume.
n = int(RADIUS / STEP) + 1
axis = np.arange(-n, n + 1) * STEP
xyz = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
inside = (xyz**2).sum(axis=1) <= RADIUS**2
sld = np.where(inside, SLD_PARTICLE, SLD_D2O) * 1e-6
path = os.path.join(tempfile.mkdtemp(), 'sphere.sld')
np.savetxt(path, np.column_stack([xyz, sld]), header='x y z sld', comments='')

# ============================================================================
# Part 2: load, look, compute
# ============================================================================
gsc = GenericScatteringCalculator()
gsc.load_structure(path)  # the load line prints the lattice step and pixel volume
gsc.plot_points()  # SasView's "Draw Points"; solvent pixels are drawn yellow
gsc.set_params(sld_solvent=SLD_D2O)  # 10^-6 Å^-2, like fit.constrain('d2o.sld_solvent', 6.34)
gsc.describe()

profile = gsc.calculate()  # pixels on a lattice use the exact FFT path
profile.describe()
gsc.plot_profile()

# ============================================================================
# Part 3: compare with the analytic sphere
# ============================================================================
# The normalisation volume is the volume of the points that scatter, so at
# scale = 1 the curve matches sasmodels' sphere at scale = 1 (volume fraction 1).
kernel = load_model('sphere').make_kernel([profile.q])
analytic = call_kernel(
    kernel,
    {'radius': RADIUS, 'sld': SLD_PARTICLE, 'sld_solvent': SLD_D2O, 'scale': 1, 'background': 0},
)
low_q = profile.q * RADIUS <= 3
deviation = profile.intensity[low_q] / analytic[low_q] - 1
print(
    f'\nPixel sphere vs analytic sphere for qR <= 3: {deviation.min():+.2%} to {deviation.max():+.2%}'
)

# ============================================================================
# Part 4: fit the computed curve
# ============================================================================
# The curve is theory: it has no dI and no dQ, so state the weighting and the
# resolution explicitly.
fitter = SANSFitter()
fitter.set_data(profile.to_data1d())
fitter.set_weighting('abs')  # sigma = |I|: equal relative weight across decades
fitter.set_resolution('none')
fitter.set_model('sphere')
fitter.set_param('sld', value=SLD_PARTICLE, vary=False)
fitter.set_param('sld_solvent', value=SLD_D2O, vary=False)
fitter.set_param('background', value=0.0, vary=False)
fitter.set_param('radius', value=25, min=10, max=60, vary=True)
fitter.set_param('scale', value=1.0, min=0.5, max=2, vary=True)
result = fitter.fit(method='lm')
print(f'Fitted radius: {result["parameters"]["radius"]["value"]:.2f} Å (pixel sphere: {RADIUS} Å)')

# ============================================================================
# Part 5: a PDB structure in vacuum
# ============================================================================
# Atomic volumes come from elemental densities, as in SasView. In vacuum that
# does not matter; in a solvent it makes I(q) quantitatively wrong, and
# calculate() says by how much.
c60 = GenericScatteringCalculator(os.path.join(DATA, 'c60.pdb'))
c60.set_q_grid(1e-2, 2.0, 100)
c60.plot_points()  # coloured by element, with the CONECT bonds
c60_profile = c60.calculate()
rg = c60.structure.radius_of_gyration()
print(
    f'C60: Rg {rg.value:.3f} Å; I(q) = {c60_profile.intensity[0]:.4g} 1/cm at '
    f'q = {c60_profile.q[0]:g} Å^-1, {c60_profile.intensity[-1]:.3g} 1/cm at q = {c60_profile.q[-1]:g} Å^-1'
)
c60.plot_profile()

# ============================================================================
# Part 6: a VTK mesh
# ============================================================================
# Every cell becomes one point at its volume centroid, carrying the cell's
# volume and SLD. calculate() warns when q * cell size exceeds 1.
cube = GenericScatteringCalculator(os.path.join(DATA, 'five_tetrahedra_cube.vtk'))
cube.describe()
print(
    f'Cell volumes: {cube.structure.volume.round(4).tolist()} (sum {cube.structure.volume.sum():.4g})'
)

print('\n' + '=' * 80)
print('Summary: Generic Scattering Calculator Key Points')
print('=' * 80)
print("""
✓ load_structure() reads PDB, SLD and VTK; plot_points() shows the points
✓ sld_solvent is in 10^-6 Å^-2, like every sasmodels SLD parameter
✓ calculate() returns a ScatteringProfile; lattices use an exact FFT path
✓ V_norm is the volume of the points that scatter, so scale is a volume fraction
✓ profile.to_data1d() feeds SANSFitter; set the weighting and resolution yourself
✓ PDB curves in a solvent use elemental volumes and are only qualitative
""")

print('\n✓ Structure calculator example completed successfully!')
