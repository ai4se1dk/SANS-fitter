"""Tests for the Generic Scattering Calculator (sans_fitter.structure)."""

import copy
import json
import os
import pickle
import warnings

import numpy as np
import pytest

from sans_fitter import GenericScatteringCalculator, SANSFitter
from sans_fitter.structure import (
    CalculationInterrupted,
    Structure,
    StructureError,
    read_pdb,
    read_sld,
    read_vtk,
)
from sans_fitter.structure.debye import debye_lattice, debye_pairs
from sans_fitter.structure.readers import cell_geometry

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'examples', 'data')
C60 = os.path.join(DATA, 'c60.pdb')
CUBE_VTK = os.path.join(DATA, 'five_tetrahedra_cube.vtk')
Q = np.r_[0.0, np.geomspace(1e-3, 3.0, 25)]


def brute(q, xyz, w):
    r = np.sqrt(((xyz[:, None] - xyz[None]) ** 2).sum(-1))
    return np.array([(w[:, None] * w[None] * np.sinc(qi * r / np.pi)).sum() for qi in q])


def close(a, b, w):
    """Kernel agreement on the scale of the terms, not of D (which can be tiny)."""
    return np.max(np.abs(a - b)) <= 1e-12 * np.abs(w).sum() ** 2


def sphere_pixels(radius=30.0, step=3.0, sld=1.0, sld_solvent=6.34):
    """A sphere of pixels in a box of solvent pixels (SLD in 10^-6 1/Ang^2)."""
    n = int(np.ceil(radius / step)) + 1
    axis = np.arange(-n, n + 1) * step
    xyz = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    inside = (xyz**2).sum(1) <= radius**2
    values = np.where(inside, sld, sld_solvent) * 1e-6
    return Structure.from_arrays(xyz, values, np.full(len(xyz), step**3))


def quiet_calculator(structure=None, qmax=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gsc = GenericScatteringCalculator(structure)
    if qmax is not None:
        gsc.set_q_grid(qmax=qmax)
    return gsc


# ============================================================================
# Kernels
# ============================================================================


class TestKernels:
    def test_two_points_and_q_zero(self):
        xyz = np.array([[0.0, 0, 0], [0, 0, 2.5]])
        w = np.array([2.0, -0.5])
        expected = (w**2).sum() + 2 * w[0] * w[1] * np.sinc(Q * 2.5 / np.pi)
        assert np.allclose(debye_pairs(Q, xyz, w), expected, rtol=0, atol=1e-14)
        assert debye_pairs([0.0], xyz, w)[0] == pytest.approx(w.sum() ** 2)

    def test_matches_brute_force_with_coincident_points_and_any_block(self):
        rng = np.random.default_rng(0)
        xyz = rng.normal(size=(120, 3)) * 10
        xyz[5] = xyz[7]
        w = rng.normal(size=120)
        reference = brute(Q, xyz, w)
        for block in (1, 7, 120, 1000, None):
            assert close(debye_pairs(Q, xyz, w, block=block), reference, w)

    def test_translation_and_rotation_invariance(self):
        rng = np.random.default_rng(1)
        xyz, w = rng.normal(size=(50, 3)) * 8, rng.normal(size=50)
        theta = 0.7
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        )
        moved = xyz @ rotation.T + [10, -3, 4]
        assert close(debye_pairs(Q, xyz, w), debye_pairs(Q, moved, w), w)

    def test_lattice_path_equals_pair_sum(self):
        rng = np.random.default_rng(2)
        grid = rng.normal(size=(8, 6, 5)) * (rng.random((8, 6, 5)) < 0.6)
        step = (3.0, 2.0, 4.5)
        index = np.argwhere(grid != 0)
        w = grid[grid != 0]
        lattice = debye_lattice(Q, grid, step)
        assert close(lattice, debye_pairs(Q, index * np.array(step), w), w)
        assert np.all(lattice >= -1e-12 * np.abs(w).sum() ** 2)  # D is an average of |A|^2

    def test_progress_can_stop_the_pair_loop(self):
        xyz = np.random.default_rng(3).normal(size=(50, 3))
        with pytest.raises(CalculationInterrupted):
            debye_pairs(Q, xyz, np.ones(50), block=5, progress=lambda f: f > 0.3)


# ============================================================================
# Physics
# ============================================================================


class TestPhysics:
    def test_sasview_cuboid_regression_values(self):
        """10x10x10 lattice, steps 6/3/6 Ang, SLD 0.1 1/Ang^2, as in SasView's 1D test."""
        index = np.stack(np.meshgrid(*[np.arange(10)] * 3, indexing='ij'), -1).reshape(-1, 3)
        xyz = index * np.array([6.0, 3.0, 6.0])
        structure = Structure.from_arrays(xyz, np.full(1000, 0.1), np.full(1000, 108.0))
        gsc = quiet_calculator(structure)
        gsc.set_q([0.01, 0.03])
        profile = gsc.calculate()
        assert profile.method == 'lattice'
        expected = 1e8 * brute([0.01, 0.03], xyz, np.full(1000, 10.8)) / 108000.0
        assert np.allclose(profile.intensity, expected, rtol=1e-10)
        assert np.allclose(profile.intensity, [1.05619e11, 8.82750e10], rtol=1e-5)

    def test_sphere_of_pixels_matches_sasmodels_at_scale_one(self):
        from sasmodels.core import load_model
        from sasmodels.direct_model import call_kernel

        gsc = quiet_calculator(sphere_pixels())
        gsc.set_params(sld_solvent=6.34)
        gsc.set_q(np.geomspace(1e-3, 0.1, 40))
        profile = gsc.calculate()
        kernel = load_model('sphere').make_kernel([gsc.q])
        analytic = call_kernel(
            kernel, {'radius': 30, 'sld': 1.0, 'sld_solvent': 6.34, 'scale': 1, 'background': 0}
        )
        low = gsc.q * 30 <= 3
        assert np.allclose(profile.intensity[low], analytic[low], rtol=0.02)

    @pytest.mark.slow
    def test_pair_loop_agrees_with_lattice_path_on_the_sphere(self):
        structure = sphere_pixels()
        w = (structure.sld - 6.34e-6) * structure.volume
        active = w != 0
        index = structure.lattice.indices[active]
        grid = np.zeros(tuple(index.max(0) + 1))
        np.add.at(grid, tuple(index.T), w[active])
        q = np.geomspace(1e-3, 0.3, 10)
        pairs = debye_pairs(q, structure.xyz[active], w[active])
        assert close(debye_lattice(q, grid, structure.lattice.step), pairs, w)


# ============================================================================
# Normalisation and warnings
# ============================================================================


class TestNormalisation:
    def setup_method(self):
        self.xyz = np.array([[0.0, 0, 0], [0, 0, 5], [0, 5, 0], [5, 0, 0]])
        self.structure = Structure.from_arrays(
            self.xyz, np.array([2, 2, 1, 1]) * 1e-6, np.array([1.0, 2.0, 3.0, 4.0])
        )

    def test_active_volume_excludes_matched_points(self):
        gsc = quiet_calculator(self.structure, qmax=0.2)
        gsc.set_params(sld_solvent=1.0)
        profile = gsc.calculate()
        assert profile.normalisation_volume == pytest.approx(3.0)
        assert profile.n_zero_contrast == 2
        w = np.array([1e-6, 2e-6])
        expected = 1e8 * debye_pairs(gsc.q, self.xyz[:2], w) / 3.0
        assert np.allclose(profile.intensity, expected, rtol=1e-12)

    def test_explicit_volume_and_auto_restore(self):
        gsc = quiet_calculator(self.structure, qmax=0.2)
        auto = gsc.calculate()
        gsc.set_params(normalisation_volume=20.0)
        explicit = gsc.calculate()
        assert explicit.normalisation == 'explicit'
        assert np.allclose(explicit.intensity * 20.0, auto.intensity * 10.0)
        gsc.set_params(normalisation_volume='auto')
        assert gsc.calculate().normalisation_volume == pytest.approx(10.0)

    def test_scale_and_background(self):
        gsc = quiet_calculator(self.structure, qmax=0.2)
        base = gsc.calculate().intensity
        gsc.set_params(scale=2.0, background=0.5)
        assert np.allclose(gsc.calculate().intensity, 2 * base + 0.5)

    def test_all_zero_contrast_returns_the_background(self):
        structure = Structure.from_arrays(self.xyz, np.full(4, 2e-6), np.ones(4))
        gsc = quiet_calculator(structure, qmax=0.2)
        gsc.set_params(sld_solvent=2.0, background=0.1)
        with pytest.warns(UserWarning, match='Every point matches the solvent'):
            profile = gsc.calculate()
        assert np.all(profile.intensity == 0.1)
        assert np.isnan(profile.normalisation_volume)
        gsc.set_params(normalisation_volume=7.0)
        with pytest.warns(UserWarning):
            assert gsc.calculate().normalisation_volume == 7.0

    def test_near_match_warns(self):
        structure = Structure.from_arrays(
            self.xyz, np.array([1e-6, 1e-6 + 1e-17, 3e-6, 3e-6]), np.ones(4)
        )
        gsc = quiet_calculator(structure, qmax=0.2)
        gsc.set_params(sld_solvent=1.0)
        with pytest.warns(UserWarning, match='almost match the solvent'):
            profile = gsc.calculate()
        assert any('almost match' in note for note in profile.notes)

    def test_solvent_in_wrong_units_warns(self):
        with pytest.warns(UserWarning, match='Did you mean 6.4'):
            quiet_calculator().set_params(sld_solvent=6.4e-6)

    def test_cell_size_check_for_pixels(self):
        gsc = quiet_calculator(sphere_pixels(radius=6.0, step=3.0))
        gsc.set_params(sld_solvent=6.34)
        gsc.set_q_grid(qmax=0.5)
        with pytest.warns(UserWarning, match='q_max \\* cell size'):
            gsc.calculate()  # 0.5 * 3 > 1
        gsc.set_q_grid(qmax=0.3)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            gsc.calculate()  # 0.3 * 3 < 1: no warning

    def test_atoms_in_solvent_warn_with_the_volume_ratio(self):
        gsc = quiet_calculator(C60)
        gsc.set_params(sld_solvent=6.34)
        with pytest.warns(UserWarning, match='times a protein-like estimate'):
            gsc.calculate()


# ============================================================================
# Readers
# ============================================================================


def pdb_line(serial, name, residue, x, element, record='ATOM', altloc=' ', occupancy=1.0, resseq=1):
    return (
        f'{record:<6}{serial:>5} {name:<4}{altloc}{residue:>3} A{resseq:>4}    '
        f'{x:8.3f}{0.0:8.3f}{0.0:8.3f}{occupancy:6.2f}{0.0:6.2f}          {element:>2}'
    )


class TestPdb:
    def test_c60(self):
        s = read_pdb(C60)
        assert (s.kind, s.n_points) == ('atoms', 60)
        assert np.allclose(s.xyz[0], [-0.733, -1.008, 3.326])
        assert set(s.symbols) == {'C'}
        assert s.bonds.tolist() == [[0, 1], [0, 4], [1, 2], [2, 3], [3, 4]]
        assert s.radius_of_gyration().value == pytest.approx(3.552, abs=1e-3)
        assert s.radius_of_gyration('mass').value == pytest.approx(3.552, abs=1e-3)
        from periodictable import nsf

        assert s.sld[0] == pytest.approx(nsf.neutron_sld('C')[0] * 1e-6)

    def test_policies(self, tmp_path):
        lines = [
            pdb_line(1, 'N', 'ALA', 0.0, 'N'),
            pdb_line(2, 'CA', 'ALA', 1.0, 'C', altloc='B'),
            pdb_line(3, 'CA', 'ALA', 1.1, 'C', altloc='A'),
            pdb_line(5, 'FE', 'HEM', 2.0, 'FE', record='HETATM', resseq=2),
            pdb_line(6, 'SE', 'MSE', 3.0, 'SE', record='HETATM', resseq=3, occupancy=0.5),
            pdb_line(7, 'O', 'HOH', 4.0, 'O', record='HETATM', resseq=4),
            'CONECT    1    2    5',
            'CONECT    5    6',
            'MODEL        2',
            pdb_line(8, 'N', 'ALA', 9.0, 'N'),
        ]
        path = tmp_path / 'policies.pdb'
        path.write_text('MODEL        1\n' + '\n'.join(lines) + '\n')

        default = read_pdb(path)
        assert default.symbols.tolist() == ['N', 'C']  # altloc B came first and is kept
        assert default.bonds.tolist() == [[0, 1]]  # bonds to skipped atoms dropped
        notes = ' '.join(default.notes)
        assert 'first of 2 models' in notes and 'HEM x1' in notes and 'MSE x1' in notes
        assert '1 water' in notes and "alternate location 'B'" in notes

        with_het = read_pdb(path, include_hetatm=True)
        assert with_het.symbols.tolist() == ['N', 'C', 'Fe', 'Se']  # case-normalised, no water
        assert with_het.bonds.tolist() == [[0, 1], [0, 2], [2, 3]]
        assert 'occupancy below 1' in ' '.join(with_het.notes)

    def test_unknown_element_is_an_error(self, tmp_path):
        path = tmp_path / 'bad.pdb'
        path.write_text(pdb_line(1, 'XX', 'UNK', 0.0, 'XX') + '\n')
        with pytest.raises(StructureError, match='Xx'):
            read_pdb(path)


class TestSld:
    def lattice(self, n=3, step=(2.0, 3.0, 4.0)):
        index = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing='ij'), -1).reshape(-1, 3)
        return index * np.array(step)

    def write(self, path, rows):
        np.savetxt(path, rows, header='x y z sld ...', comments='')
        return path

    def test_four_seven_and_eight_columns(self, tmp_path):
        xyz = self.lattice()
        sld = np.full((len(xyz), 1), 2e-6)
        four = read_sld(self.write(tmp_path / 'a.sld', np.hstack([xyz, sld])))
        assert np.allclose(four.volume, 24.0) and four.lattice.step == (2.0, 3.0, 4.0)
        magnetic = np.hstack([xyz, sld, np.ones((len(xyz), 3)) * 1e-7])
        seven = read_sld(self.write(tmp_path / 'b.sld', magnetic))
        assert 'Magnetic SLD columns are ignored.' in seven.notes
        eight = read_sld(
            self.write(tmp_path / 'c.sld', np.hstack([magnetic, np.full((len(xyz), 1), 5.0)]))
        )
        assert np.allclose(eight.volume, 5.0)

    def test_single_row_and_plane_with_volumes(self, tmp_path):
        one = read_sld(self.write(tmp_path / 'one.sld', [[0, 0, 0, 1e-6, 0, 0, 0, 7.0]]))
        assert one.n_points == 1 and one.volume[0] == 7.0
        plane = np.hstack(
            [self.lattice()[:9], np.full((9, 1), 1e-6), np.zeros((9, 3)), np.ones((9, 1))]
        )
        assert read_sld(self.write(tmp_path / 'plane.sld', plane)).n_points == 9

    def test_shuffled_imprecise_and_sparse_lattices(self, tmp_path):
        xyz = self.lattice() + 1e-7  # written with limited precision
        rng = np.random.default_rng(0)
        rows = np.hstack([xyz, np.full((len(xyz), 1), 1e-6)])[rng.permutation(len(xyz))]
        shuffled = read_sld(self.write(tmp_path / 'shuffled.sld', rows))
        assert np.allclose(shuffled.lattice.step, (2.0, 3.0, 4.0))
        sparse = read_sld(self.write(tmp_path / 'sparse.sld', rows[::2]))
        assert sparse.lattice is not None

    def test_refusals(self, tmp_path):
        irregular = np.random.default_rng(0).random((10, 3)) * 10
        with pytest.raises(StructureError, match='column 8'):
            read_sld(
                self.write(tmp_path / 'irregular.sld', np.hstack([irregular, np.ones((10, 1))]))
            )
        with pytest.raises(StructureError, match='magnetic SLD only'):
            read_sld(self.write(tmp_path / 'mag.sld', np.zeros((4, 6))))
        duplicated = np.array([[0, 0, 0, 1e-6], [0, 0, 0, 1e-6]])
        with pytest.raises(StructureError, match='same position'):
            read_sld(self.write(tmp_path / 'dup.sld', duplicated))


UNIT_CELLS = {
    # type: (vertices in VTK order, volume, centroid)
    10: ([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], 1 / 6, [0.25, 0.25, 0.25]),
    11: (
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        1.0,
        [0.5, 0.5, 0.5],
    ),
    12: (
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        1.0,
        [0.5, 0.5, 0.5],
    ),
    13: (
        [[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1]],
        0.5,
        [1 / 3, 1 / 3, 0.5],
    ),
    14: ([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]], 1 / 3, [0.5, 0.5, 0.25]),
}


def vtk_text(points, cells, types, scalars, where='CELL_DATA', layout='legacy', extra=''):
    points = np.asarray(points, dtype=float)
    out = ['# vtk DataFile Version 3.0', 'test', 'ASCII', 'DATASET UNSTRUCTURED_GRID']
    out.append(f'POINTS {len(points)} float')
    out += [' '.join(map(str, p)) for p in points]
    if layout == 'legacy':
        out.append(f'CELLS {len(cells)} {sum(len(c) + 1 for c in cells)}')
        out += [' '.join(map(str, [len(c), *c])) for c in cells]
    else:
        offsets = np.cumsum([0] + [len(c) for c in cells])
        out.append(f'CELLS {len(offsets)} {offsets[-1]}')
        out += ['OFFSETS vtktypeint64', ' '.join(map(str, offsets)), 'CONNECTIVITY vtktypeint64']
        out.append(' '.join(str(i) for c in cells for i in c))
    out.append(f'CELL_TYPES {len(types)}')
    out += [str(t) for t in types]
    out.append(f'{where} {len(scalars)}')
    out += ['SCALARS nuclear float 1', 'LOOKUP_TABLE default', ' '.join(map(str, scalars))]
    return '\n'.join(out) + '\n' + extra


class TestVtk:
    def test_sasview_cube(self):
        s = read_vtk(CUBE_VTK)
        assert (s.kind, s.n_points) == ('cells', 5)
        assert s.volume.sum() == pytest.approx(8.0)
        assert np.allclose(s.sld, 1.0)
        assert np.allclose(s.xyz[0], 0.0)
        assert 'Magnetic SLD fields are ignored.' in s.notes

    @pytest.mark.parametrize('cell_type', sorted(UNIT_CELLS))
    def test_unit_cells_translated_and_rotated(self, cell_type):
        vertices, volume, centroid = UNIT_CELLS[cell_type]
        vertices = np.array(vertices, dtype=float)
        theta = 0.4
        rotation = np.array(
            [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
        )
        for moved, centre in (
            (vertices + 10, np.array(centroid) + 10),
            (vertices @ rotation.T, rotation @ centroid),
        ):
            v, c = cell_geometry(moved[None], cell_type)
            assert v[0] == pytest.approx(volume) and np.allclose(c[0], centre)

    def test_non_convex_and_inverted_cells(self):
        # A pyramid on a dart-shaped (non-convex) base of area 1: V = area * height / 3.
        dart = np.array([[0, 0, 0], [2, 0, 0], [0.5, 0.5, 0], [0, 2, 0], [0.3, 0.3, 1.0]])
        volume, _ = cell_geometry(dart[None], 14)
        assert volume[0] == pytest.approx(1 / 3)
        inverted = np.array(UNIT_CELLS[10][0], dtype=float)[[0, 2, 1, 3]]
        assert cell_geometry(inverted[None], 10)[0][0] < 0

    def test_layouts_mixed_cells_and_point_data(self, tmp_path):
        tet, _, _ = UNIT_CELLS[10]
        hexa, _, _ = UNIT_CELLS[12]
        points = np.vstack([np.array(hexa) + [5, 0, 0], tet])
        cells = [list(range(8)), [8, 9, 10, 11]]
        values = np.arange(12, dtype=float)
        structures = []
        for layout in ('legacy', '5.1'):
            path = tmp_path / f'{layout}.vtk'
            path.write_text(vtk_text(points, cells, [12, 10], values, 'POINT_DATA', layout))
            structures.append(read_vtk(path))
        assert np.allclose(structures[0].xyz, structures[1].xyz)
        assert np.allclose(structures[0].sld, structures[1].sld)
        assert np.allclose(structures[0].volume, [1.0, 1 / 6])
        shifted = tmp_path / 'shifted.vtk'
        shifted.write_text(vtk_text(points + 100, cells, [12, 10], values, 'POINT_DATA'))
        assert np.allclose(read_vtk(shifted).sld, structures[0].sld)  # translation invariant

    def test_refusals(self, tmp_path):
        tet = UNIT_CELLS[10][0]
        two = vtk_text(
            tet,
            [[0, 1, 2, 3]],
            [10],
            [1.0],
            extra='SCALARS other float 1\nLOOKUP_TABLE default\n2\n',
        )
        (tmp_path / 'two.vtk').write_text(two)
        with pytest.raises(StructureError, match='several candidate SLD fields'):
            read_vtk(tmp_path / 'two.vtk')
        flat = vtk_text([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], [[0, 1, 2, 3]], [10], [1.0])
        (tmp_path / 'flat.vtk').write_text(flat)
        with pytest.raises(StructureError, match='degenerate'):
            read_vtk(tmp_path / 'flat.vtk')
        binary = vtk_text(tet, [[0, 1, 2, 3]], [10], [1.0]).replace('ASCII', 'BINARY')
        (tmp_path / 'binary.vtk').write_text(binary)
        with pytest.raises(StructureError, match='binary=False'):
            read_vtk(tmp_path / 'binary.vtk')

    def test_cell_size_check(self):
        gsc = quiet_calculator(CUBE_VTK)  # cell size 2*sqrt(2)/sqrt(3) = 1.63 Ang
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            gsc.calculate()  # q_max 0.3: q * size = 0.49
        gsc.set_q_grid(qmax=1.0)
        with pytest.warns(UserWarning, match='treating each cell as a point'):
            gsc.calculate()

    def test_mesh_refinement_trend(self, tmp_path):
        """A voxelised sphere split into tetrahedra approaches the sphere at low q."""
        errors = []
        q = np.array([0.02, 0.05])
        for step in (4.0, 2.0):
            radius = 10.0
            n = int(radius / step) + 1
            axis = np.arange(-n, n) * step
            corners = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
            corners = corners[((corners + step / 2) ** 2).sum(1) <= radius**2]
            hexa = np.array(UNIT_CELLS[12][0], dtype=float) * step
            six = (
                (0, 1, 2, 6),
                (0, 2, 3, 6),
                (0, 3, 7, 6),
                (0, 7, 4, 6),
                (0, 4, 5, 6),
                (0, 5, 1, 6),
            )
            points, cells = [], []
            for corner in corners:
                base = len(points)
                points.extend(hexa + corner)
                cells.extend([base + i for i in tet] for tet in six)
            path = tmp_path / f'sphere{step}.vtk'
            path.write_text(vtk_text(points, cells, [10] * len(cells), [1e-6] * len(cells)))
            gsc = quiet_calculator(path)
            gsc.set_q(q)
            intensity = gsc.calculate().intensity
            volume = gsc.structure.volume.sum()
            x = q * (3 * volume / (4 * np.pi)) ** (1 / 3)
            sphere = 1e8 * (1e-6) ** 2 * volume * (3 * (np.sin(x) - x * np.cos(x)) / x**3) ** 2
            errors.append(np.max(np.abs(intensity / sphere - 1)))
        assert errors[1] < errors[0]


# ============================================================================
# Calculator state and integration
# ============================================================================


class TestCalculator:
    def test_setters_clear_the_profile_and_failures_are_atomic(self):
        gsc = quiet_calculator(C60)
        first = gsc.calculate()
        gsc.set_params(normalisation_volume=100.0)
        assert gsc.profile is None
        with pytest.raises(ValueError):
            gsc.set_params(scale=2.0, background=-1.0)
        assert gsc.get_params()['scale'] == 1.0
        with pytest.raises(StructureError):
            gsc.load_structure('missing.pdb')
        assert gsc.structure.n_points == 60
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gsc.load_structure(C60)
        assert gsc.get_params()['normalisation_volume'] == 'auto'
        with pytest.raises(AttributeError):
            gsc.structure = None
        assert first.params['normalisation_volume'] == 'auto'  # the old profile is unchanged

    def test_no_structure_and_no_profile(self):
        gsc = GenericScatteringCalculator()
        with pytest.raises(ValueError, match='load_structure'):
            gsc.calculate()
        with pytest.raises(ValueError, match='calculate'):
            gsc.plot_profile(show=False)

    def test_profiles_and_structures_are_snapshots(self):
        xyz = np.array([[0.0, 0, 0], [0, 0, 5]])
        values = np.array([1e-6, 2e-6])
        structure = Structure.from_arrays(xyz, values, np.ones(2))
        values[0] = 9.0  # the caller edits their array afterwards
        assert structure.sld[0] == 1e-6
        profile = quiet_calculator(structure, qmax=0.2).calculate()
        params = profile.params
        params['scale'] = 99
        assert profile.params['scale'] == 1.0
        for clone in (pickle.loads(pickle.dumps(profile)), copy.deepcopy(profile)):
            assert np.array_equal(clone.intensity, profile.intensity)
        assert np.array_equal(pickle.loads(pickle.dumps(structure)).xyz, structure.xyz)

    def test_progress_stop_leaves_the_profile_unchanged(self):
        structure = Structure.from_arrays(
            np.random.default_rng(0).normal(size=(200, 3)), np.full(200, 1e-6), np.ones(200)
        )
        gsc = quiet_calculator(structure)
        with pytest.raises(CalculationInterrupted):
            gsc.calculate(progress=lambda fraction: True)
        assert gsc.profile is None
        with pytest.raises(KeyboardInterrupt):
            gsc.calculate(progress=lambda fraction: (_ for _ in ()).throw(KeyboardInterrupt))

    def test_to_data1d_fits_with_sasmodels(self):
        gsc = quiet_calculator(sphere_pixels())
        gsc.set_params(sld_solvent=6.34)
        data = gsc.calculate().to_data1d()
        assert data.dy is None and data.dx is None
        assert data.process[-1].name == 'sans_fitter.structure'
        fitter = SANSFitter()
        fitter.set_data(data)
        fitter.set_weighting('abs')
        fitter.set_resolution('none')
        fitter.set_model('sphere')
        fitter.set_param('sld', value=1.0, vary=False)
        fitter.set_param('sld_solvent', value=6.34, vary=False)
        fitter.set_param('radius', value=25, min=10, max=60, vary=True)
        fitter.set_param('scale', value=1.0, min=0.5, max=2, vary=True)
        fitter.set_param('background', value=0.0, vary=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = fitter.fit(method='lm')
        assert not [w for w in caught if 'resolution' in str(w.message).lower()]
        assert result['parameters']['radius']['value'] == pytest.approx(30.0, rel=0.005)
        with pytest.raises(ValueError, match='dI'):
            gsc.profile.to_data1d(dI=[1.0])

    def test_describe_settings_and_replay(self, tmp_path, capsys):
        path = tmp_path / 'c60.pdb'
        path.write_bytes(open(C60, 'rb').read())
        gsc = quiet_calculator(path)
        gsc.set_q_grid(1e-3, 0.5, 20, spacing='linear')
        gsc.set_params(scale=2.0, normalisation_volume=600.0)
        gsc.describe()
        profile = gsc.calculate()
        profile.describe()
        assert 'V_norm' in capsys.readouterr().out and 'V_norm' in profile._repr_html_()
        assert 'Rg (mass)' in gsc.structure._repr_html_()
        settings = json.loads(json.dumps(gsc.get_settings()))
        assert settings['params']['normalisation_volume'] == 600.0
        assert settings['versions']['periodictable']
        replay = GenericScatteringCalculator.from_settings(settings)
        assert np.allclose(replay.calculate().intensity, profile.intensity)
        with open(path, 'a') as handle:
            handle.write('REMARK changed\n')
        with pytest.warns(UserWarning, match='has changed'):
            GenericScatteringCalculator.from_settings(settings)
        profile.save_csv(tmp_path / 'profile.csv')
        assert np.loadtxt(tmp_path / 'profile.csv', delimiter=',').shape == (20, 2)


# ============================================================================
# Plots
# ============================================================================


class TestPlots:
    def test_points_by_element_with_bonds(self):
        fig = quiet_calculator(C60).plot_points(show=False)
        assert [t.name for t in fig.data] == ['C', 'bonds']

    def test_thinning_reports_counts_and_keeps_the_lattice(self):
        structure = sphere_pixels(radius=12.0, step=1.0)
        fig = structure.plot_points(max_points=2000, show=False)
        shown = sum(len(t.x) for t in fig.data)
        assert shown <= 2000 and f'showing {shown} of {structure.n_points}' in fig.layout.title.text
        atoms = Structure.from_arrays(
            np.random.default_rng(0).normal(size=(3000, 3)) * 10,
            np.full(3000, 1e-6),
            np.ones(3000),
            symbols=['C'] * 3000,
        )
        assert sum(len(t.x) for t in atoms.plot_points(max_points=500, show=False).data) <= 500

    def test_zero_contrast_points_follow_the_solvent(self):
        gsc = quiet_calculator(sphere_pixels(radius=6.0))
        gsc.set_params(sld_solvent=6.34)
        names = [t.name for t in gsc.plot_points(show=False).data]
        assert names == ['SLD', 'zero contrast']

    def test_profile_plot(self):
        gsc = quiet_calculator(C60)
        profile = gsc.calculate()
        fig = gsc.plot_profile(data=profile.to_data1d(), show=False)
        assert len(fig.data) == 2 and fig.layout.xaxis.type == 'log'
        zero = Structure.from_arrays([[0.0, 0, 0]], [1e-6], [1.0])
        flat = quiet_calculator(zero)
        flat.set_params(sld_solvent=1.0)
        with pytest.warns(UserWarning):
            flat.calculate()
        assert flat.plot_profile(show=False).layout.yaxis.type == 'linear'
