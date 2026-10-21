"""Readers for PDB, SLD and legacy VTK structure files.

All readers return a :class:`Structure` with positions in Å and SLD in Å⁻²,
raise :class:`StructureError` on unusable input, and record the file's
SHA-256. Behaviour follows SasView's Generic Scattering Calculator readers
(``sas.sascalc.calculator.sas_gen``, BSD-3-Clause, Copyright (c) 2009-2026,
SasView Developers), with the deviations documented in ``docs/usage.md``: the
VTK cell geometry is recomputed, because SasView's volumes for hexahedra,
wedges and pyramids are wrong.
"""

import os
from collections import Counter

import numpy as np

from ..data.provenance import _hash_file
from .model import AVOGADRO, Structure, StructureError, detect_lattice

WATER_RESIDUES = ('HOH', 'DOD', 'WAT')


def load(path, *, include_hetatm: bool = False) -> Structure:
    """Read a structure file, choosing the reader by extension.

    Args:
        path: A ``.pdb``, ``.sld`` (or ``.txt``) or ``.vtk`` file.
        include_hetatm: PDB only; also read ``HETATM`` records (never waters).
    """
    extension = os.path.splitext(os.fspath(path))[1].lower()
    if extension == '.pdb':
        return read_pdb(path, include_hetatm=include_hetatm)
    if extension in ('.sld', '.txt'):
        return read_sld(path)
    if extension == '.vtk':
        return read_vtk(path)
    raise StructureError(
        f'Unknown structure file type {extension!r} for {path}. Use .pdb, .sld or .vtk.'
    )


def _source(path) -> tuple[str, str | None]:
    path = os.path.abspath(os.fspath(path))
    if not os.path.isfile(path):
        raise StructureError(f'Structure file not found: {path}')
    return path, _hash_file(path)


# ============================================================================
# PDB
# ============================================================================


def _element_from_name(line: str) -> str:
    """SasView's guess of the element from the atom-name columns 13-16."""
    name = line[12:16].strip()
    if line[12].isdigit():
        return name[1]
    if len(name) == 4 or line[12] == ' ':
        return name[0]
    return name[:2]


def _element_constants(symbol: str) -> tuple[float, float]:
    """SLD (Å⁻²) and natural atomic volume (Å³) of an element, as SasView computes them."""
    from periodictable import formula, nsf

    atom = formula(symbol)
    sld = nsf.neutron_sld(symbol)[0] * 1e-6
    volume = 1e24 * atom.mass / atom.density / AVOGADRO
    if not (np.isfinite(sld) and np.isfinite(volume) and volume > 0):
        raise ValueError(symbol)
    return sld, volume


def read_pdb(path, *, include_hetatm: bool = False) -> Structure:
    """Read the atoms of a PDB file.

    Policies: the first ``MODEL`` only; one alternate-location ID for the whole
    structure (blank plus the first non-blank ID seen); occupancy not applied;
    waters never read; elements from columns 77-78, else from the atom name.
    Each policy that drops or ignores something adds a note.
    """
    path, sha = _source(path)
    with open(path, encoding='utf-8', errors='replace') as handle:
        lines = handle.read().splitlines()

    xyz, symbols, serials, conect = [], [], [], []
    dropped_het: Counter = Counter()
    dropped_water = dropped_altloc = models = 0
    partial_occupancy = False
    altloc_kept = None
    for number, line in enumerate(lines, start=1):
        record = line[:6]
        if record == 'MODEL ':
            models += 1
        if models > 1:
            continue
        if record == 'CONECT':
            fields = [line[i : i + 5].strip() for i in range(6, 31, 5)]
            conect.append([int(f) for f in fields if f.isdigit()])
            continue
        if record not in ('ATOM  ', 'HETATM'):
            continue
        residue = line[17:20].strip()
        if residue in WATER_RESIDUES:
            dropped_water += 1
            continue
        if record == 'HETATM' and not include_hetatm:
            dropped_het[residue] += 1
            continue
        altloc = line[16:17].strip()
        if altloc:
            altloc_kept = altloc_kept or altloc
            if altloc != altloc_kept:
                dropped_altloc += 1
                continue
        try:
            position = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            symbol = line[76:78].strip() or _element_from_name(line)
            occupancy = line[54:60].strip()
            partial_occupancy |= bool(occupancy) and float(occupancy) < 1.0
        except (ValueError, IndexError) as exc:
            raise StructureError(f'{path}, line {number}: cannot read the atom record.') from exc
        xyz.append(position)
        symbols.append(symbol[0].upper() + symbol[1:].lower())
        serial = line[6:11].strip()
        serials.append(int(serial) if serial.isdigit() else None)

    if not xyz:
        raise StructureError(f'{path} contains no atom records.')

    constants, unknown = {}, []
    for symbol in sorted(set(symbols)):
        try:
            constants[symbol] = _element_constants(symbol)
        except (ValueError, TypeError, AttributeError):
            unknown.append(symbol)
    if unknown:
        raise StructureError(
            f'{path}: no neutron SLD or density for element(s) {", ".join(unknown)}.'
        )
    sld = np.array([constants[s][0] for s in symbols])
    volume = np.array([constants[s][1] for s in symbols])

    index_of = {s: i for i, s in enumerate(serials) if s is not None}
    bonds = [
        (index_of[group[0]], index_of[other])
        for group in conect
        if group and group[0] in index_of
        for other in group[1:]
        if other in index_of
    ]

    notes = []
    if models > 1:
        notes.append(f'Read the first of {models} models.')
    if dropped_het:
        listed = ', '.join(f'{name} x{count}' for name, count in sorted(dropped_het.items()))
        notes.append(
            f'Skipped {sum(dropped_het.values())} HETATM records ({listed}); '
            'pass include_hetatm=True to read them.'
        )
    if dropped_water:
        notes.append(f'Skipped {dropped_water} water atoms.')
    if dropped_altloc:
        notes.append(f'Kept alternate location {altloc_kept!r}; skipped {dropped_altloc} atoms.')
    if partial_occupancy:
        notes.append('Some atoms have occupancy below 1; occupancy is not applied.')

    return Structure(
        'atoms',
        np.array(xyz),
        sld,
        volume,
        symbols=symbols,
        bonds=np.array(bonds, dtype=int).reshape(-1, 2),
        source=path,
        source_sha256=sha,
        format='pdb',
        notes=tuple(notes),
    )


# ============================================================================
# SLD
# ============================================================================


def read_sld(path) -> Structure:
    """Read a SasView SLD text file.

    The first line is a header and is always skipped. Columns: ``x y z sld``
    (4), ``x y z sld mx my mz`` (7) or ``x y z sld mx my mz volume`` (8);
    six columns are magnetic-only and refused. Without a volume column the
    points must form a lattice, and each pixel's volume is the product of the
    steps. A sparse subset that skips planes looks like a coarser lattice, so
    such files need the volume column.
    """
    path, sha = _source(path)
    try:
        data = np.loadtxt(path, skiprows=1, ndmin=2)
    except ValueError as exc:
        raise StructureError(f'{path} is not a numeric SLD file: {exc}') from exc
    columns = data.shape[1]
    if columns == 6:
        raise StructureError(
            f'{path} holds magnetic SLD only (6 columns); a nuclear SLD is needed.'
        )
    if columns not in (4, 7, 8) or data.shape[0] == 0:
        raise StructureError(f'{path} must have 4, 7 or 8 columns after one header line.')
    if not np.all(np.isfinite(data)):
        raise StructureError(f'{path} contains non-finite values.')
    xyz = data[:, :3]
    if np.unique(xyz, axis=0).shape[0] != xyz.shape[0]:
        raise StructureError(f'{path} lists the same position more than once.')

    notes = []
    if columns >= 7 and np.any(data[:, 4:7] != 0):
        notes.append('Magnetic SLD columns are ignored.')
    lattice = detect_lattice(xyz)
    if columns == 8:
        volume = data[:, 7]
    elif lattice is not None and all(step > 0 for step in lattice.step):
        volume = np.full(xyz.shape[0], float(np.prod(lattice.step)))
    else:
        raise StructureError(
            f'{path}: without a volume column the points must form a 3D lattice with at '
            'least two positions on each axis. Add the pixel volume as column 8.'
        )
    if np.any(volume <= 0):
        raise StructureError(f'{path}: every pixel volume must be positive.')
    return Structure(
        'pixels',
        xyz,
        data[:, 3],
        volume,
        lattice=lattice,
        source=path,
        source_sha256=sha,
        format='sld',
        notes=tuple(notes),
    )


# ============================================================================
# VTK (legacy ASCII, UNSTRUCTURED_GRID)
# ============================================================================

# Outward-oriented faces per VTK cell type, in VTK's vertex order. Voxels
# (type 11) are reordered to the hexahedron order before use.
_FACES = {
    10: ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
    12: ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)),
    13: ((0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)),
    14: ((0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
}
_VERTICES = {10: 4, 11: 8, 12: 8, 13: 6, 14: 5}
_VOXEL_TO_HEXAHEDRON = (0, 1, 3, 2, 4, 5, 7, 6)
_SECTIONS = {
    'POINTS',
    'CELLS',
    'CELL_TYPES',
    'CELL_DATA',
    'POINT_DATA',
    'SCALARS',
    'VECTORS',
    'METADATA',
}


def cell_geometry(vertices: np.ndarray, cell_type: int) -> tuple[np.ndarray, np.ndarray]:
    """Signed volumes and volume centroids for cells of one VTK type.

    Each face is split into triangles (quadrilaterals are fanned from their
    face centre) and each triangle is joined to the cell's vertex mean. The
    signed tetrahedron volumes sum to the cell volume by the divergence
    theorem, convex or not; their volume-weighted centroids give the cell's
    centroid.

    Args:
        vertices: Shape (n_cells, n_vertices, 3), in VTK vertex order.
        cell_type: 10, 11, 12, 13 or 14.
    """
    if cell_type == 11:
        vertices, cell_type = vertices[:, _VOXEL_TO_HEXAHEDRON], 12
    reference = vertices.mean(axis=1)
    volume = np.zeros(vertices.shape[0])
    moment = np.zeros((vertices.shape[0], 3))
    for face in _FACES[cell_type]:
        corners = vertices[:, face]
        if len(face) == 3:
            triangles = [(corners[:, 0], corners[:, 1], corners[:, 2])]
        else:
            centre = corners.mean(axis=1)
            triangles = [(corners[:, i], corners[:, (i + 1) % 4], centre) for i in range(4)]
        for a, b, c in triangles:
            tet = np.einsum('ij,ij->i', a - reference, np.cross(b - reference, c - reference)) / 6
            volume += tet
            moment += tet[:, None] * (a + b + c + reference) / 4
    with np.errstate(invalid='ignore', divide='ignore'):
        centroid = moment / volume[:, None]
    return volume, centroid


class _Tokens:
    """A cursor over the whitespace-separated tokens of a VTK body."""

    def __init__(self, text: str, path: str) -> None:
        self.items = text.split()
        self.pos = 0
        self.path = path

    def more(self) -> bool:
        return self.pos < len(self.items)

    def peek(self) -> str:
        return self.items[self.pos] if self.more() else ''

    def next(self) -> str:
        if not self.more():
            raise StructureError(f'{self.path}: unexpected end of file.')
        self.pos += 1
        return self.items[self.pos - 1]

    def numbers(self, count: int, dtype=float) -> np.ndarray:
        if self.pos + count > len(self.items):
            raise StructureError(f'{self.path}: expected {count} more values.')
        try:
            values = np.array(self.items[self.pos : self.pos + count], dtype=float)
        except ValueError as exc:
            raise StructureError(f'{self.path}: non-numeric value in a data block.') from exc
        self.pos += count
        return values.astype(dtype)


def _read_cells(tokens: _Tokens, n_cells_field: int, size: int) -> list[np.ndarray]:
    """Cell connectivity in either the legacy (<= 4.2) or the 5.1 layout."""
    if tokens.peek().upper() == 'OFFSETS':  # 5.1: CELLS n_offsets n_connectivity
        tokens.next()
        tokens.next()  # data type
        offsets = tokens.numbers(n_cells_field, int)
        if tokens.next().upper() != 'CONNECTIVITY':
            raise StructureError(f'{tokens.path}: CONNECTIVITY expected after OFFSETS.')
        tokens.next()
        connectivity = tokens.numbers(size, int)
        return [connectivity[offsets[i] : offsets[i + 1]] for i in range(n_cells_field - 1)]
    flat = tokens.numbers(size, int)
    cells, i = [], 0
    while i < flat.size:
        count = flat[i]
        cells.append(flat[i + 1 : i + 1 + count])
        i += count + 1
    if len(cells) != n_cells_field:
        raise StructureError(
            f'{tokens.path}: CELLS declares {n_cells_field} cells, found {len(cells)}.'
        )
    return cells


def read_vtk(path) -> Structure:
    """Read a legacy ASCII VTK ``UNSTRUCTURED_GRID`` as one point per cell.

    Supports cell types 10 (tetrahedron), 11 (voxel), 12 (hexahedron), 13
    (wedge) and 14 (pyramid), in the classic and the 5.1 ``OFFSETS`` layout.
    The nuclear SLD is the single one-component ``SCALARS`` field, given per
    cell or per point (point values are averaged to cells with SasView's
    inverse-square weights about the cell's vertex mean). Three-component
    fields are magnetic and ignored. Each cell becomes a point at its volume
    centroid with the cell's volume and SLD. Units are taken as Å and Å⁻².
    """
    path, sha = _source(path)
    with open(path, encoding='utf-8', errors='replace') as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if len(lines) < 4 or not lines[0].lower().startswith('# vtk datafile version'):
        raise StructureError(f'{path} is not a legacy VTK file.')
    if lines[2].upper() != 'ASCII':
        raise StructureError(
            f'{path} is binary. Re-export it as ASCII, for example '
            "meshio.write(path, mesh, binary=False), or ParaView's Save Data with ASCII."
        )
    if lines[3].upper().split() != ['DATASET', 'UNSTRUCTURED_GRID']:
        raise StructureError(
            f'{path}: only DATASET UNSTRUCTURED_GRID is supported, got {lines[3]!r}.'
        )

    tokens = _Tokens('\n'.join(lines[4:]), path)
    points = cells = types = None
    association, fields = None, []
    while tokens.more():
        word = tokens.next().upper()
        if word == 'POINTS':
            n = int(tokens.next())
            tokens.next()
            points = tokens.numbers(3 * n).reshape(n, 3)
        elif word == 'CELLS':
            first, size = int(tokens.next()), int(tokens.next())
            cells = _read_cells(tokens, first, size)
        elif word == 'CELL_TYPES':
            types = tokens.numbers(int(tokens.next()), int)
        elif word in ('CELL_DATA', 'POINT_DATA'):
            association = (word, int(tokens.next()))
        elif word in ('SCALARS', 'VECTORS'):
            if association is None:
                raise StructureError(f'{path}: {word} outside CELL_DATA or POINT_DATA.')
            name = tokens.next()
            tokens.next()  # data type
            components = 3
            if word == 'SCALARS':
                components = int(tokens.next()) if tokens.peek().isdigit() else 1
                if tokens.peek().upper() == 'LOOKUP_TABLE':
                    tokens.next()
                    tokens.next()
            values = tokens.numbers(association[1] * components).reshape(association[1], components)
            fields.append((name, association[0], values))
        elif word == 'METADATA':
            while tokens.more() and tokens.peek().upper() not in _SECTIONS:
                tokens.next()
        else:
            raise StructureError(f'{path}: unsupported VTK section {word!r}.')

    if points is None or cells is None or types is None:
        raise StructureError(f'{path} needs POINTS, CELLS and CELL_TYPES sections.')
    if len(types) != len(cells):
        raise StructureError(f'{path}: {len(cells)} cells but {len(types)} cell types.')

    nuclear = [f for f in fields if f[2].shape[1] == 1]
    magnetic = [f for f in fields if f[2].shape[1] == 3]
    if not nuclear:
        what = 'only magnetic data' if magnetic else 'no SLD data'
        raise StructureError(f'{path} has {what}; one single-component SCALARS field is needed.')
    if len(nuclear) > 1:
        names = ', '.join(f[0] for f in nuclear)
        raise StructureError(f'{path} has several candidate SLD fields ({names}); keep only one.')
    name, where, values = nuclear[0]
    values = values[:, 0]

    n_cells = len(cells)
    centroid = np.empty((n_cells, 3))
    volume = np.empty(n_cells)
    diameter = np.empty(n_cells)
    sld = values if where == 'CELL_DATA' else np.empty(n_cells)
    if where == 'CELL_DATA' and values.size != n_cells:
        raise StructureError(f'{path}: CELL_DATA has {values.size} values for {n_cells} cells.')
    if where == 'POINT_DATA' and values.size != len(points):
        raise StructureError(
            f'{path}: POINT_DATA has {values.size} values for {len(points)} points.'
        )
    for cell_type in np.unique(types):
        if cell_type not in _VERTICES:
            raise StructureError(f'{path}: unsupported cell type {cell_type}; use types 10-14.')
        members = np.flatnonzero(types == cell_type)
        connectivity = [cells[i] for i in members]
        if any(c.size != _VERTICES[cell_type] for c in connectivity):
            raise StructureError(
                f'{path}: a type-{cell_type} cell has the wrong number of vertices.'
            )
        index = np.array(connectivity)
        if index.min() < 0 or index.max() >= len(points):
            raise StructureError(f'{path}: a cell refers to a point that does not exist.')
        vertices = points[index]
        volume[members], centroid[members] = cell_geometry(vertices, int(cell_type))
        spans = vertices[:, :, None, :] - vertices[:, None, :, :]
        diameter[members] = np.sqrt((spans**2).sum(axis=-1)).max(axis=(1, 2))
        if where == 'POINT_DATA':
            distance2 = ((vertices - vertices.mean(axis=1, keepdims=True)) ** 2).sum(axis=-1)
            weight = 1.0 / distance2
            sld[members] = (weight * values[index]).sum(axis=1) / weight.sum(axis=1)

    floor = 1e-12 * np.median(np.abs(volume))
    bad = np.flatnonzero(volume <= floor)
    if bad.size:
        kind = 'inverted' if volume[bad[0]] < 0 else 'degenerate'
        raise StructureError(f'{path}: cell {bad[0]} is {kind} (volume {volume[bad[0]]:.3g}).')

    notes = []
    if magnetic:
        notes.append('Magnetic SLD fields are ignored.')
    if where == 'POINT_DATA':
        notes.append(f'SLD field {name!r} was averaged from points to cells.')
    return Structure(
        'cells',
        centroid,
        sld,
        volume,
        cell_diameter=diameter,
        source=path,
        source_sha256=sha,
        format='vtk',
        notes=tuple(notes),
    )
