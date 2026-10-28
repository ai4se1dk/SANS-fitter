# Debye-bye: initial design

Status: draft, 2026-10-28
Source: `sans_fitter.structure` on the `gsc` branch (commit `2e9674e`, "Initial version of the GSC calculator")
Target: a standalone project **Debye-bye** (repository `ai4se1dk/Debye-bye`, distribution `debye-bye`, import `debyebye`) that IQuana, SANS-webapp, notebooks and other tools import.

## 1. Goal

Turn the Generic Scattering Calculator (GSC) from a subpackage of the fitter into a library that:

- installs and imports without IQuana, sasmodels, sasdata, bumps or plotly;
- has a small, versioned public API that a consumer can pin;
- treats logging, warnings and plotting the way a library should: quiet by default, filterable, optional;
- hands a computed curve to IQuana (or anything that speaks sasdata) through a narrow, documented contract;
- leaves room to fit structures later, not only compute them.

Physics, readers and numerics stay as they are on `gsc`. This is an extraction and an API cleanup, not a rewrite.

## 2. What exists on `gsc`

Only commit `2e9674e` is GSC work. The other four commits on `gsc` (multifit, weighting, the four-form-factor notebook) are already in IQuana, so they are out of scope here.

| File | Lines | Role |
|---|---|---|
| `structure/model.py` | 295 | `Structure`, `Lattice`, `RgResult`, `StructureError`, lattice detection |
| `structure/debye.py` | 136 | `debye_pairs` (N² pair sum), `debye_lattice` (FFT autocorrelation), `CalculationInterrupted` |
| `structure/readers.py` | 491 | PDB, SLD and legacy-VTK readers, `load()` by extension, VTK cell geometry |
| `structure/calculator.py` | 557 | `GenericScatteringCalculator` (stateful), `ScatteringProfile` (result) |
| `structure/plotting.py` | 195 | plotly `plot_points` and `plot_profile` |
| `tests/test_structure.py` | 650 | kernels, physics vs sasmodels, normalisation, readers, calculator, plots |
| `examples/data/` | — | `c60.pdb`, `five_tetrahedra_cube.vtk` |

The physics core (`model`, `debye`, `readers`) is already almost independent. The coupling to the fitter is a short list:

| Import on `gsc` | Used for | In Debye-bye |
|---|---|---|
| `..console`: `logger`, `OK`, `ANGSTROM`, `CUBIC_ANGSTROM`, `INVERSE_ANGSTROM`, `SLD_UNIT` | status lines, unit glyphs | own `debyebye._console` (logger `'debyebye'`, the same glyph fallback) |
| `..data.loader.validate_q_grid` | q validation | copy (15 lines) into `debyebye._validate` |
| `..data.loader.normalize_sans_data` | qmin/qmax/mask on the `Data1D` | drop: `IQuana.set_data()` already normalises |
| `..data.provenance._hash_file` | SHA-256 of structure files | copy (10 lines) |
| `..persistence._package_version` | versions in `get_settings()` | copy (6 lines) |
| `..plotting.error_bars`, `resolve_show` | plotly helpers | copy into `debyebye.plotting` |
| `sasdata.dataloader.data_info.Data1D, Process` | `ScatteringProfile.to_data1d()` | optional extra, lazy import |
| `plotly` | both plots | optional extra, lazy import |
| `periodictable` | element SLDs, volumes and masses | required, isolated in one module |
| `sasmodels` (tests only) | analytic-sphere check, fit round trip | test dependency only |

Nothing in the fitter imports `structure` except the re-export in `sans_fitter/__init__.py` and `tests/test_public_api.py`.

## 3. Decisions

**D1. Core dependencies are `numpy` and `periodictable`, nothing else.** periodictable is pure Python and small, and reading PDB files is a headline feature, so it stays required. Its use goes behind one module (`debyebye.elements`) so that a different source of element constants could be swapped in later.

**D2. Plotting and sasdata are optional extras.** `debye-bye[plot]` adds plotly; `debye-bye[sasdata]` adds sasdata. The methods that need them (`plot_points`, `plot_profile`, `to_data1d`) stay where they are, import lazily, and raise `ImportError` naming the extra to install. Importing `debyebye` must never import plotly or sasdata.

**D3. A pure function at the core, the stateful calculator on top.** The `gsc` calculator mixes state (structure, grid, parameters, last profile) with the calculation. Debye-bye splits them:

```python
profile = debyebye.compute(structure, q, sld_solvent=6.34, scale=1.0,
                           background=0.0, normalisation_volume='auto',
                           progress=None)
```

`compute()` holds the body of today's `calculate()`. `GenericScatteringCalculator` keeps its current API for notebooks and calls `compute()`. Consumers that manage their own state (a web app session, a fitter, a batch script) call `compute()` and never touch the calculator class.

**D4. Quiet by default.** The library logger is `logging.getLogger('debyebye')` with a `NullHandler`, as libraries are expected to do; it does not print. `debyebye.set_verbosity('info')` installs the stdout handler that `sans_fitter.console` uses, so a standalone notebook gets today's status lines with one call. IQuana attaches its own handler to the `'debyebye'` logger at import and forwards its `set_verbosity()`, so an IQuana user sees exactly what they see on `gsc`.

**D5. Warnings get categories.** Every `warnings.warn` uses a subclass of `debyebye.DebyeByeWarning`:

| Category | Raised for |
|---|---|
| `ReaderWarning` | a reader dropped or approximated something (the structure's `notes`) |
| `NormalisationWarning` | near-matched populations, everything matched |
| `ApproximationWarning` | q × cell size above 1, elemental atomic volumes in a solvent |
| `PerformanceWarning` | a pair sum over more than `LARGE_PAIR_LOOP` points |
| `UnitsWarning` | `sld_solvent` that looks like Å⁻² |
| `ReproducibilityWarning` | `from_settings()`: changed file or periodictable version |

A consumer can then silence one kind (`warnings.simplefilter('ignore', debyebye.PerformanceWarning)`) without silencing everything. The text and the copy in `ScatteringProfile.notes` stay as on `gsc`.

**D6. The interop contract is the profile's fields, not sasdata.** A consumer needs `q` (1/Å), `intensity` (1/cm), and a provenance record. `ScatteringProfile` exposes those as plain data (`q`, `intensity`, `provenance()` returning a dict, `provenance_text()` returning the ASCII line now built by `_process_text`). `to_data1d()` is a convenience built on them. IQuana relies on the fields, so it would keep working if a consumer had no sasdata at all.

**D7. Units stay as on `gsc`.** Positions in Å, structure SLDs in Å⁻², `sld_solvent` in 10⁻⁶ Å⁻² (the sasmodels convention), q in 1/Å, I in 1/cm. The unit of every argument is in its docstring and in `docs/units.md`; these are part of the API and cannot change in a minor release.

**D8. Module names lose the redundancy.** `debyebye.debye` becomes `debyebye.kernels`. Readers become a subpackage, one module per format, with a registry `load()` dispatches on.

## 4. Package layout

```text
Debye-bye/
├── pyproject.toml              # name = "debye-bye", src layout, BSD-3-Clause
├── LICENSE                     # BSD-3-Clause, plus the SasView notice the readers carry
├── CHANGELOG.md
├── src/debyebye/
│   ├── __init__.py             # public API, __version__, set_verbosity
│   ├── _console.py             # logger 'debyebye', NullHandler, glyphs, set_verbosity
│   ├── _util.py                # validate_q_grid, hash_file, package_version
│   ├── warnings.py             # DebyeByeWarning and subclasses
│   ├── elements.py             # periodictable access: element SLD, volume, mass
│   ├── model.py                # Structure, Lattice, RgResult, StructureError, detect_lattice
│   ├── kernels.py              # debye_pairs, debye_lattice, lattice_cells, CalculationInterrupted
│   ├── compute.py              # compute(), ScatteringProfile
│   ├── calculator.py           # GenericScatteringCalculator, settings round trip
│   ├── readers/
│   │   ├── __init__.py         # load(), register_reader(), READERS
│   │   ├── pdb.py
│   │   ├── sld.py
│   │   └── vtk.py              # including cell_geometry
│   ├── plotting.py             # lazy plotly; plot_points, plot_profile
│   ├── interop/
│   │   └── sasdata.py          # to_data1d(profile, dI=None), lazy sasdata
│   └── examples/               # package data: c60.pdb, five_tetrahedra_cube.vtk
│       └── __init__.py         # path(name) via importlib.resources
├── tests/
├── docs/                       # mkdocs, moved from the GSC section of docs/usage.md
├── examples/structure_calculator_example.py
└── notebooks/structure_calculator.ipynb
```

The example structures ship as package data so that IQuana's docs and notebooks can use them without copying files: `debyebye.examples.path('c60.pdb')`.

## 5. Public API, v0.1

Everything in `debyebye.__all__` is public and covered by the version policy (§10). Anything else, including `debyebye.readers.vtk.cell_geometry`, is internal even if tests import it.

```python
import debyebye as dbb

# Structures
s = dbb.load('particle.sld')                 # .pdb, .sld/.txt, .vtk
s = dbb.read_pdb('protein.pdb', include_hetatm=False)
s = dbb.read_sld('particle.sld')
s = dbb.read_vtk('mesh.vtk')
s = dbb.Structure.from_arrays(xyz, sld, volume, symbols=None, bonds=None)
s.describe(); s.summary(); s.radius_of_gyration('sld', sld_solvent=6.34)

# Calculation, stateless
q = dbb.q_grid(3e-4, 0.3, 30, spacing='log')  # the private _q_grid, made public
p = dbb.compute(s, q, sld_solvent=6.34)       # -> ScatteringProfile

# Calculation, stateful (unchanged from gsc)
gsc = dbb.GenericScatteringCalculator('particle.sld')
gsc.set_q_grid(3e-4, 0.3, 30)
gsc.set_params(sld_solvent=6.34)
p = gsc.calculate(progress=callback)
settings = gsc.get_settings()
gsc2 = dbb.GenericScatteringCalculator.from_settings(settings)

# Result
p.q, p.intensity, p.method, p.normalisation_volume, p.notes
p.provenance()       # dict: structure summary, params, kernel, versions
p.save_csv(path)
p.to_data1d(dI=None) # needs debye-bye[sasdata]
p.plot()             # needs debye-bye[plot]

# Kernels, for people building on the library
dbb.kernels.debye_pairs(q, xyz, w, progress=None)
dbb.kernels.debye_lattice(q, grid_w, step)

# Misc
dbb.set_verbosity('info')
dbb.CalculationInterrupted, dbb.StructureError, dbb.DebyeByeWarning, ...
```

Changes from `gsc` that a user can see:

- `sans_fitter.structure.X` becomes `debyebye.X`; `sans_fitter.structure.debye` becomes `debyebye.kernels`.
- New: `compute()`, `q_grid()`, `ScatteringProfile.provenance()`, the warning categories, `set_verbosity()`, `examples.path()`.
- The `Process` attached by `to_data1d()` is named `'debyebye'`, not `'sans_fitter.structure'`.
- `get_settings()` gains `"schema": "debyebye.settings/1"` and records the `debye-bye` version instead of `sans-fitter`. `from_settings()` refuses an unknown schema and accepts settings without one (those written by `gsc`).
- Nothing prints until `set_verbosity()` is called or a host such as IQuana attaches a handler.

## 6. Dependencies and packaging

```toml
[project]
name = "debye-bye"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24", "periodictable>=1.7.1"]

[project.optional-dependencies]
plot = ["plotly>=5.0"]
sasdata = ["sasdata>=0.8"]
all = ["debye-bye[plot,sasdata]"]
test = ["pytest>=7.0", "pytest-cov>=4.0", "sasmodels>=1.0", "debye-bye[all]"]
docs = ["mkdocs>=1.5", "mkdocs-material>=9.5", "mkdocstrings[python]>=0.25"]
```

- numpy ≥ 1.24 rather than IQuana's ≥ 2.0, so the library can sit in older environments (SasView's included). CI tests both a numpy 1.x and a 2.x lower bound.
- The periodictable floor stays at 1.7.1, as `gsc` pinned it in pixi; scattering lengths depend on its tables, which is why its version is recorded in settings.
- pixi for development, as in IQuana, with conda-forge as the channel. Publish to PyPI first; a conda-forge feedstock once the API settles, so that IQuana's pixi environment can take it from conda-forge rather than PyPI.
- The name `debye-bye` still has to be checked and reserved on PyPI and TestPyPI.

## 7. IQuana side

What "importable by IQuana" comes down to:

1. **Dependency.** Add `debye-bye[sasdata]>=0.1,<0.2` to IQuana's required dependencies. IQuana already has plotly and sasdata, so this only adds periodictable. A required dependency keeps `from iquana import GenericScatteringCalculator` working; an optional `iquana[structure]` extra is the alternative if the dependency count matters more (see §12).
2. **Re-export, no wrapping.** `iquana/structure.py` re-exports the debyebye public names and `iquana/__init__.py` exports `GenericScatteringCalculator` and `structure`, as on `gsc`. IQuana adds no subclasses: a wrapper would drift from the library and give users two sets of docs.
3. **Accept a profile directly.** `IQuana.set_data()` accepts a `ScatteringProfile` as well as a `Data1D`, converting through `to_data1d()`. It then reminds the user, once, that the curve has no dI and no dQ and needs `set_weighting` and `set_resolution('none')`, as the `gsc` docs say. IQuana duck-types (checks for `q`, `intensity` and `provenance`) rather than `isinstance`, so it does not import debyebye at module import.
4. **Logging.** At import, IQuana adds its stdout handler to `logging.getLogger('debyebye')`, and `iquana.set_verbosity()` sets both loggers.
5. **Persistence.** `save_analysis()` keeps a profile as in-memory data, as on `gsc`, and also stores `profile.provenance()` (with the gsc settings when the profile came from a calculator), so a report says where the curve came from. Rebuilding the structure on reload stays out of scope.
6. **Tests.** `test_to_data1d_fits_with_sasmodels` and the plot-over-data test move to IQuana, together with a test that `set_data(profile)` works. `test_public_api.py` keeps its `GenericScatteringCalculator` and `structure` entries.
7. **Docs.** IQuana's usage guide keeps a short "Fitting a computed structure" section and links to the Debye-bye docs for everything else.

SANS-webapp, which already drives the fitter, would use `compute()` with a progress callback for a progress bar and a cancel button, and `Structure.summary()` for display. Nothing in the design is specific to it, but it is the reason D3 and D4 exist.

## 8. Provenance and settings

`ScatteringProfile.provenance()` returns:

```json
{
  "schema": "debyebye.profile/1",
  "structure": {"source": "...", "basename": "...", "source_sha256": "...", "format": "sld",
                "kind": "pixels", "n_points": 12167, "reader_options": {}, "notes": []},
  "params": {"scale": 1.0, "background": 0.0, "sld_solvent": 6.34, "normalisation_volume": "auto"},
  "normalisation_volume": 112563.0,
  "normalisation": "active-volume",
  "n_zero_contrast": 0,
  "method": "lattice",
  "notes": [],
  "versions": {"debye-bye": "0.1.0", "numpy": "2.1.3", "periodictable": "2.0.2"}
}
```

`get_settings()` / `from_settings()` keep their `gsc` behaviour and add the schema key. Both are JSON-serialisable, so IQuana's analysis files can embed them unchanged.

## 9. Tests and CI

- The 650 lines of `test_structure.py` move almost unchanged, split by module (`test_kernels.py`, `test_readers_*.py`, `test_compute.py`, `test_calculator.py`, `test_plotting.py`). Imports change from `sans_fitter...` to `debyebye...`.
- The sphere-vs-sasmodels physics test stays in Debye-bye, with sasmodels as a test-only dependency. It is the main check that the normalisation matches sasmodels at `scale=1`.
- New tests:
  - `import debyebye` with plotly and sasdata absent (a CI job installs only the core) does not fail, and the plot and `to_data1d` methods raise `ImportError` naming the extra;
  - nothing is printed on import or on `calculate()` until `set_verbosity()` is called;
  - each warning is raised with its category;
  - `compute()` and `GenericScatteringCalculator.calculate()` return the same profile;
  - `from_settings()` reads settings written by `gsc` (no schema key).
- CI: ruff format and lint as in IQuana; pytest on Linux, Windows and macOS; one job with only the core installed; one at the numpy and periodictable lower bounds.
- IQuana's CI runs its integration tests against the released `debye-bye`, plus one optional job against Debye-bye's `main` to catch breakage before a release.

## 10. Versioning and release

- SemVer from 0.1.0. Before 1.0 a minor release can break the API; IQuana pins `<0.(n+1)`.
- What counts as API: `debyebye.__all__`, the signatures and units in §5, the settings and provenance schemas, and the warning categories. Warning text and log lines are not API.
- A change to numerical output beyond rounding (a reader policy, an element constant, the normalisation) is called out in the changelog under "Changed results", even in a patch release.
- Release: a tag pushed to `main` builds the sdist and wheel and publishes to PyPI through trusted publishing; TestPyPI for release candidates.

## 11. Migration plan

1. **Create `ai4se1dk/Debye-bye`.** Copy `src/sans_fitter/structure/`, `tests/test_structure.py`, `examples/data/{c60.pdb,five_tetrahedra_cube.vtk,README.md}`, the example script, the notebook and the GSC section of `docs/usage.md`. The history is one commit, so `git filter-repo` buys little; the first commit message references `SANS-fitter@2e9674e`.
2. **Cut the couplings** listed in §2. At this point the tests pass with only numpy, periodictable, pytest and sasmodels installed.
3. **Make the API changes** in §3 and §5: `compute()`, `q_grid()`, the module renames, warning categories, quiet logging, lazy extras, provenance, settings schema.
4. **Release `0.1.0rc1`** to TestPyPI and try it from an IQuana branch.
5. **IQuana changes** from §7 on a branch, against the release candidate.
6. **Release `debye-bye` 0.1.0**, switch IQuana to it, merge.
7. **Retire `gsc`.** Leave a note on the branch pointing to Debye-bye and do not merge its structure package into SANS-fitter or IQuana.

## 12. After 0.1

**Closed-form solvent dependence, for fitting.** With `w_j = (s_j − ρ) v_j`, the Debye sum is exactly quadratic in the solvent SLD ρ:

```text
D(q; ρ) = A(q) − 2ρ B(q) + ρ² C(q)
A = Σ_jk s_j s_k v_j v_k sinc(q r_jk)
B = Σ_jk s_j v_j v_k   sinc(q r_jk)
C = Σ_jk v_j v_k       sinc(q r_jk)
```

One pass over the pairs, computing each sinc once and taking three dot products, gives A, B and C. After that, I(q) for any `scale`, `background` and `sld_solvent` costs O(len(q)). V_norm in auto mode is piecewise constant in ρ and only changes at exact matches, so it is cheap to track separately. This is what makes a structure fittable: a `StructureModel` that IQuana can fit like any sasmodels model, with those three parameters, and contrast-variation series in `MultiFitter` that share one structure. The kernels then take two weight vectors (`debye_pairs(q, xyz, w1, w2)`), and the lattice path uses cross-correlations instead of an autocorrelation.

**The structure model in IQuana.** Wrap a cached `(A, B, C)` as a sasmodels-compatible Python model so that IQuana's resolution handling and both fit engines work unchanged. Open point: sasmodels asks for intensities on its own expanded q grid when smearing, so the cache has to be per q array, or the model has to evaluate on demand.

**Faster kernels.** The kernel signatures are the backend contract. An optional numba or compiled backend for the pair sum (the N² path for atoms and cells) can sit behind them, selected at run time, with the NumPy version as the reference that tests compare against.

**Not planned:** 2D patterns, magnetic and polarised scattering, hydration shells. They are what separates this from SasView's full GSC, and each is a project of its own.

## 13. Open questions

1. **Required or optional in IQuana?** Required costs one pure-Python dependency (periodictable) and keeps the `gsc` import path. Optional (`iquana[structure]`) keeps IQuana's core smaller, but `iquana.GenericScatteringCalculator` then has to raise a helpful error when the extra is missing. Recommendation: required.
2. **Class name.** `GenericScatteringCalculator` matches SasView and the `gsc` docs. A shorter alias (`debyebye.Calculator`) would read better in the library's own docs. Recommendation: keep the long name as the primary and add no alias until users ask for one.
3. **Repository owner and licence notice.** `ai4se1dk/Debye-bye` under BSD-3-Clause, keeping the SasView copyright notice for the reader code derived from `sas.sascalc.calculator.sas_gen`. Confirm that is wanted, or whether the project should be offered to SasView as a standalone package later.
4. **SANS-fitter or IQuana as the reference consumer?** IQuana is the fitter's successor and lacks `structure`; this design assumes only IQuana integrates Debye-bye and SANS-fitter is left alone.
