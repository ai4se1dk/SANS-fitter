# Changelog

All notable changes to SANS-fitter are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] — `release_0.5`

Changes since 0.4.0 (`main` at `9eb15b4`).

### Added

- **Generic Scattering Calculator** (`GenericScatteringCalculator`, module
  `sans_fitter.structure`). The nuclear, 1D part of SasView's Generic
  Scattering Calculator, without 2D, magnetic or polarisation options.
  - Reads atoms from PDB files, pixels from SLD files and the cells of legacy
    ASCII VTK meshes, with each reader's policies and limits stated in the
    user guide.
  - Computes the orientationally averaged I(q) with the exact Debye sum. Pixels
    on a lattice use an exact FFT autocorrelation path; atoms and cells use the
    pair sum.
  - `plot_points()` draws the structure in 3D (SasView's *Draw Points*), and
    `plot_profile()` draws the curve, optionally over measured data.
  - `profile.to_data1d()` hands the curve to `SANSFitter` or `pr_inversion`;
    `get_settings()` and `from_settings()` record and replay a calculation.
  - The solvent SLD is `sld_solvent` in 10⁻⁶ Å⁻², as in sasmodels.
  - Walkthroughs: `examples/structure_calculator_example.py` and
    `notebooks/structure_calculator.ipynb`; example structures in
    `examples/data/`.
- **Simultaneous, constrained fitting** of several datasets (`MultiFitter`,
  #84, #85).
  - Each dataset has its own model, Q range and resolution.
  - `share()`, `link_params()` and `constrain()` relate parameters across
    datasets, with arithmetic expressions such as `'0.8 * h2o.scale'`.
  - Constraint bounds are enforced and shared parameters are counted once.
  - Results come as `MultiFitResult` and `MultiFitReport`, with per-dataset
    diagnostics, joint plots and a CSV export.
  - Guide: `docs/multifit.md`; walkthroughs:
    `examples/simultaneous_fitting_example.py` and
    `notebooks/simultaneous_fitting.ipynb`.
- **Data weighting options** (`set_weighting('dI' | 'sqrt' | 'abs' | 'none')`,
  #72 item 3, #86).
  - The four modes are SasView's Fit Page options.
  - The bumps engine and `fit_bayesian()` can now fit data without dI.
  - The mode appears in the fit header, `FitReport`, CSV header and HTML
    report, and is saved in analysis files (schema version 2). A saved fit is
    dropped on reload if the weighting changed.
  - Walkthrough: `examples/weighting_example.py`.
- **Four-form-factor walkthrough**: `notebooks/four_form_factors.ipynb` and
  `examples/four_form_factors_example.py` fit a sum of four form factors and
  show how to read the remaining parameter correlations.

### Changed

- `periodictable` is now a required dependency, used for PDB scattering
  lengths and densities.
- The console gained unit glyphs for Å, Å³ and 10⁻⁶ Å⁻², with ASCII fallbacks.

### Notes

- `MultiFitter` always weights each dataset by its own dI; weighting modes
  for joint fits are tracked in #87.
