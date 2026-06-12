# SANS-fitter — Deep Code Review (post-refactor)

**Branch:** `main` (HEAD `104ce11` "Refactoring (#14)", plus uncommitted line-ending churn) · **Date:** 2026-06-12
**Scope:** large-scale layout, class structure, coding accuracy. Security out of scope.
**Method:** full read of all source, tests, packaging, CI and docs. Behavioral claims about bumps/sasmodels were verified against the installed library sources in `.pixi/envs/default` (bumps `fitproblem.py`/`fitters.py`, sasmodels `bumps_model.py`/`direct_model.py`/`models/sphere.py`), not assumed. This supersedes the previous review of `fb9d514`; differences against that state are called out explicitly.

---

## 1. Executive summary

The refactor in #14 is a clear net improvement to the architecture. The former two-module design (886-line facade + 627-line manager) is now nine focused modules: `data_loader`, `plotting`, `results`, `structure_factor`, `polydispersity`, and a `fitting/` package in which the two engines are pure functions consuming an immutable-ish `ParameterStateSnapshot` and returning a uniform `EngineFitOutput`/`FitResultContract`. Several defects from the previous review were actually fixed along the way — most importantly the worst one (the scipy-engine plot/save bug). Test fixtures were de-duplicated into `tests/helpers.py`.

However, the refactor was structural, not substantive: **the core statistical defects survived it untouched**, two **new regressions** were introduced, and **none of the packaging/docs findings were addressed**. Current high-severity items:

1. **(Carried over)** scipy-engine parameter uncertainties omit the residual-variance scale factor — reported errors are wrong whenever reduced χ² ≠ 1 (`fitting/scipy_engine.py:77-91`).
2. **(Carried over)** the two engines report incompatible χ² definitions: bumps = DOF-normalized, scipy = raw sum of squares (`bumps_engine.py:70` vs `scipy_engine.py:81/91/101`).
3. **(Carried over)** the auto-generated default bounds clamp lower bounds to 0 and use `10×default` as upper bound — negative SLDs unreachable, degenerate `[0,0]` ranges for zero defaults — and the refactor **duplicated** this heuristic into a second module (`parameter_manager.py:119-120` and `structure_factor.py:75-76`).
4. **(New)** masked data points break the scipy engine and all plot/save paths: the loader now deliberately masks NaN rows, but `DirectModel` returns a curve only for unmasked points while `scipy_engine.residual`, `plotting.plot_fit` and `results.save_csv` all compute against full-length arrays — shape-mismatch crash or silently truncated CSV (§4.1, H4).
5. **(New)** an invalid `radius_effective_mode` now corrupts state instead of failing cleanly: the PD backup is taken *before* mode validation, and the kernel has already been swapped to the product model (§4.1, H5).

Fixed since last review and confirmed by code reading: the scipy plot/save PD bug (fitted curve is now captured at fit time via `build_parameter_dict(fitted_params)`, including PD config and `radius_effective` linking); PD input validation (`pd_width ≥ 0`, `pd_n > 0`, `pd_nsigma > 0`); the loader's NaN mask now covers `x`, `y` and `dy` and respects a pre-existing mask; fitted values now flow back through `set_param`, so a bumps `link_radius` fit re-syncs `radius_effective` in stored state; PD-inclusion logic is now defined once (`fitting/base.py:pd_is_active`) instead of three drifting copies.

---

## 2. Large-scale layout

### 2.1 New module map

```
src/sans_fitter/
  sans_fitter.py        520  facade: lifecycle, delegation, legacy result dict
  parameter_manager.py  533  coordinator over two sub-managers + param dict
  structure_factor.py   131  StructureFactorManager (SF state, FF backup)
  polydispersity.py     199  PolydispersityManager + PD_DEFAULTS/TYPES
  contracts.py           16  ParameterStateSnapshot (engine input)
  results.py             78  FitArtifacts, FitResultContract, CSV export
  data_loader.py         32  load_sans_data
  plotting.py           126  plot_fit (pure function of data + contract)
  fitting/
    base.py              52  pd_is_active, link helpers, EngineFitOutput, Protocol
    bumps_engine.py      80  fit_bumps
    scipy_engine.py     139  fit_scipy
```

This is the right decomposition. The engine boundary is the standout: `fit_bumps`/`fit_scipy` are stateless functions of `(data, kernel, ParameterStateSnapshot, method)` — engines no longer reach into `self`, the snapshot is taken via per-entry `dict()` copies (`parameter_manager.py:168-181`), and post-fit mutation is centralized in `SANSFitter._finalize_fit` → `apply_fitted_values`. A `FittingEngine` Protocol (`fitting/base.py:42-52`) documents the seam for future engines. The shared helpers `pd_is_active`, `link_radius_effective_model`, `link_radius_effective_dict` eliminate the previous review's triplicated-logic complaint.

### 2.2 Layout issues

**The engine is still publicly named `'lmfit'` while the module is now honestly named `scipy_engine`.** The refactor renamed the file but kept `engine='lmfit'`, `LMFIT_AVAILABLE` (now a pointless alias of `SCIPY_AVAILABLE`, `sans_fitter.py:40`), and `contract.engine = 'lmfit'` (`scipy_engine.py:129`). README still advertises "LMFit" with a link to lmfit-py and promises methods (`'powell'`, `'nelder'`) that `fit_scipy` rejects. The internal rename makes the external misnomer more conspicuous, not less. Rename the public engine to `'scipy'` with `'lmfit'` as a deprecated alias, and fix README/docs.

**`run_tests.py` survives unchanged** — still a redundant unittest runner whose `sys.path` hack is a no-op under src layout. The same dead hack moved into `tests/helpers.py:7-9` (the package is only importable because it's pip/pixi-installed). Delete both.

**Uncommitted line-ending churn.** `git status` shows essentially every file modified; the diffs are full-file rewrites with identical content (CRLF noise), and `tests/test_polydispersity.py` currently trips `grep` as a binary file. This will destroy `git blame` and produce unreviewable PRs. Add a `.gitattributes` (`* text=auto`, `*.py text eol=lf`) and renormalize once.

**Root-level data files** (`example_sans_data.dat`, `simulated_sans_data.csv`) and examples' CWD-relative paths — unchanged from last review.

---

## 3. Class structure

### 3.1 What improved

The old god-manager is now a coordinator over `StructureFactorManager` and `PolydispersityManager`, each independently testable with clear backup/restore semantics. `FitResultContract.to_legacy_dict()` (`results.py:27-42`) is a clean compatibility shim: the contract is the source of truth, the legacy dict is derived. `_finalize_fit` gives both engines a single post-fit path, fixing the previous asymmetry where each engine updated fitter state its own way.

### 3.2 Remaining and new structural problems

**`ParameterManager` carries ~60 lines of private-name property shims** (`parameter_manager.py:40-94`): `_structure_factor_name`, `_radius_effective_mode`, `_form_factor_params`, `_polydisperse_param_names`, `_pd_enabled`, `_backed_up_pd_state` — getter/setter pairs forwarding to the sub-managers. These exist to keep *private* attribute names stable, i.e., backward compatibility for code (tests) that pokes internals. That inverts the point of the refactor: the sub-managers are hidden behind a simulation of the old object's private layout. Worse, one shim is load-bearing in a fragile way: `initialize_from_kernel` does `self._polydisperse_param_names.append(param.name)` (`:127`), which only works because `PolydispersityManager.param_names` returns the **live list** (`polydispersity.py:23-25`) — while the adjacent `get_parameters()` returns a copy. If anyone "fixes" the property to return a copy too, polydispersity detection silently breaks with no test failure at the manager level. Collect names locally and pass them to `initialize()` instead.

**Public mutable state is unchanged.** `SANSFitter.params` / `model_name` still have public setters "used internally" that nothing internal uses (`sans_fitter.py:130-143`), and `fitter.params['x']['vary'] = True` still bypasses every invariant. The snapshot protects the *engines* from mid-fit mutation now, but pre-fit state is as open as before.

**The kernel/manager two-phase commit problem is unchanged and slightly worse** (see H5): `set_structure_factor` swaps `self.kernel` to the product model *before* the manager validates anything (`sans_fitter.py:222-227`), and `remove_structure_factor` reloads the kernel before the manager restore (`:251-254`). A failure between the two steps desynchronizes kernel and parameters.

**The legacy adapter quietly preserves a fixed bug.** `_get_active_fit_contract` (`sans_fitter.py:393-420`) rebuilds a contract from the legacy dict when `_fit_contract` is None — and its lmfit branch recomputes the curve from `fit_result['parameters']` values only, without PD config (`:412-419`): exactly the old plot-wrong-curve bug, kept alive as a fallback. In the normal flow `_fit_contract` is always set, so this path only triggers for code that assigns `fitter.fit_result` directly. Either support that use case correctly or drop the adapter and document `fit_result` as read-only.

**Duplication regression:** the default-bounds heuristic and the scale/background blocks are now copy-pasted in *two* modules (`parameter_manager.initialize_from_kernel` and `StructureFactorManager.apply`, `structure_factor.py:68-103`). When the bounds heuristic is fixed (H3), it must now be fixed twice. Extract a `build_param_entry(param)` helper.

**`ParameterManager`'s dead API got bigger, not smaller.** `get_param_dict`, `get_param_values`, `validate_param`, `update_param_value`, `has_backed_up_params`, `get_backed_up_params`, `has_backed_up_pd_state`, `get_pd_params_for_fitting` remain unused by the application (`get_pd_params_for_fitting` still differs from the engines' `pd_is_active` filter — it includes *all* PD params when enabled — so the latent inconsistency flagged last time also survived).

---

## 4. Correctness findings

### 4.1 High severity

**H1 — scipy-engine parameter uncertainties are statistically wrong (carried over, unchanged).** `scipy_engine.py:77-80` takes `sqrt(diag(cov_x))` from `leastsq`; `:85-90` takes `sqrt(diag(inv(JᵀJ)))` from `least_squares`. Both omit the residual-variance factor `s² = SSR/(N−p)`; scipy's own `leastsq` docs state `cov_x` must be multiplied by it. With weighted residuals the factor equals reduced χ², so at χ²ᵣ = 4 every reported error is 2× too small. `inv(JᵀJ)` is also numerically fragile — prefer `pinv`/SVD. The bumps path (`result.dx` from `driver.stderr()`) is fine.

**H2 — χ² is engine-dependent (carried over, unchanged).** `bumps_engine.py:70` stores `problem.chisq()`, which bumps normalizes by degrees of freedom (verified, `bumps/fitproblem.py:478-490`; target ≈ 1 for a good fit). `scipy_engine.py:81/91/101` stores raw `sum(residuals²)` (≈ N−p for a good fit). Same fit, ~30× different "χ²" in print-out, plot title, and the `# Chi-squared:` CSV header, with no label distinguishing them. Standardize on storing both raw and reduced values, computed identically in `FitResultContract`.

**H3 — Default-bounds heuristic still wrong, now in two places (carried over, worsened).** `min = 0` when the model declares `-inf`, `max = 10×default` when it declares `+inf` (`parameter_manager.py:119-120`, duplicated `structure_factor.py:75-76`). Verified against sasmodels: `sphere.sld` has limits `(-inf, inf)`, default 1 ⇒ generated bounds `[0, 10]`. Negative SLDs — routine in SANS contrast-variation work — are unreachable without manual override; a parameter with default 0 gets the degenerate range `[0, 0]`; a negative default produces `max < min`. Use declared finite limits; for open limits derive bounds from `|default|` with a floor and keep them sign-symmetric for signed quantities.

**H4 — Masked data points crash the scipy engine and all plot/save paths (new, latent).** The improved loader (`data_loader.py:22-29`) masks rows with NaN in `x`, `y` *or* `dy` — a good feature. But `DirectModel` evaluates the theory only at unmasked indices (verified: `direct_model.py:240-264` builds `index = (mask == 0) & ~isnan(y) & qmin/qmax` and constructs the resolution on `data.x[index]`), so for data with k masked points the returned curve has length N−k. Three consumers assume length N: `scipy_engine.residual` (`:69`, `(data.y - i_calc)/data.dy` → shape-mismatch `ValueError` *during the fit*), `plotting.plot_fit:44` (same expression), and `results.save_csv:53/68` (residual computation fails; if it didn't, the `zip` would silently misalign Q values against the truncated curve). Net effect: any data file containing a single NaN row fits under bumps (the `Experiment` handles indexing internally) but cannot be fitted with the scipy engine and cannot be plotted or saved at all. Fix by evaluating/storing the curve on the masked index and emitting masked rows explicitly (or filtering the arrays consistently everywhere).

**H5 — Invalid `radius_effective_mode` now corrupts state (new regression).** Previously `update_for_product_model` validated the mode *first*. Now the sequence in `set_structure_factor` is: (1) kernel swapped to the product model (`sans_fitter.py:222`); (2) `update_for_product_model` takes a **PD backup** (`parameter_manager.py:316-317`); (3) `StructureFactorManager.apply` raises for the bad mode (`structure_factor.py:57-60`). The exception is then wrapped as `"Failed to load model 'sphere@hardsphere': ..."` — a misleading message — and the object is left with: product-model kernel + form-factor params + a spurious PD backup. Because `update_for_product_model` only backs up PD state when no backup exists, the stale backup also poisons the *next* legitimate SF apply/remove cycle, silently restoring outdated PD state on `remove_structure_factor`. The existing test (`test_structure_factor.py:55-58`) still passes because it only does `assertIn('Invalid radius_effective_mode', str(e))` on the wrapped message and never inspects state afterwards. Fix: validate the mode at the top of `set_structure_factor` (before touching the kernel), and take backups only after all validation.

### 4.2 Medium severity

**M1 — Post-fit PD write-back can raise after a successful fit (new).** `apply_fitted_values` routes fitted PD widths through `set_pd_param` (`parameter_manager.py:188-191`), which now validates `pd_width ≥ 0` (`polydispersity.py:96-97`). The `leastsq` method ignores bounds (see M3), so a varying PD width can legitimately converge to a tiny negative value — at which point `_finalize_fit` raises `ValueError('pd_width must be non-negative')` *after* the optimizer succeeded, discarding the entire fit. The new validation and the unbounded optimizer are mutually inconsistent; clamp write-backs or bound the optimizer.

**M2 — `set_model(platform=...)` accepted and ignored (carried over).** Docstring advertises `'cpu'`/`'opencl'`; body hardcodes `platform='dll'`, `dtype='single'` (`sans_fitter.py:110`).

**M3 — `leastsq` silently ignores user bounds (carried over).** `scipy_engine.py` builds `bounds_lower/upper` (`:29-30, 39-40`) and the `leastsq` branch never uses them; convergence flag `ier` is also never checked, so non-convergence is reported as success (`:73-81`).

**M4 — Result schema still differs between engines (carried over).** bumps stores only varied parameters (`bumps_engine.py:59-65`); scipy stores varied + fixed (`scipy_engine.py:120-126`). The new `FitResultContract` standardizes the envelope but not the contents — a missed opportunity of the refactor, now cheap to fix in one place.

**M5 — `load_data` documented contract still wrong (carried over).** Docstring promises `FileNotFoundError` (`sans_fitter.py:86`); `data_loader.py:31-32` converts everything to `ValueError`. Test suite still codifies the wrong behavior.

**M6 — Hardcoded structure-factor whitelist (carried over).** `sans_fitter.py:210`. Query sasmodels instead.

**M7 — `data.dy` still unvalidated for zeros (carried over, narrowed).** The loader now masks NaN `dy` but `dy == 0` still produces `inf`/`nan` in `residual`, `plot_fit` and `save_csv` divisions. (And for files with no `dI` column, `dy=None` fails with `TypeError` rather than a clear message — note `data_loader.py:27` guards `hasattr(data, 'dy')` but not `dy is None`.)

**M8 — Dead-string test skips (carried over).** `test_fitting_lmfit.py:36/52` skip on `'lmfit is not installed'`; the code raises `'scipy is not installed'` (`sans_fitter.py:453`). These skips can never trigger.

### 4.3 Low severity / style

- `print` instead of `logging` throughout (facade, both engines, both display methods, plotting) — a library that cannot be silenced; the `✓`/`Å⁻¹`/`χ²` glyphs risk `UnicodeEncodeError` on legacy Windows consoles (the project targets win-64).
- `plot_fit` unconditionally calls `fig.show()` *and* returns the figure (`plotting.py:39/125`); add `show: bool = True`.
- `get_all_models()` swallows all exceptions and returns `[]` (`sans_fitter.py:32-37`).
- `LMFIT_AVAILABLE = SCIPY_AVAILABLE` plus a module-import-time `warnings.warn` (`sans_fitter.py:40-42`) — scipy is a hard dependency in `pyproject.toml`, so the entire optional-import path is dead weight.
- `test_fitting_bumps.py:47` still contains the no-op expression statement (`self.fitter.params['radius']['value']` computed and discarded) — the test asserts nothing about the update it is named for.
- `set_param` still performs no `min ≤ value ≤ max` cross-validation, and `vary=True` on a linked `radius_effective` is still accepted silently (the bumps engine then overwrites the parameter object after `.range()`; the scipy engine fits a dimension whose value the residual overwrites each evaluation — a wasted optimizer dimension).
- `FitArtifacts.runtime_handle`/`runtime_key` is a stringly-typed escape hatch used only to re-inject `'problem'` into the legacy dict; fine as transitional code, but it should not outlive the legacy dict itself.

### 4.4 Verified-OK notes (do not re-flag)

1. `link_radius_effective_model`'s `model.radius_effective = model.radius` works: sasmodels' `bumps_model.Model.parameters()` resolves attributes per call (verified `bumps_model.py:109-115`), and the assignment precedes `Experiment` construction (`bumps_engine.py:46-49`).
2. The mask convention is correct: sasmodels includes points where `mask == 0` (verified `direct_model.py:240-243`), matching the loader's "True = excluded" semantics. (The *length* consequence is H4, but the polarity is right.)
3. The stored bumps χ² is evaluated at the optimum: `bumps.fitters.fit()` calls `problem.setp(x)` before returning (verified `fitters.py:1464`).
4. The scipy engine's fitted curve is computed at fit time with the *full* parameter dict — PD width, `pd_n`, `pd_nsigma`, `pd_type` and `radius_effective` link included (`scipy_engine.py:134` via `build_parameter_dict`) — confirming the previous review's H3 is fixed for the normal flow.

---

## 5. Tests

The split into per-module files (`test_data_loader`, `test_fitting_bumps`, `test_fitting_lmfit`, `test_structure_factor`, `test_polydispersity`, `test_plotting`, `test_results`, `test_sans_fitter`) mirrors the new source layout, and `tests/helpers.py` eliminates the seven copy-pasted data-file factories — both real improvements.

What has *not* improved is assertion depth, and the suite demonstrably cannot see this review's findings: every fitting test still asserts only key presence ("result has 'chisq'"), no test generates data from a known sasmodels model and asserts parameter recovery (would catch H1–H3), no test fits or plots data containing a NaN row (would catch H4 — notable since the loader's NaN handling was explicitly extended in this refactor, the feature was built but never exercised end-to-end), no test inspects state after a failed `set_structure_factor` (would catch H5), and no test compares the two engines on the same problem (would catch H2 and M4). The PD-manager unit tests (Mock kernels) are thorough for configuration plumbing but stop at the fitting boundary.

Smaller points: `--disable-warnings` remains in `addopts`, hiding deprecation warnings from three fast-moving scientific dependencies; the dead skip strings (M8) and the no-op assertion line survived the file moves verbatim.

---

## 6. Packaging, CI, docs — unchanged, all previous findings stand

Nothing in this area was touched by the refactor (the working-tree diffs are line-ending-only). Verbatim from the previous review, still true at `104ce11`:

- `requires-python = ">=3.10"` vs ruff `target-version = "py39"`.
- `jupyter`, `notebook`, `ipykernel`, `ipywidgets` are hard runtime dependencies of a fitting library; move to an extra.
- Coverage `source = ["."]` measures examples and `run_tests.py`; point at `src/sans_fitter`.
- The CI test matrix lists Python 3.10–3.13 but every leg installs the single interpreter pinned in `pixi.lock` — four identical runs.
- Dual coverage services (Codecov in ci.yml, Codacy in a second workflow).
- `publish` depends only on `[test]`, so a lint-failing tag still publishes.
- `docs/api.md` still directs mkdocstrings at `ParameterManager` members that don't exist (`initialize_parameters`, `get_parameter_names`, `get_parameter`, `set_parameter`) and documents constants that don't exist (`DEFAULT_PD_WIDTH` etc. — the code exports `PD_DEFAULTS`). The refactor *moved* `PD_DEFAULTS` to `polydispersity.py` (re-exported in `__init__.py`, so imports keep working) but the docs were not updated. README still over-promises LMFit methods and links the demo notebook at the wrong path.

---

## 7. Prioritized recommendations

1. **H4**: make masked data work end-to-end — evaluate/store curves on the masked index; add a NaN-row fixture test for both engines plus plot/save.
2. **H1**: scale scipy covariances by `SSR/(N−p)`; check `ier`/`result.success`; use `pinv`.
3. **H5**: validate `radius_effective_mode` in `set_structure_factor` before swapping the kernel or taking backups; assert clean state in the invalid-mode test.
4. **H2 + M4**: compute raw and reduced χ² uniformly in `FitResultContract`; unify which parameters appear in `parameters` across engines.
5. **H3**: extract a single `build_param_entry` helper (kills the new duplication) and fix the bounds heuristic in it once.
6. **M1**: clamp PD write-backs in `apply_fitted_values` (or bound `leastsq` parameters) so a successful fit can't throw during finalization.
7. Rename engine `'lmfit'` → `'scipy'` (deprecated alias), drop `LMFIT_AVAILABLE`, fix README/docs method lists and `docs/api.md` member references.
8. Add truth-recovery tests (synthetic `DirectModel` data, assert recovery + stderr sanity, both engines); remove `--disable-warnings`; fix dead skips and the no-op assertion.
9. Replace the private-name property shims in `ParameterManager` with direct sub-manager use; stop appending through the live-list property; delete or wire up the dead manager API (incl. reconciling `get_pd_params_for_fitting` with `pd_is_active`).
10. Housekeeping: `.gitattributes` + line-ending renormalization; drop `run_tests.py` and the `helpers.py` path hack; ruff `py310`; notebook stack to an extra; coverage source; real CI matrix; single coverage service; `show=` on `plot_fit`; `logging` instead of `print`.

---

## 8. Verdict

The refactor did what a refactor should: the seams are now in the right places, the engines are isolated and testable, and one major user-facing bug was fixed in the process. But it also illustrates the limit of structure-only work — the statistically wrong error bars, the incomparable χ² values, and the unphysical default bounds all passed through untouched because no test constrains them, and the two genuine regressions (H4, H5) slipped in precisely where the test suite asserts existence rather than behavior. The highest-leverage next step is not more code movement; it is the truth-recovery and masked-data tests in §5, which would pin down the physics and make every remaining fix verifiable.
