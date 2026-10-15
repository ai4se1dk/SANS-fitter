"""Saving and reloading a complete analysis as JSON.

The file records the fitter's configuration, and separately the result of the
last fit when that result still describes the configuration. See
``SANSFitter.save_analysis`` and ``SANSFitter.load_analysis``.

Three rules shape this module:

**JSON, not pickle.** An analysis file is something a collaborator receives by
email. Pickle executes on load and is unreadable in a diff.

**JSON parsing safety is not model-loading safety.** ``_validate_model_expression``
deliberately forwards non-identifier atoms to sasmodels, which resolves
``custom.<name>`` to a Python plugin module, so a field read from a data-only
file can still cause code to run once it is interpreted as a model-loading
instruction. Only built-in model names are accepted unless the caller opts in.

**No bulk arrays.** Data curves and posterior chains stay out. A covariance
matrix does not: it is ``n_free`` by ``n_free``, and it is the most useful part
of a result to keep.
"""

import hashlib
import json
import math
import os
import warnings
from contextlib import contextmanager
from typing import Any

import numpy as np
from sasmodels import core

from . import __version__
from .console import OK, logger
from .data.provenance import DataSource, fingerprint_arrays
from .data.resolution import ResolutionSetting
from .fileio import atomic_write
from .report import json_safe
from .results import FitArtifacts, FitResultContract, PosteriorDigest

SCHEMA_FORMAT = 'sans-fitter-analysis'
SCHEMA_VERSION = 1
SUPPORTED_VERSIONS = (1,)

#: Relative tolerance for the reconstructed-chi-squared self-check.
CHISQ_TOLERANCE = 1e-6

_POSITIVE_INFINITY = 'Infinity'
_NEGATIVE_INFINITY = '-Infinity'


class AnalysisFileError(ValueError):
    """An analysis file is missing, malformed, or describes something else.

    A subclass of ``ValueError`` so existing ``except ValueError`` handlers
    keep working.
    """


# =========================================================================
# Numeric codec
# =========================================================================


def encode_numbers(value: Any, path: str = '') -> Any:
    """Make a configuration section strictly JSON-encodable.

    A ``JSONEncoder.default`` hook cannot do this: it is never called for
    built-in floats, so ``json.dumps({'b': float('inf')})`` emits a bare
    ``Infinity`` that is not JSON and that strict parsers reject. The document
    is normalized up front instead and written with ``allow_nan=False``, so a
    missed branch fails at save time rather than producing a file other tools
    cannot read.

    Infinite **bounds** are preserved as ``"Infinity"`` / ``"-Infinity"``: they
    are the default for ``scale`` and ``background``, and losing them would
    silently narrow a fit. ``report.json_safe`` is deliberately not used here,
    because it maps every non-finite float to None. NaN has no meaning in a
    configuration and is rejected.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            raise AnalysisFileError(f'Cannot save NaN at {path or "the document root"}.')
        if math.isinf(number):
            return _POSITIVE_INFINITY if number > 0 else _NEGATIVE_INFINITY
        return number
    if isinstance(value, np.ndarray):
        return [encode_numbers(item, f'{path}[{i}]') for i, item in enumerate(value.tolist())]
    if isinstance(value, dict):
        return {str(key): encode_numbers(item, f'{path}.{key}') for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [encode_numbers(item, f'{path}[{i}]') for i, item in enumerate(value)]
    raise AnalysisFileError(f'Cannot save a {type(value).__name__} at {path}.')


def decode_numbers(value: Any) -> Any:
    """Reverse :func:`encode_numbers` over a configuration section.

    Only the two infinity tokens are special. The configuration section holds
    no free text (model names, monikers and distribution names are all
    identifiers), so a string can be mapped back without ambiguity. The result
    section is never passed through here.
    """
    if isinstance(value, str):
        if value == _POSITIVE_INFINITY:
            return math.inf
        if value == _NEGATIVE_INFINITY:
            return -math.inf
        return value
    if isinstance(value, dict):
        return {key: decode_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_numbers(item) for item in value]
    return value


def config_digest(config: dict[str, Any]) -> str:
    """A stable hash of a manager configuration, for staleness comparison."""
    canonical = json.dumps(encode_numbers(config), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()


# =========================================================================
# Fit context
# =========================================================================


def build_fit_context(
    config: dict[str, Any],
    resolution: dict[str, Any],
    q_range: tuple[float, float] | None,
    n_points: int,
    data_fingerprint: str | None,
) -> dict[str, Any]:
    """Describe the configuration and data a fit result belongs to."""
    return {
        'config_digest': config_digest(config),
        'resolution': dict(resolution),
        'qmin': None if q_range is None else float(q_range[0]),
        'qmax': None if q_range is None else float(q_range[1]),
        'n_points': int(n_points),
        'data_fingerprint': data_fingerprint,
    }


def compare_fit_context(saved: dict[str, Any] | None, current: dict[str, Any]) -> str | None:
    """Return why a saved result no longer describes *current*, or None if it does.

    Checked facet by facet so the message can name what moved. Comparing fitted
    parameter values alone would not do: polydispersity, bounds, vary flags,
    links, the Q selection, the resolution mode and the data itself all change
    what a chi-squared means.
    """
    if saved is None:
        return 'the fit predates analysis saving, so its configuration is unknown'
    checks = (
        ('config_digest', 'the model or its parameters changed after the fit'),
        ('resolution', 'the resolution setting changed after the fit'),
        ('qmin', 'the fitting Q range changed after the fit'),
        ('qmax', 'the fitting Q range changed after the fit'),
        ('data_fingerprint', 'the data changed after the fit'),
    )
    for key, reason in checks:
        if saved.get(key) != current.get(key):
            return reason
    return None


# =========================================================================
# Result adapter
# =========================================================================


def _posterior_to_dict(posterior: Any) -> dict[str, Any] | None:
    """Statistics only. The sample chain is bulk data and is not persisted."""
    if posterior is None:
        return None
    return {
        'labels': list(posterior.labels),
        'n_samples': int(posterior.n_samples),
        'n_params': int(posterior.n_params),
        'best': dict(posterior.best),
        'mean': dict(posterior.mean),
        'median': dict(posterior.median),
        'std': dict(posterior.std),
        'ci_68': {name: list(bounds) for name, bounds in posterior.ci_68.items()},
        'ci_95': {name: list(bounds) for name, bounds in posterior.ci_95.items()},
        'diagnostics': posterior.diagnostics,
    }


def _posterior_from_dict(payload: dict[str, Any] | None) -> PosteriorDigest | None:
    if payload is None:
        return None
    return PosteriorDigest(
        labels=list(payload['labels']),
        n_samples=int(payload['n_samples']),
        n_params=int(payload['n_params']),
        best=dict(payload.get('best', {})),
        mean=dict(payload.get('mean', {})),
        median=dict(payload.get('median', {})),
        std=dict(payload.get('std', {})),
        ci_68={name: tuple(bounds) for name, bounds in payload.get('ci_68', {}).items()},
        ci_95={name: tuple(bounds) for name, bounds in payload.get('ci_95', {}).items()},
        diagnostics=payload.get('diagnostics'),
    )


def result_to_dict(contract: FitResultContract) -> dict[str, Any]:
    """Serialize the whole result contract, not a narrower parallel summary.

    Every field a reader needs is here, including the ones that are easy to
    forget: ``weighting_note`` is a required constructor argument, and each
    parameter's ``formatted`` string is read directly by
    ``FitResultContract.save_csv``.
    """
    payload = {
        'engine': contract.engine,
        'method': contract.method,
        'chisq': contract.chisq,
        'reduced_chisq': contract.reduced_chisq,
        'n_points': contract.n_points,
        'n_free': contract.n_free,
        'dof': contract.dof,
        'weighting_note': contract.weighting_note,
        'converged': contract.converged,
        'message': contract.message,
        'resolution': contract.resolution,
        'parameters': {name: dict(info) for name, info in contract.parameters.items()},
        'cov': None if contract.cov is None else np.asarray(contract.cov).tolist(),
        'cov_labels': list(contract.cov_labels),
        'cov_source': contract.cov_source,
        'on_bounds': [list(hit) for hit in contract.on_bounds],
        'posterior': _posterior_to_dict(contract.artifacts.posterior),
        'fit_context': contract.fit_context,
    }
    # Statistics, unlike bounds, are legitimately unavailable: non-finite means
    # "no value", and null is the honest spelling of that.
    return json_safe(payload)


def _normalize_linked_uncertainty(
    parameters: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Give equality followers saved before 0.5 the uncertainty of their target.

    Files written by 0.4 and earlier stored ``stderr = 0.0`` for every follower,
    because that is what the engines reported then. That zero is not a
    measurement: an equality link makes follower and target one quantity, so the
    follower carries whatever its target carries.

    Three cases, and they are not the same:

    - **a fixed target** never moved, so zero is correct and is left alone;
    - **a varying target** hands over its error verbatim, including ``None``
      when that error could not be estimated — propagating the absence rather
      than reading the legacy zero as a precise measurement;
    - **a target that is not in the file at all** leaves the follower's
      uncertainty unknowable, so it becomes ``None`` rather than zero.

    A compatibility normalization on read, not a schema change: nothing about
    the stored document changes, and a file written by this version already
    carries the corrected numbers, so this is a no-op for it.
    """
    for name, info in parameters.items():
        target = info.get('linked_to')
        if not target or info.get('stderr'):
            continue

        source = parameters.get(str(target))
        if source is None:
            info['stderr'] = None
            info['formatted'] = f'{_display_value(info)} (uncertainty unavailable)'
            logger.debug(
                f"Restored parameter '{name}' follows '{target}', which is not in the "
                'file; its uncertainty is recorded as unavailable.'
            )
            continue
        if source.get('fixed', True) and not source.get('linked_to'):
            continue  # a genuinely fixed target: zero is the right answer

        info['stderr'] = source.get('stderr')
        info['formatted'] = (
            source.get('formatted', info.get('formatted'))
            if info['stderr']
            else f'{_display_value(info)} (uncertainty unavailable)'
        )
    return parameters


def _display_value(info: dict[str, Any]) -> str:
    try:
        return f'{float(info.get("value")):.6g}'
    except (TypeError, ValueError):
        return str(info.get('value'))


def result_from_dict(payload: dict[str, Any]) -> FitResultContract:
    """Rebuild a contract from :func:`result_to_dict`, without artifacts."""
    _require(isinstance(payload, dict), 'result', 'must be a mapping')
    for key in ('engine', 'method', 'chisq', 'parameters', 'weighting_note'):
        _require(key in payload, f'result.{key}', 'is required')

    cov = payload.get('cov')
    labels = list(payload.get('cov_labels') or [])
    matrix = None
    if cov is not None:
        matrix = np.asarray(cov, dtype=float)
        _require(
            matrix.ndim == 2 and matrix.shape == (len(labels), len(labels)),
            'result.cov',
            f'has shape {matrix.shape}, which does not match {len(labels)} label(s)',
        )

    return FitResultContract(
        engine=str(payload['engine']),
        method=str(payload['method']),
        chisq=_float_or_nan(payload.get('chisq')),
        reduced_chisq=_float_or_nan(payload.get('reduced_chisq')),
        n_points=int(payload.get('n_points') or 0),
        n_free=int(payload.get('n_free') or 0),
        dof=int(payload.get('dof') or 0),
        weighting_note=str(payload['weighting_note']),
        parameters=_normalize_linked_uncertainty(
            {str(name): dict(info) for name, info in (payload.get('parameters') or {}).items()}
        ),
        resolution=payload.get('resolution'),
        converged=payload.get('converged'),
        message=str(payload.get('message') or ''),
        cov=matrix,
        cov_labels=labels,
        cov_source=payload.get('cov_source'),
        on_bounds=[tuple(hit) for hit in (payload.get('on_bounds') or [])],
        fit_context=payload.get('fit_context'),
        artifacts=FitArtifacts(posterior=_posterior_from_dict(payload.get('posterior'))),
    )


def _float_or_nan(value: Any) -> float:
    return float('nan') if value is None else float(value)


# =========================================================================
# Model trust boundary
# =========================================================================


def is_builtin_expression(expression: str) -> bool:
    """True when every atom of *expression* is a built-in sasmodels model."""
    available = set(core.list_models())
    for part in expression.replace('*', '+').split('+'):
        for atom in part.split('@'):
            if atom.strip() not in available:
                return False
    return True


# =========================================================================
# Writing
# =========================================================================


def _relative_path(target: str, analysis_path: str) -> str | None:
    """*target* relative to the analysis file's directory, when expressible.

    On Windows a path on another drive has no relative form at all, so this
    returns None rather than inventing one; the absolute path carries the
    analysis in that case.
    """
    try:
        return os.path.relpath(target, os.path.dirname(os.path.abspath(analysis_path)))
    except ValueError:
        return None


def _data_section(source: DataSource, data: Any, analysis_path: str) -> dict[str, Any]:
    section: dict[str, Any] = {
        'source': source.kind,
        'label': source.label,
        'n_points': source.n_points,
        'array_fingerprint_at_load': source.array_fingerprint,
        'array_fingerprint_now': fingerprint_arrays(data),
        'processes': list(source.processes),
    }
    if source.kind == 'file':
        section['path_absolute'] = source.path
        section['path_relative'] = _relative_path(source.path or '', analysis_path)
        section['dataset'] = source.dataset_requested
        section['dataset_index'] = source.dataset_index
        section['n_datasets'] = source.n_datasets
        section['file_sha256_at_load'] = source.file_sha256
        section['file_sha256_now'] = source.current_file_sha256()
    return section


def build_analysis_dict(
    fitter: Any, analysis_path: str, *, include_result: bool = True
) -> dict[str, Any]:
    """Assemble the analysis document for *fitter*."""
    if fitter.kernel is None:
        raise AnalysisFileError('No model to save. Use set_model() or set_models() first.')

    config = fitter._param_manager.export_config()
    document: dict[str, Any] = {
        'schema': {
            'format': SCHEMA_FORMAT,
            'version': SCHEMA_VERSION,
            'written_by': __version__,
            'sasmodels': _package_version('sasmodels'),
        },
        'data': (
            None
            if fitter._data_source is None
            else _data_section(fitter._data_source, fitter.data, analysis_path)
        ),
        'configuration': encode_numbers(config, 'configuration'),
        'resolution': fitter.get_resolution(),
        'weighting': None,  # reserved for item 3 of #72
        'fit_range': _fit_range_section(fitter),
        'result': None,
        'result_omitted': None,
    }

    if include_result:
        contract = fitter._fit_contract
        if contract is not None:
            reason = compare_fit_context(contract.fit_context, _current_fit_context(fitter, config))
            if reason is None:
                document['result'] = result_to_dict(contract)
            else:
                document['result_omitted'] = reason
    return document


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001 - a missing distribution must not block a save
        return None


def _fit_range_section(fitter: Any) -> dict[str, Any] | None:
    """The restricted Q range, or None when the full range is in use."""
    if fitter.data is None or fitter._full_q_range is None:
        return None
    qmin, qmax = fitter.get_q_range()
    full_min, full_max = fitter._full_q_range
    if qmin == full_min and qmax == full_max:
        return None
    return {'qmin': float(qmin), 'qmax': float(qmax)}


def _current_fit_context(fitter: Any, config: dict[str, Any]) -> dict[str, Any]:
    """The fit context the fitter would record if it were fitting right now."""
    return build_fit_context(
        config,
        fitter.get_resolution(),
        fitter.get_q_range(),
        _fitted_point_count(fitter),
        None if fitter.data is None else fingerprint_arrays(fitter.data),
    )


def _fitted_point_count(fitter: Any) -> int:
    from .data.loader import get_fit_index

    if fitter.data is None:
        return 0
    return int(get_fit_index(fitter.data).sum())


def write_analysis(fitter: Any, filename: str, *, include_result: bool = True) -> dict[str, Any]:
    """Write *fitter* to *filename* and return the document that was written.

    Rendered fully before anything is replaced, through a temporary sibling and
    an atomic rename, so a failure part-way cannot destroy a good analysis file
    that is already there.
    """
    document = build_analysis_dict(fitter, filename, include_result=include_result)
    text = json.dumps(document, indent=2, allow_nan=False, ensure_ascii=False)

    target = os.path.abspath(filename)
    directory = os.path.dirname(target)
    if directory and not os.path.isdir(directory):
        raise AnalysisFileError(f'Directory does not exist: {directory}')
    atomic_write(target, text)
    return document


# =========================================================================
# Reading
# =========================================================================


def _require(condition: bool, path: str, message: str) -> None:
    if not condition:
        raise AnalysisFileError(f'Analysis file: {path} {message}.')


def read_document(filename: str) -> dict[str, Any]:
    """Load and validate the envelope of an analysis file."""
    if not os.path.isfile(filename):
        raise AnalysisFileError(f'Analysis file not found: {os.path.abspath(filename)}')
    try:
        with open(filename, encoding='utf-8') as handle:
            document = json.load(handle, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise AnalysisFileError(f'Analysis file is not valid JSON: {exc}') from exc

    _require(isinstance(document, dict), 'the document', 'must be a JSON object')
    schema = document.get('schema')
    _require(isinstance(schema, dict), 'schema', 'is required and must be a mapping')
    _require(
        schema.get('format') == SCHEMA_FORMAT,
        'schema.format',
        f'must be {SCHEMA_FORMAT!r}, got {schema.get("format")!r}',
    )
    version = schema.get('version')
    _require(
        isinstance(version, int) and not isinstance(version, bool),
        'schema.version',
        'must be an integer',
    )
    if version > SCHEMA_VERSION:
        raise AnalysisFileError(
            f'Analysis file: schema.version {version} was written by a newer '
            f'sans-fitter than this one (which supports {SCHEMA_VERSION}). Upgrade to read it.'
        )
    _require(
        version in SUPPORTED_VERSIONS,
        'schema.version',
        f'{version} is not one this version can migrate from (supported: '
        f'{", ".join(str(v) for v in SUPPORTED_VERSIONS)})',
    )
    _require('configuration' in document, 'configuration', 'section is required')
    return document


def _reject_constant(token: str) -> Any:
    raise AnalysisFileError(
        f'Analysis file contains the non-standard JSON constant {token}. '
        'Values are written as null or as an explicit "Infinity" string.'
    )


def resolve_data(
    document: dict[str, Any], filename: str, data: Any = None
) -> tuple[Any, str | int | None]:
    """Decide which dataset to load, returning ``(source, dataset_selector)``.

    *source* is a path to load, an in-memory dataset to adopt, or None when the
    caller must supply one. An explicit *data* wins outright: it was requested,
    so it fails as the requested override rather than falling back silently to
    a different sample.
    """
    section = document.get('data')
    if data is not None:
        if isinstance(data, (str, os.PathLike)):
            path = os.fspath(data)
            if not os.path.isfile(path):
                raise AnalysisFileError(f'Data file given as data= not found: {path}')
            return path, (section or {}).get('dataset', 0)
        return data, None

    if not isinstance(section, dict):
        raise AnalysisFileError(
            'This analysis was saved without any data. Supply one with '
            'load_analysis(path, data=...).'
        )
    if section.get('source') != 'file':
        detail = '; '.join(section.get('processes') or []) or section.get('label', '')
        raise AnalysisFileError(
            'This analysis used an in-memory dataset, which cannot be reloaded '
            f'from the file. Supply it with load_analysis(path, data=...). It was: {detail}'
        )

    tried: list[str] = []
    base = os.path.dirname(os.path.abspath(filename))
    relative = section.get('path_relative')
    if relative:
        candidate = os.path.normpath(os.path.join(base, relative))
        tried.append(candidate)
        if os.path.isfile(candidate):
            return candidate, section.get('dataset', 0)
    absolute = section.get('path_absolute')
    if absolute:
        tried.append(absolute)
        if os.path.isfile(absolute):
            return absolute, section.get('dataset', 0)
    raise AnalysisFileError(
        'Could not find the data file for this analysis. Tried: '
        + '; '.join(tried or ['no path recorded'])
        + '. Supply it with load_analysis(path, data=...).'
    )


def warn_about_source_changes(document: dict[str, Any], data: Any, supplied: bool) -> None:
    """Warn when the data no longer matches what the analysis was saved from."""
    section = document.get('data')
    if not isinstance(section, dict):
        return
    if fingerprint_arrays(data) == section.get('array_fingerprint_now'):
        return
    if supplied:
        warnings.warn(
            'The dataset supplied with data= is not the one this analysis was '
            'saved from. The configuration is restored; any saved fit result is not.',
            stacklevel=3,
        )
    else:
        warnings.warn(
            'The data file has changed since this analysis was saved. The '
            'configuration is restored; any saved fit result is not.',
            stacklevel=3,
        )


def restore_configuration(
    fitter: Any, document: dict[str, Any], *, allow_custom_models: bool
) -> None:
    """Rebuild the model, alias layer, structure factor and parameter state."""
    config = decode_numbers(document['configuration'])
    _require(isinstance(config, dict), 'configuration', 'must be a mapping')

    expression = config.get('model_name')
    _require(isinstance(expression, str) and expression, 'configuration.model_name', 'is required')
    if not allow_custom_models and not is_builtin_expression(expression):
        raise AnalysisFileError(
            f"Analysis file: model expression '{expression}' is not built into "
            'sasmodels. Loading it would import a plugin module named by the '
            'file. Pass allow_custom_models=True if you trust this file.'
        )

    fitter.set_model(expression)

    components = [tuple(entry) for entry in (config.get('components') or [])]
    if components:
        _verify_components(fitter, components)
        pairs = [(moniker, part_name) for _prefix, moniker, part_name in components]
        with _as_file_error('configuration.components'):
            fitter._param_manager.register_aliases(pairs, list(config.get('shared') or []))

    sf_section = config.get('structure_factor') or {}
    sf_name = sf_section.get('name')
    if sf_name:
        with _as_file_error('configuration.structure_factor'):
            fitter.set_structure_factor(
                sf_name, sf_section.get('radius_effective_mode', 'unconstrained')
            )

    # One bulk restore rather than a replay of the public setters: see
    # ParameterManager.export_config for why the setters cannot do this.
    with _as_file_error('configuration'):
        fitter._param_manager.import_config(config)


@contextmanager
def _as_file_error(section: str):
    """Re-raise a manager validation failure as an analysis-file failure.

    The managers raise plain ``ValueError`` because they are normally driven by
    a user calling setters, where that is the right type. Reaching them through
    a file makes the same failure a statement about the file, so callers can
    catch ``AnalysisFileError`` and know the difference.
    """
    try:
        yield
    except AnalysisFileError:
        raise
    except (ValueError, KeyError) as exc:
        message = exc.args[0] if exc.args else str(exc)
        raise AnalysisFileError(f'Analysis file: {section}: {message}') from exc


def _verify_components(fitter: Any, saved: list[tuple[str, str, str]]) -> None:
    """Check the saved component metadata against the kernel actually loaded.

    ``register_aliases`` overlays monikers by position and never checks the
    model name at that position, so without this a hand-edited file could
    silently rename another component's parameters.
    """
    from .modeling.parameters import derive_mixture_components

    derived = derive_mixture_components(fitter.kernel)
    _require(
        len(derived) == len(saved),
        'configuration.components',
        f'lists {len(saved)} component(s) but the model has {len(derived)}',
    )
    for position, (derived_entry, saved_entry) in enumerate(zip(derived, saved, strict=True)):
        derived_prefix, _derived_moniker, derived_part = derived_entry
        saved_prefix, _saved_moniker, saved_part = saved_entry
        _require(
            (derived_prefix, derived_part) == (saved_prefix, saved_part),
            f'configuration.components[{position}]',
            f'describes {saved_part!r} at prefix {saved_prefix!r}, but the loaded '
            f'model has {derived_part!r} at {derived_prefix!r}',
        )


def restore_result(fitter: Any, document: dict[str, Any]) -> str | None:
    """Attach the saved result to *fitter*, or return why it was not attached."""
    payload = document.get('result')
    if payload is None:
        return document.get('result_omitted')

    contract = result_from_dict(payload)
    config = fitter._param_manager.export_config()
    reason = compare_fit_context(contract.fit_context, _current_fit_context(fitter, config))
    if reason is not None:
        return reason.replace('after the fit', 'since the fit')

    problem = _rebuild_artifacts(fitter, contract)
    if problem is not None:
        return problem

    fitter._fit_contract = contract
    fitter.fit_result = contract.to_legacy_dict()
    return None


def _rebuild_artifacts(fitter: Any, contract: FitResultContract) -> str | None:
    """Recompute the curve, index and residuals, and check them against the file.

    Evaluated through ``_evaluation_data`` so the restored resolution mode is
    applied, exactly as a fit would. Nothing is attached unless the rebuilt
    point count and chi-squared agree with what was saved, so a file whose
    numbers do not describe its configuration produces an explanation rather
    than a plausible-looking wrong plot.
    """
    if fitter.data is None:
        return 'no data is loaded, so the fitted curve cannot be rebuilt'

    evaluation_data = fitter._evaluation_data(warn=False)
    curve, fit_index = fitter._evaluate(evaluation_data)

    n_points = int(fit_index.sum())
    if n_points != contract.n_points:
        return (
            f'the rebuilt fit selects {n_points} points but the saved result '
            f'used {contract.n_points}'
        )

    residuals = _rebuild_residuals(evaluation_data, curve, fit_index, contract.engine)
    if residuals is not None and math.isfinite(contract.chisq):
        rebuilt = float(np.sum(residuals**2))
        if not math.isclose(rebuilt, contract.chisq, rel_tol=CHISQ_TOLERANCE, abs_tol=1e-12):
            return (
                f'the rebuilt chi-squared ({rebuilt:.6g}) does not match the saved '
                f'value ({contract.chisq:.6g})'
            )

    contract.artifacts.fitted_curve = curve
    contract.artifacts.fit_index = fit_index
    contract.artifacts.residuals = residuals
    contract.artifacts.component_curves = fitter._compute_component_curves()
    return None


#: Engines that report ``(theory - I) / dI`` rather than ``(I - theory) / dI``.
#: bumps' ``problem.residuals()`` uses the first convention and the scipy engine
#: the second, so the two differ in sign for the same fit. chi-squared cannot
#: reveal which, being a sum of squares, but the exported residual column and
#: the residual panel of a plot both would.
_NEGATED_RESIDUAL_ENGINES = frozenset({'bumps'})


def _rebuild_residuals(
    data: Any, curve: np.ndarray, fit_index: np.ndarray, engine: str
) -> np.ndarray | None:
    """The weighted residuals *engine* would have produced, or None without dI.

    Two engine conventions have to be honoured, not one. Both weight by dI and
    unit-weight zero-dI points (deriving ``(I - fit) / dI`` naively would write
    inf at exactly those points, and the exported residuals would no longer
    square-sum to the reported chi-squared), but they differ in sign: see
    ``_NEGATED_RESIDUAL_ENGINES``.
    """
    if data.dy is None:
        return None
    dy = np.asarray(data.dy, dtype=float)[fit_index]
    if dy.size == 0:
        return None
    y = np.asarray(data.y, dtype=float)[fit_index]
    if engine in _NEGATED_RESIDUAL_ENGINES:
        y, curve = curve, y
    sigma = np.where(np.nan_to_num(dy) == 0, 1.0, dy)
    return (y - curve) / sigma


def read_analysis(
    fitter_cls: type,
    filename: str,
    *,
    data: Any = None,
    allow_custom_models: bool = False,
) -> Any:
    """Build a configured fitter from the analysis file at *filename*.

    *fitter_cls* is passed in rather than imported so this module stays free of
    a circular import with :mod:`sans_fitter.fitter`.
    """
    document = read_document(filename)
    fitter = fitter_cls()

    source, selector = resolve_data(document, filename, data)
    if isinstance(source, str):
        fitter.load_data(source, dataset=0 if selector is None else selector)
    else:
        fitter.set_data(source)
    warn_about_source_changes(document, fitter.data, supplied=data is not None)

    restore_configuration(fitter, document, allow_custom_models=allow_custom_models)

    resolution = document.get('resolution')
    if isinstance(resolution, dict):
        setting = ResolutionSetting(**resolution)
        fitter.set_resolution(
            setting.mode,
            dq_over_q=setting.dq_over_q,
            slit_length=setting.slit_length,
            slit_width=setting.slit_width,
        )

    fit_range = document.get('fit_range')
    if isinstance(fit_range, dict):
        fitter.set_q_range(fit_range.get('qmin'), fit_range.get('qmax'))

    omitted = restore_result(fitter, document)
    if omitted:
        logger.info(f'  Fit result not restored: {omitted}.')
    logger.info(f'{OK} Analysis loaded from {filename}')
    return fitter
