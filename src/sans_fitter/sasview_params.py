"""
Parser for SasView parameter export files.

Supports the plain-text CSV format exported by SasView's parameter save
feature.  Phase 1 covers only plain form-factor scalar parameters.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

MAGIC_HEADER = 'sasview_parameter_values'


@dataclass
class SasViewParam:
    """A single parsed parameter row from a SasView export file."""

    name: str
    vary: bool
    value: float
    stderr: float | None
    min: float | None
    max: float | None


@dataclass
class SasViewParamFile:
    """Parsed representation of a complete SasView parameter export file."""

    model_name: str
    params: list[SasViewParam] = field(default_factory=list)


def _parse_optional_float(raw: str) -> float | None:
    """Parse a string into a float, returning None for empty/None strings."""
    stripped = raw.strip()
    if stripped == '' or stripped == 'None':
        return None
    return float(stripped)


def _parse_bound(raw: str) -> float | None:
    """Parse a bound field (min/max), handling inf / -inf / empty."""
    stripped = raw.strip()
    if stripped == '':
        return None
    if stripped == 'inf':
        return float('inf')
    if stripped == '-inf':
        return float('-inf')
    return float(stripped)


def _parse_vary(raw: str) -> bool | None:
    """Parse the vary column.  Returns None for separator rows."""
    stripped = raw.strip()
    if stripped == 'True':
        return True
    if stripped == 'False':
        return False
    if stripped == 'None' or stripped == '':
        return None
    raise ValueError(f'Unexpected vary value: {stripped!r}')


def parse_sasview_params(filepath: str | Path) -> SasViewParamFile:
    """Parse a SasView parameter export file.

    Parameters
    ----------
    filepath : str | Path
        Path to the SasView parameter file.

    Returns
    -------
    SasViewParamFile
        Parsed model name and parameter list.

    Raises
    ------
    ValueError
        If the file is empty, has the wrong header, is missing the
        ``model_name`` row, or contains malformed parameter rows.
    """
    filepath = Path(filepath)
    text = filepath.read_text(encoding='utf-8')
    lines = text.strip().splitlines()

    if not lines:
        raise ValueError('SasView parameter file is empty')

    # --- magic header ---
    if lines[0].strip() != MAGIC_HEADER:
        raise ValueError(f'Expected header {MAGIC_HEADER!r}, got {lines[0].strip()!r}')

    if len(lines) < 2:
        raise ValueError('Missing model_name row after header')

    # --- model_name row ---
    reader = csv.reader(io.StringIO(lines[1]))
    model_row = next(reader)
    if len(model_row) < 2 or model_row[0].strip() != 'model_name':
        raise ValueError(f"Expected 'model_name,<name>' on line 2, got {lines[1]!r}")
    model_name = model_row[1].strip()

    # --- parameter rows ---
    params: list[SasViewParam] = []
    for lineno, line in enumerate(lines[2:], start=3):
        row = next(csv.reader(io.StringIO(line)))

        # Expect 7 fields: name, vary, value, stderr, min, max, constraints
        if len(row) < 6:
            raise ValueError(f'Line {lineno}: expected at least 6 CSV fields, got {len(row)}')

        name = row[0].strip()
        vary = _parse_vary(row[1])

        # Separator rows have vary=None and empty value — skip them
        value_raw = row[2].strip()
        if vary is None or value_raw == '':
            continue

        try:
            value = float(value_raw)
        except ValueError:
            raise ValueError(
                f'Line {lineno}: cannot parse value {value_raw!r} for parameter {name!r}'
            ) from None

        stderr = _parse_optional_float(row[3])
        min_bound = _parse_bound(row[4])
        max_bound = _parse_bound(row[5])

        params.append(
            SasViewParam(
                name=name,
                vary=vary,
                value=value,
                stderr=stderr,
                min=min_bound,
                max=max_bound,
            )
        )

    return SasViewParamFile(model_name=model_name, params=params)
