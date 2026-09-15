"""Where a fitter's dataset came from, recorded when it is ingested.

``load_data()`` and ``set_data()`` are the only places that know the answer.
``load_data(filename, dataset=...)`` resolves a name to a position and then
drops both the filename and the selector, and a dataset's ``.filename``
metadata is not an authoritative path (a text file loaded from another
directory carries only its basename, and ``data_ops`` results carry an
operation string such as ``'sample - background'``). So a saved analysis
cannot reconstruct afterwards which file, or which entry of a multi-entry
file, the numbers came from: the record has to be made at ingestion.

Fingerprints are deliberately taken of two different things:

- ``file_sha256`` is the file on disk **as it was when loaded**. Re-hashing at
  save time and comparing detects a source that changed underneath a long
  notebook session.
- ``array_fingerprint`` is the data actually held by the fitter. It is the one
  that matters for reproducibility, because it also moves when a caller edits
  ``data.y`` or ``data.mask`` in memory, which no file hash can see.
"""

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_ARRAY_FIELDS = ('x', 'y', 'dy', 'dx', 'dxl', 'dxw')


def _hash_file(path: str) -> str | None:
    """SHA-256 of a file, or None when it cannot be read."""
    try:
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for block in iter(lambda: handle.read(1 << 20), b''):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def fingerprint_arrays(data: Any) -> str:
    """SHA-256 over the numeric content of a dataset.

    Covers every array sasmodels may read (``x``, ``y``, ``dy`` and the four
    resolution columns) plus the mask, each tagged by name so that moving
    values between columns changes the result. Absent columns contribute their
    name alone, so "no dx" and "dx of zeros" are distinguishable.
    """
    digest = hashlib.sha256()
    for name in _ARRAY_FIELDS:
        digest.update(name.encode())
        values = getattr(data, name, None)
        if values is None:
            digest.update(b'|none')
            continue
        digest.update(b'|')
        digest.update(np.ascontiguousarray(values, dtype=float).tobytes())
    mask = getattr(data, 'mask', None)
    digest.update(b'mask')
    if mask is None:
        digest.update(b'|none')
    else:
        digest.update(b'|')
        digest.update(np.ascontiguousarray(np.asarray(mask), dtype=bool).tobytes())
    return digest.hexdigest()


def describe_processes(data: Any) -> list[str]:
    """Human-readable provenance lines from a dataset's sasdata Process records."""
    lines: list[str] = []
    for process in getattr(data, 'process', None) or []:
        name = str(getattr(process, 'name', '') or '').strip()
        description = str(getattr(process, 'description', '') or '').strip()
        if name and description:
            lines.append(f'{name}: {description}')
        elif name or description:
            lines.append(name or description)
    return lines


@dataclass(slots=True)
class DataSource:
    """Provenance of the dataset a fitter currently holds."""

    kind: str  # 'file' or 'memory'
    n_points: int
    array_fingerprint: str
    path: str | None = None  # absolute, file only
    dataset_requested: int | str | None = None
    dataset_index: int | None = None
    n_datasets: int | None = None
    file_sha256: str | None = None  # of the file as it was at load time
    label: str = ''
    processes: list[str] = field(default_factory=list)

    @classmethod
    def from_file(
        cls,
        filename: str,
        data: Any,
        *,
        requested: int | str,
        index: int,
        n_datasets: int,
    ) -> 'DataSource':
        path = os.path.abspath(filename)
        return cls(
            kind='file',
            n_points=int(np.asarray(data.x).size),
            array_fingerprint=fingerprint_arrays(data),
            path=path,
            dataset_requested=requested,
            dataset_index=index,
            n_datasets=n_datasets,
            file_sha256=_hash_file(path),
            label=os.path.basename(path),
            processes=describe_processes(data),
        )

    @classmethod
    def from_memory(cls, data: Any) -> 'DataSource':
        label = (
            str(getattr(data, 'title', '') or '')
            or str(getattr(data, 'filename', '') or '')
            or 'in-memory dataset'
        )
        return cls(
            kind='memory',
            n_points=int(np.asarray(data.x).size),
            array_fingerprint=fingerprint_arrays(data),
            label=label,
            processes=describe_processes(data),
        )

    def current_file_sha256(self) -> str | None:
        """Re-hash the source file now, or None when there is no readable file."""
        return None if self.path is None else _hash_file(self.path)

    def describe(self) -> str:
        """One line naming the source, for messages and report headers."""
        if self.kind == 'file':
            text = self.path or self.label
            if self.n_datasets and self.n_datasets > 1:
                text += f' (dataset {self.dataset_index})'
            return text
        if self.processes:
            return f'{self.label} [{"; ".join(self.processes)}]'
        return self.label
