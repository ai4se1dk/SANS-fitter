"""Writing files without destroying the ones already there.

Every output this package produces — a saved analysis, a report, an export
directory — replaces something a user may already be relying on. Writing
straight to the final path means a failure part-way leaves a truncated file
where a good one used to be, which is worse than not having written at all.

All of them therefore render fully in memory, write a temporary sibling, and
rename it into place. The rename is atomic on every platform the package
supports, so the target is either the old file or the new one and never a
half-written mixture.
"""

import os

__all__ = ['atomic_write', 'atomic_write_bytes']


def atomic_write(target: str, text: str) -> None:
    """Replace *target* with *text*, or leave it exactly as it was.

    Newlines are written as ``\\n`` on every platform, and the encoding is
    always UTF-8: these files are read by other tools and checked into version
    control, so they must not depend on the locale of the machine that wrote
    them.
    """
    atomic_write_bytes(target, text.encode('utf-8'))


def atomic_write_bytes(target: str, payload: bytes) -> None:
    """Byte-level counterpart of :func:`atomic_write`."""
    temporary = f'{target}.tmp-{os.getpid()}'
    try:
        with open(temporary, 'wb') as handle:
            handle.write(payload)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise
