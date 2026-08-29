"""Console output: the package logger and encoding-safe glyphs.

Status messages that are a *side effect* of a call — data loaded, model set,
fit finished — go through ``logging.getLogger('sans_fitter')`` instead of
``print``, so a caller can turn them off::

    import sans_fitter
    sans_fitter.set_verbosity('quiet')

The logger owns a stdout handler and does not propagate, so by default the
output is byte-for-byte what the ``print`` calls produced and it does not
duplicate into an application's own logging configuration. To hand it over to
that configuration instead::

    import logging
    log = logging.getLogger('sans_fitter')
    log.handlers.clear()
    log.propagate = True

Tables the user asks for explicitly (``display_params``, ``display_pd_params``,
``examples.describe``) still ``print``: they *are* the result of the call
rather than a side effect of it, so routing them through the logger would
leave those methods doing nothing when the package is silenced.

The glyph constants fall back to ASCII when the active stdout cannot encode
them. A legacy Windows console (cp1252) has no ``✓``, ``χ``, ``→`` or ``⁻``,
and printing them there raises ``UnicodeEncodeError``.
"""

import logging
import sys
from typing import Any

LOGGER_NAME = 'sans_fitter'

logger = logging.getLogger(LOGGER_NAME)


class _StdoutHandler(logging.StreamHandler):
    """A ``StreamHandler`` that resolves ``sys.stdout`` at emit time.

    ``StreamHandler`` binds its stream at construction, which would miss every
    later redirection: the Jupyter kernel's streams, pytest's capture,
    ``contextlib.redirect_stdout``. Resolving per emit keeps the logger
    behaving like the ``print`` calls it replaced.
    """

    @property
    def stream(self) -> Any:
        return sys.stdout

    @stream.setter
    def stream(self, value: Any) -> None:
        """Ignore the stream ``StreamHandler.__init__`` tries to bind."""


def _install_default_handler() -> None:
    handler = _StdoutHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


if not logger.handlers:
    _install_default_handler()


_LEVEL_ALIASES = {'quiet': logging.WARNING}


def set_verbosity(level: int | str = 'info') -> None:
    """Set how much SANS-fitter reports about what it is doing.

    Args:
        level: ``'quiet'`` to drop the progress messages and keep only
            warnings, or any standard logging level as a name (``'info'``,
            ``'debug'``, ``'warning'``, ``'error'``) or as an integer. Pass
            ``logging.CRITICAL`` to suppress the warnings as well.

    Raises:
        ValueError: If *level* is not a recognized level name.
    """
    if isinstance(level, str):
        resolved = _LEVEL_ALIASES.get(level.lower(), logging.getLevelName(level.upper()))
        if not isinstance(resolved, int):
            raise ValueError(
                f"Unknown verbosity '{level}'. Use 'quiet', 'info', 'debug', "
                "'warning', 'error', or a logging level number."
            )
        level = resolved
    logger.setLevel(level)


def _encodable(text: str) -> bool:
    """Return True when the current stdout can encode *text*."""
    encoding = getattr(sys.stdout, 'encoding', None) or 'utf-8'
    try:
        text.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def _glyph(preferred: str, fallback: str) -> str:
    return preferred if _encodable(preferred) else fallback


#: Prefix marking a completed action in a status message.
OK = _glyph('✓', '[ok]')
#: Table cell for a varying parameter.
YES = _glyph('✓', 'yes')
#: Table cell for a fixed parameter.
NO = _glyph('✗', 'no')
#: Points a follower parameter at its link target.
ARROW = _glyph('→', '->')
#: Unit of Q.
INVERSE_ANGSTROM = _glyph('Å⁻¹', '1/Ang')
#: Goodness-of-fit symbol.
CHI_SQUARED = _glyph('χ²', 'chi^2')
