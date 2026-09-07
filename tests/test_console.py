"""Progress reporting goes through the package logger, not bare ``print``.

Two properties are pinned here: the messages a call emits as a side effect can
be silenced (``set_verbosity``), and the glyphs those messages carry never
reach a console that cannot encode them.
"""

import io
import logging
import sys

import pytest

import sans_fitter
from sans_fitter import SANSFitter, console


@pytest.fixture(autouse=True)
def restore_verbosity():
    """Keep a test's verbosity change from leaking into the rest of the suite."""
    previous = console.logger.level
    yield
    console.logger.setLevel(previous)


class _FakeStream:
    """Minimal stdout stand-in: ``_encodable`` only reads ``encoding``."""

    def __init__(self, encoding):
        self.encoding = encoding


class TestVerbosity:
    def test_progress_is_reported_by_default(self, capsys):
        SANSFitter().set_model('sphere')
        assert 'sphere' in capsys.readouterr().out

    def test_quiet_suppresses_progress(self, capsys):
        sans_fitter.set_verbosity('quiet')
        SANSFitter().set_model('sphere')
        assert capsys.readouterr().out == ''

    def test_verbosity_can_be_turned_back_on(self, capsys):
        sans_fitter.set_verbosity('quiet')
        sans_fitter.set_verbosity('info')
        SANSFitter().set_model('sphere')
        assert 'sphere' in capsys.readouterr().out

    def test_quiet_keeps_warnings(self, capsys):
        sans_fitter.set_verbosity('quiet')
        console.logger.warning('kept')
        assert 'kept' in capsys.readouterr().out

    def test_critical_silences_warnings_too(self, capsys):
        sans_fitter.set_verbosity(logging.CRITICAL)
        console.logger.warning('dropped')
        assert capsys.readouterr().out == ''

    def test_integer_levels_are_accepted(self):
        sans_fitter.set_verbosity(logging.DEBUG)
        assert console.logger.level == logging.DEBUG

    def test_unknown_level_is_rejected(self):
        with pytest.raises(ValueError, match='Unknown verbosity'):
            sans_fitter.set_verbosity('loud')


class TestLoggerWiring:
    def test_messages_do_not_duplicate_into_the_root_logger(self, capsys):
        root_output = io.StringIO()
        root_handler = logging.StreamHandler(root_output)
        logging.getLogger().addHandler(root_handler)
        try:
            console.logger.info('once')
        finally:
            logging.getLogger().removeHandler(root_handler)

        assert root_output.getvalue() == ''
        assert capsys.readouterr().out == 'once\n'

    def test_output_follows_stdout_redirection(self):
        """The handler must resolve sys.stdout per emit, not bind it at import."""
        import contextlib

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            console.logger.info('redirected')
        assert buffer.getvalue() == 'redirected\n'


class TestGlyphFallback:
    """cp1252 (the legacy Windows console) cannot encode the preferred glyphs."""

    def test_glyph_falls_back_when_stdout_cannot_encode_it(self, monkeypatch):
        monkeypatch.setattr(sys, 'stdout', _FakeStream('cp1252'))
        assert console._glyph('✓', '[ok]') == '[ok]'

    def test_glyph_is_kept_when_stdout_can_encode_it(self, monkeypatch):
        monkeypatch.setattr(sys, 'stdout', _FakeStream('utf-8'))
        assert console._glyph('✓', '[ok]') == '✓'

    def test_unknown_encoding_falls_back(self, monkeypatch):
        monkeypatch.setattr(sys, 'stdout', _FakeStream('not-a-real-codec'))
        assert console._glyph('✓', '[ok]') == '[ok]'

    @pytest.mark.parametrize('glyph', ['✓', '✗', '→', 'Å⁻¹', 'χ²'])
    def test_every_preferred_glyph_is_covered_by_a_fallback(self, glyph, monkeypatch):
        monkeypatch.setattr(sys, 'stdout', _FakeStream('cp1252'))
        assert not console._encodable(glyph)

    def test_selected_glyphs_are_printable_on_this_console(self):
        for glyph in (
            console.OK,
            console.YES,
            console.NO,
            console.ARROW,
            console.INVERSE_ANGSTROM,
            console.CHI_SQUARED,
        ):
            assert console._encodable(glyph), glyph

    def test_status_messages_use_the_selected_glyphs(self, capsys):
        SANSFitter().set_model('sphere')
        assert capsys.readouterr().out.startswith(console.OK)

    def test_parameter_table_uses_the_selected_glyphs(self, capsys):
        fitter = SANSFitter()
        fitter.set_model('sphere')
        fitter.set_param('radius', vary=True)
        capsys.readouterr()

        fitter.get_params()
        table = capsys.readouterr().out
        assert console.YES in table
        assert console.NO in table
