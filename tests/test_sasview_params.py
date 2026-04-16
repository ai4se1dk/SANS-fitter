"""Unit tests for the SasView parameter file parser and fitter import."""

import math
import os
import tempfile
import unittest
import warnings
from pathlib import Path

from sans_fitter import SANSFitter
from sans_fitter.sasview_params import (
    SasViewParam,
    SasViewParamFile,
    parse_sasview_params,
)

# Resolve paths relative to the tests directory
_SASVIEW_PARAMS_TXT = Path(__file__).resolve().parent / 'fixtures' / 'sasview_params.txt'
_FITTED_FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'sasview_params_fitted.txt'


class TestParseSasviewParamsTxt(unittest.TestCase):
    """Parse the repo-level sasview_params.txt (capped_cylinder, no fit)."""

    def test_parse_succeeds(self):
        result = parse_sasview_params(_SASVIEW_PARAMS_TXT)
        self.assertIsInstance(result, SasViewParamFile)

    def test_model_name(self):
        result = parse_sasview_params(_SASVIEW_PARAMS_TXT)
        self.assertEqual(result.model_name, 'capped_cylinder')

    def test_expected_param_count(self):
        """Separator row should be skipped, leaving 7 real params."""
        result = parse_sasview_params(_SASVIEW_PARAMS_TXT)
        self.assertEqual(len(result.params), 7)

    def test_param_names(self):
        result = parse_sasview_params(_SASVIEW_PARAMS_TXT)
        names = [p.name for p in result.params]
        self.assertIn('scale', names)
        self.assertIn('radius', names)
        self.assertIn('radius_cap', names)
        self.assertIn('length', names)

    def test_none_stderr_parsed_as_none(self):
        """stderr='None' in file should become Python None."""
        result = parse_sasview_params(_SASVIEW_PARAMS_TXT)
        for p in result.params:
            self.assertIsNone(p.stderr)

    def test_inf_bounds(self):
        """background has min=-inf, max=inf."""
        result = parse_sasview_params(_SASVIEW_PARAMS_TXT)
        bg = next(p for p in result.params if p.name == 'background')
        self.assertEqual(bg.min, float('-inf'))
        self.assertEqual(bg.max, float('inf'))

    def test_finite_bounds(self):
        """radius has min=0.0, max=inf."""
        result = parse_sasview_params(_SASVIEW_PARAMS_TXT)
        r = next(p for p in result.params if p.name == 'radius')
        self.assertEqual(r.min, 0.0)
        self.assertEqual(r.max, float('inf'))


class TestParseFittedFixture(unittest.TestCase):
    """Parse the fitted barbell fixture (has numeric stderr, vary=True rows)."""

    def test_parse_succeeds(self):
        result = parse_sasview_params(_FITTED_FIXTURE)
        self.assertIsInstance(result, SasViewParamFile)

    def test_model_name(self):
        result = parse_sasview_params(_FITTED_FIXTURE)
        self.assertEqual(result.model_name, 'barbell')

    def test_expected_param_count(self):
        result = parse_sasview_params(_FITTED_FIXTURE)
        self.assertEqual(len(result.params), 7)

    def test_empty_stderr_parsed_as_none(self):
        """scale row has empty stderr -> None."""
        result = parse_sasview_params(_FITTED_FIXTURE)
        scale = next(p for p in result.params if p.name == 'scale')
        self.assertIsNone(scale.stderr)

    def test_numeric_stderr(self):
        """radius row has stderr=0.61597."""
        result = parse_sasview_params(_FITTED_FIXTURE)
        r = next(p for p in result.params if p.name == 'radius')
        self.assertIsNotNone(r.stderr)
        self.assertAlmostEqual(r.stderr, 0.61597, places=4)

    def test_vary_true(self):
        result = parse_sasview_params(_FITTED_FIXTURE)
        r = next(p for p in result.params if p.name == 'radius')
        self.assertTrue(r.vary)

    def test_vary_false(self):
        result = parse_sasview_params(_FITTED_FIXTURE)
        scale = next(p for p in result.params if p.name == 'scale')
        self.assertFalse(scale.vary)

    def test_empty_bounds_parsed_as_none_for_no_field(self):
        """Bounds columns are present as -inf/inf/0.0 in fitted fixture; check a known one."""
        result = parse_sasview_params(_FITTED_FIXTURE)
        sld = next(p for p in result.params if p.name == 'sld')
        self.assertEqual(sld.min, float('-inf'))
        self.assertEqual(sld.max, float('inf'))


class TestSeparatorRowSkipping(unittest.TestCase):
    """Verify that separator rows are excluded from parsed params."""

    def test_separator_not_in_params(self):
        result = parse_sasview_params(_SASVIEW_PARAMS_TXT)
        names = [p.name for p in result.params]
        self.assertNotIn('capped_cylinder', names)

    def test_separator_not_in_fitted_params(self):
        result = parse_sasview_params(_FITTED_FIXTURE)
        names = [p.name for p in result.params]
        self.assertNotIn('barbell', names)


class TestRejections(unittest.TestCase):
    """Rejection cases for malformed files."""

    def _write_tmp(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix='.txt')
        os.close(fd)
        Path(path).write_text(content, encoding='utf-8')
        return path

    def test_reject_empty_file(self):
        path = self._write_tmp('')
        try:
            with self.assertRaises(ValueError):
                parse_sasview_params(path)
        finally:
            os.unlink(path)

    def test_reject_wrong_magic_header(self):
        path = self._write_tmp('wrong_header\nmodel_name,sphere\n')
        try:
            with self.assertRaises(ValueError):
                parse_sasview_params(path)
        finally:
            os.unlink(path)

    def test_reject_missing_model_name_row(self):
        path = self._write_tmp('sasview_parameter_values\n')
        try:
            with self.assertRaises(ValueError):
                parse_sasview_params(path)
        finally:
            os.unlink(path)

    def test_reject_bad_model_name_key(self):
        path = self._write_tmp('sasview_parameter_values\nwrong_key,sphere\n')
        try:
            with self.assertRaises(ValueError):
                parse_sasview_params(path)
        finally:
            os.unlink(path)

    def test_reject_malformed_value(self):
        content = (
            'sasview_parameter_values\nmodel_name,sphere\nradius,True,not_a_number,,0.0,inf,()\n'
        )
        path = self._write_tmp(content)
        try:
            with self.assertRaises(ValueError):
                parse_sasview_params(path)
        finally:
            os.unlink(path)

    def test_reject_unexpected_vary(self):
        content = 'sasview_parameter_values\nmodel_name,sphere\nradius,Maybe,20.0,,0.0,inf,()\n'
        path = self._write_tmp(content)
        try:
            with self.assertRaises(ValueError):
                parse_sasview_params(path)
        finally:
            os.unlink(path)


# =========================================================================
# Fitter integration tests
# =========================================================================


class TestLoadSasviewParamsIntegration(unittest.TestCase):
    """Round-trip tests: file -> SANSFitter.load_sasview_params()."""

    def _write_tmp(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix='.txt')
        os.close(fd)
        Path(path).write_text(content, encoding='utf-8')
        return path

    def test_load_capped_cylinder(self):
        """Plain form-factor file loads successfully and sets parameters."""
        fitter = SANSFitter()
        result = fitter.load_sasview_params(_SASVIEW_PARAMS_TXT)

        self.assertEqual(fitter.model_name, 'capped_cylinder')
        self.assertAlmostEqual(fitter.params['radius']['value'], 20.0)
        self.assertAlmostEqual(fitter.params['length']['value'], 400.0)
        self.assertIsInstance(result, SasViewParamFile)

    def test_load_fitted_barbell(self):
        """Fitted file with vary=True and stderr loads correctly."""
        fitter = SANSFitter()
        _ = fitter.load_sasview_params(_FITTED_FIXTURE)

        self.assertEqual(fitter.model_name, 'barbell')
        self.assertAlmostEqual(fitter.params['radius']['value'], 19.89)
        self.assertTrue(fitter.params['radius']['vary'])
        self.assertFalse(fitter.params['scale']['vary'])

    def test_stderr_not_applied_to_fitter(self):
        """stderr is in parsed result but not in fitter param state."""
        fitter = SANSFitter()
        result = fitter.load_sasview_params(_FITTED_FIXTURE)

        # The parsed result preserves stderr
        r_parsed = next(p for p in result.params if p.name == 'radius')
        self.assertAlmostEqual(r_parsed.stderr, 0.61597, places=4)

        # The fitter param dict should not have a stderr key
        self.assertNotIn('stderr', fitter.params['radius'])

    def test_bounds_applied(self):
        """min/max from the file are applied to fitter parameters."""
        fitter = SANSFitter()
        fitter.load_sasview_params(_SASVIEW_PARAMS_TXT)

        self.assertEqual(fitter.params['radius']['min'], 0.0)
        self.assertEqual(fitter.params['radius']['max'], float('inf'))

    def test_reject_product_model(self):
        """Product models (form@structure) raise NotImplementedError."""
        content = (
            'sasview_parameter_values\n'
            'model_name,sphere@hardsphere\n'
            'scale,False,1.0,None,0.0,inf,()\n'
        )
        path = self._write_tmp(content)
        try:
            fitter = SANSFitter()
            with self.assertRaises(NotImplementedError):
                fitter.load_sasview_params(path)
        finally:
            os.unlink(path)

    def test_skip_pd_params_with_warning(self):
        """PD-specific rows are skipped with a warning."""
        content = (
            'sasview_parameter_values\n'
            'model_name,sphere\n'
            'scale,False,1.0,None,0.0,inf,()\n'
            'background,False,0.001,None,-inf,inf,()\n'
            'sld,False,1.0,None,-inf,inf,()\n'
            'sld_solvent,False,1.0,None,-inf,inf,()\n'
            'radius,True,50.0,None,0.0,inf,()\n'
            'radius_pd,True,0.1,None,0.0,inf,()\n'
            'radius_pd_n,False,35,None,0,inf,()\n'
            'radius_pd_nsigma,False,3,None,0,inf,()\n'
            'radius_pd_type,False,1,None,0,inf,()\n'
        )
        path = self._write_tmp(content)
        try:
            fitter = SANSFitter()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                _ = fitter.load_sasview_params(path)
                self.assertTrue(len(w) >= 1)
                self.assertIn('radius_pd', str(w[-1].message))
            # The non-PD params should still be applied
            self.assertAlmostEqual(fitter.params['radius']['value'], 50.0)
        finally:
            os.unlink(path)

    def test_skip_unknown_param_with_warning(self):
        """Unknown parameter names (not in model) are skipped with a warning."""
        content = (
            'sasview_parameter_values\n'
            'model_name,sphere\n'
            'scale,False,1.0,None,0.0,inf,()\n'
            'background,False,0.001,None,-inf,inf,()\n'
            'sld,False,1.0,None,-inf,inf,()\n'
            'sld_solvent,False,1.0,None,-inf,inf,()\n'
            'radius,True,50.0,None,0.0,inf,()\n'
            'fake_param,True,99.0,None,0.0,inf,()\n'
        )
        path = self._write_tmp(content)
        try:
            fitter = SANSFitter()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                fitter.load_sasview_params(path)
                self.assertTrue(len(w) >= 1)
                self.assertIn('fake_param', str(w[-1].message))
            self.assertAlmostEqual(fitter.params['radius']['value'], 50.0)
        finally:
            os.unlink(path)

    def test_return_value_is_full_parsed_file(self):
        """The returned SasViewParamFile contains all parsed params including skipped."""
        fitter = SANSFitter()
        result = fitter.load_sasview_params(_SASVIEW_PARAMS_TXT)
        self.assertEqual(result.model_name, 'capped_cylinder')
        self.assertEqual(len(result.params), 7)


if __name__ == '__main__':
    unittest.main()
