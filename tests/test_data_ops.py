import os
import unittest
import warnings

import numpy as np
from sasdata.dataloader.data_info import Data1D

from sans_fitter import SANSFitter, data_ops
from tests.helpers import (
    create_background_data_file,
    create_decay_data_file,
    create_loading_test_data_file,
    create_offset_grid_data_file,
)


def make_data1d(num_points=20, y_value=10.0, dy_value=1.0, dx_value=None):
    """Build an in-memory Data1D on a fixed positive Q grid."""
    x = np.linspace(0.01, 0.5, num_points)
    y = np.full(num_points, y_value)
    dy = None if dy_value is None else np.full(num_points, dy_value)
    dx = None if dx_value is None else np.full(num_points, dx_value)
    return Data1D(x=x, y=y, dy=dy, dx=dx)


class TestLoad(unittest.TestCase):
    """data_ops.load is the canonical standalone loader."""

    def setUp(self):
        self.data_file = create_loading_test_data_file()

    def tearDown(self):
        if os.path.exists(self.data_file):
            os.unlink(self.data_file)

    def test_load_returns_fit_ready_data1d(self):
        data = data_ops.load(self.data_file)
        self.assertIsInstance(data, Data1D)
        self.assertEqual(data.qmin, data.x.min())
        self.assertEqual(data.qmax, data.x.max())
        self.assertEqual(len(data.mask), len(data.y))

    def test_load_matches_fitter_load_data(self):
        data = data_ops.load(self.data_file)
        fitter = SANSFitter()
        fitter.load_data(self.data_file)
        np.testing.assert_array_equal(data.x, fitter.data.x)
        np.testing.assert_array_equal(data.y, fitter.data.y)
        np.testing.assert_array_equal(data.dy, fitter.data.dy)

    def test_load_missing_file_raises(self):
        with self.assertRaises(ValueError):
            data_ops.load('nonexistent_file_xyz.csv')


class TestDatasetArithmetic(unittest.TestCase):
    """Dataset ⊕ dataset operations with hand-computed expectations."""

    def setUp(self):
        self.sample_file = create_loading_test_data_file()
        self.background_file = create_background_data_file()
        self.sample = data_ops.load(self.sample_file)
        self.background = data_ops.load(self.background_file)

    def tearDown(self):
        for path in (self.sample_file, self.background_file):
            if os.path.exists(path):
                os.unlink(path)

    def test_subtract_values_and_errors(self):
        result = data_ops.subtract(self.sample, self.background)
        np.testing.assert_allclose(result.y, self.sample.y - self.background.y)
        expected_dy = np.sqrt(self.sample.dy**2 + self.background.dy**2)
        np.testing.assert_allclose(result.dy, expected_dy)

    def test_add_values_and_errors(self):
        result = data_ops.add(self.sample, self.background)
        np.testing.assert_allclose(result.y, self.sample.y + self.background.y)
        expected_dy = np.sqrt(self.sample.dy**2 + self.background.dy**2)
        np.testing.assert_allclose(result.dy, expected_dy)

    def test_multiply_values_and_errors(self):
        result = data_ops.multiply(self.sample, self.background)
        np.testing.assert_allclose(result.y, self.sample.y * self.background.y)
        expected_dy = np.abs(result.y) * np.sqrt(
            (self.sample.dy / self.sample.y) ** 2 + (self.background.dy / self.background.y) ** 2
        )
        np.testing.assert_allclose(result.dy, expected_dy)

    def test_divide_values_and_errors(self):
        result = data_ops.divide(self.sample, self.background)
        np.testing.assert_allclose(result.y, self.sample.y / self.background.y)
        expected_dy = np.abs(result.y) * np.sqrt(
            (self.sample.dy / self.sample.y) ** 2 + (self.background.dy / self.background.y) ** 2
        )
        np.testing.assert_allclose(result.dy, expected_dy)

    def test_subtract_is_not_commutative(self):
        forward = data_ops.subtract(self.sample, self.background)
        reverse = data_ops.subtract(self.background, self.sample)
        np.testing.assert_allclose(forward.y, self.sample.y - self.background.y)
        np.testing.assert_allclose(reverse.y, self.background.y - self.sample.y)
        self.assertFalse(np.allclose(forward.y, reverse.y))

    def test_q_grid_preserved(self):
        result = data_ops.subtract(self.sample, self.background)
        np.testing.assert_array_equal(result.x, self.sample.x)


class TestScalarArithmetic(unittest.TestCase):
    """Dataset ⊕ scalar semantics."""

    def setUp(self):
        self.data = make_data1d(y_value=10.0, dy_value=1.0)

    def test_multiply_by_scalar(self):
        result = data_ops.multiply(self.data, 2.0)
        np.testing.assert_allclose(result.y, 20.0)
        np.testing.assert_allclose(result.dy, 2.0)
        np.testing.assert_array_equal(result.x, self.data.x)

    def test_divide_by_scalar(self):
        result = data_ops.divide(self.data, 2.0)
        np.testing.assert_allclose(result.y, 5.0)
        np.testing.assert_allclose(result.dy, 0.5)
        np.testing.assert_array_equal(result.x, self.data.x)

    def test_subtract_scalar_keeps_dy(self):
        result = data_ops.subtract(self.data, 0.05)
        np.testing.assert_allclose(result.y, 9.95)
        np.testing.assert_allclose(result.dy, 1.0)
        np.testing.assert_array_equal(result.x, self.data.x)

    def test_add_scalar(self):
        result = data_ops.add(self.data, 0.5)
        np.testing.assert_allclose(result.y, 10.5)
        np.testing.assert_allclose(result.dy, 1.0)


class TestFitReadiness(unittest.TestCase):
    """Arithmetic results can be fed straight into SANSFitter."""

    def setUp(self):
        self.data_file = create_decay_data_file()

    def tearDown(self):
        if os.path.exists(self.data_file):
            os.unlink(self.data_file)

    def test_result_has_fit_attributes(self):
        data = data_ops.load(self.data_file)
        result = data_ops.subtract(data, 0.005)
        self.assertEqual(result.qmin, result.x.min())
        self.assertEqual(result.qmax, result.x.max())
        self.assertEqual(len(result.mask), len(result.y))
        self.assertFalse(result.mask.any())

    def test_end_to_end_set_data_and_fit(self):
        data = data_ops.load(self.data_file)
        result = data_ops.subtract(data, 0.005)

        fitter = SANSFitter()
        fitter.set_data(result)
        fitter.set_model('sphere')
        fitter.set_param('radius', value=20.0, min=10.0, max=30.0, vary=True)
        fitter.set_param('scale', value=0.1, min=0.01, max=1.0, vary=True)
        fitter.set_param('background', value=0.005, min=0, max=0.1, vary=True)
        fitter.set_param('sld', value=2.0, vary=False)
        fitter.set_param('sld_solvent', value=3.0, vary=False)

        fit_result = fitter.fit(engine='lmfit', method='leastsq')
        self.assertTrue(np.isfinite(fit_result['chisq']))


class TestOperandValidation(unittest.TestCase):
    """Rejection of unusable operands with actionable messages."""

    def setUp(self):
        self.sample_file = create_loading_test_data_file()
        self.offset_file = create_offset_grid_data_file()

    def tearDown(self):
        for path in (self.sample_file, self.offset_file):
            if os.path.exists(path):
                os.unlink(path)

    def test_mismatched_q_grids_raise_with_diagnostics(self):
        sample = data_ops.load(self.sample_file)
        offset = data_ops.load(self.offset_file)
        with self.assertRaises(ValueError) as ctx:
            data_ops.subtract(sample, offset)
        message = str(ctx.exception)
        self.assertIn('Q grid', message)
        self.assertIn('10 points', message)
        self.assertIn('12 points', message)
        self.assertIn(os.path.basename(self.sample_file), message)
        self.assertIn(os.path.basename(self.offset_file), message)

    def test_2d_data_rejected(self):
        class Fake2D:
            qx_data = np.array([0.1, 0.2])

        with self.assertRaises(TypeError):
            data_ops.subtract(make_data1d(), Fake2D())

    def test_non_numeric_operand_rejected(self):
        with self.assertRaises(TypeError):
            data_ops.subtract(make_data1d(), 'not a dataset')

    def test_scalar_first_operand_rejected(self):
        with self.assertRaises(TypeError):
            data_ops.subtract(2.0, make_data1d())

    def test_empty_operand_rejected(self):
        empty = Data1D(x=np.array([]), y=np.array([]))
        with self.assertRaises(ValueError):
            data_ops.subtract(make_data1d(), empty)

    def test_missing_dy_warns(self):
        no_errors = make_data1d(dy_value=None)
        with self.assertWarnsRegex(UserWarning, 'no intensity uncertainties'):
            data_ops.subtract(make_data1d(), no_errors)


class TestNaNPropagation(unittest.TestCase):
    """NaN points propagate through matching-grid operations and get masked."""

    def test_nan_in_second_operand_is_masked_and_warned(self):
        a = make_data1d()
        b = make_data1d(y_value=5.0, dy_value=0.5)
        b.y[3] = np.nan

        with self.assertWarnsRegex(UserWarning, 'masked'):
            result = data_ops.subtract(a, b)

        self.assertTrue(np.isnan(result.y[3]))
        self.assertTrue(result.mask[3])
        self.assertEqual(int(result.mask.sum()), 1)


class TestZeroErrorHandling(unittest.TestCase):
    """Zero/absent dI: warnings and engine guards."""

    def setUp(self):
        self.data_file = create_decay_data_file()

    def tearDown(self):
        if os.path.exists(self.data_file):
            os.unlink(self.data_file)

    def _error_free_fitter(self):
        data = data_ops.load(self.data_file)
        data.dy = np.zeros_like(data.y)
        fitter = SANSFitter()
        fitter.set_data(data)
        fitter.set_model('sphere')
        fitter.set_param('radius', value=20.0, min=10.0, max=30.0, vary=True)
        fitter.set_param('scale', value=0.1, min=0.01, max=1.0, vary=True)
        fitter.set_param('background', value=0.01, min=0, max=0.1, vary=True)
        fitter.set_param('sld', value=2.0, vary=False)
        fitter.set_param('sld_solvent', value=3.0, vary=False)
        return fitter

    def test_error_free_result_warns(self):
        a = make_data1d(dy_value=None)
        b = make_data1d(y_value=2.0, dy_value=None)
        with self.assertWarnsRegex(UserWarning, 'no intensity uncertainties'):
            result = data_ops.subtract(a, b)
        np.testing.assert_allclose(result.y, 8.0)

    def test_lmfit_engine_handles_zero_dy(self):
        fitter = self._error_free_fitter()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = fitter.fit(engine='lmfit', method='leastsq')
        self.assertTrue(np.isfinite(result['chisq']))

    def test_bumps_engine_rejects_zero_dy(self):
        fitter = self._error_free_fitter()
        with self.assertRaises(ValueError) as ctx:
            fitter.fit(engine='bumps')
        self.assertIn('bumps engine cannot weight', str(ctx.exception))


class TestSetData(unittest.TestCase):
    """SANSFitter.set_data validation."""

    def test_set_valid_data(self):
        fitter = SANSFitter()
        data = make_data1d()
        fitter.set_data(data)
        self.assertIs(fitter.data, data)
        self.assertEqual(fitter.data.qmin, data.x.min())
        self.assertEqual(fitter.data.qmax, data.x.max())
        self.assertEqual(fitter.get_q_range(), (data.x.min(), data.x.max()))

    def test_empty_data_rejected(self):
        fitter = SANSFitter()
        with self.assertRaises(ValueError):
            fitter.set_data(Data1D(x=np.array([]), y=np.array([])))

    def test_non_positive_q_rejected(self):
        fitter = SANSFitter()
        data = Data1D(x=np.array([-0.1, 0.1, 0.2]), y=np.ones(3))
        with self.assertRaises(ValueError):
            fitter.set_data(data)

    def test_missing_arrays_rejected(self):
        fitter = SANSFitter()
        with self.assertRaises(TypeError):
            fitter.set_data(object())

    def test_few_points_warns(self):
        fitter = SANSFitter()
        data = Data1D(x=np.array([0.1, 0.2, 0.3]), y=np.ones(3), dy=np.full(3, 0.1))
        with self.assertWarnsRegex(UserWarning, 'only 3 points'):
            fitter.set_data(data)


class TestMetadata(unittest.TestCase):
    """Provenance: title, filename and process history."""

    def setUp(self):
        self.sample_file = create_loading_test_data_file()
        self.background_file = create_background_data_file()

    def tearDown(self):
        for path in (self.sample_file, self.background_file):
            if os.path.exists(path):
                os.unlink(path)

    def test_title_and_process_record_operation(self):
        sample = data_ops.load(self.sample_file)
        background = data_ops.load(self.background_file)
        result = data_ops.subtract(sample, background)

        self.assertIn('-', result.title)
        self.assertIn(os.path.basename(self.sample_file), result.title)
        self.assertIn(os.path.basename(self.background_file), result.title)
        self.assertTrue(result.process)
        self.assertEqual(result.process[-1].name, 'sans_fitter.data_ops')
        self.assertIn('Dataset arithmetic', result.process[-1].description)

    def test_scalar_operation_recorded(self):
        sample = data_ops.load(self.sample_file)
        result = data_ops.multiply(sample, 2.0)
        self.assertIn('* 2', result.title)


class TestResolutionWarning(unittest.TestCase):
    """Resolution (dx) present on an operand triggers the caveat warning."""

    def test_dx_triggers_warning(self):
        a = make_data1d(dx_value=0.001)
        b = make_data1d(y_value=2.0, dy_value=0.2, dx_value=0.002)
        with self.assertWarnsRegex(UserWarning, 'Resolution'):
            data_ops.subtract(a, b)

    def test_no_dx_no_resolution_warning(self):
        a = make_data1d()
        b = make_data1d(y_value=2.0, dy_value=0.2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            data_ops.subtract(a, b)
        messages = [str(w.message) for w in caught]
        self.assertFalse(any('Resolution' in m for m in messages))


if __name__ == '__main__':
    unittest.main()
