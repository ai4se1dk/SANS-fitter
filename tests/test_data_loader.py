import os
import unittest

import numpy as np

from sans_fitter import SANSFitter
from tests.helpers import create_loading_test_data_file


class TestDataLoading(unittest.TestCase):
    """Test data loading functionality."""

    def setUp(self):
        self.fitter = SANSFitter()

    def test_load_data_success(self):
        data_file = create_loading_test_data_file()
        try:
            self.fitter.load_data(data_file)
            self.assertIsNotNone(self.fitter.data)
            self.assertTrue(hasattr(self.fitter.data, 'x'))
            self.assertTrue(hasattr(self.fitter.data, 'y'))
            self.assertTrue(hasattr(self.fitter.data, 'dy'))
            self.assertIsNotNone(self.fitter.data.qmin)
            self.assertIsNotNone(self.fitter.data.qmax)
        finally:
            os.unlink(data_file)

    def test_load_data_file_not_found(self):
        with self.assertRaises(ValueError):
            self.fitter.load_data('nonexistent_file.csv')

    def test_load_data_sets_mask(self):
        data_file = create_loading_test_data_file()
        try:
            self.fitter.load_data(data_file)
            self.assertTrue(hasattr(self.fitter.data, 'mask'))
            self.assertIsInstance(self.fitter.data.mask, np.ndarray)
        finally:
            os.unlink(data_file)


if __name__ == '__main__':
    unittest.main(verbosity=2)
