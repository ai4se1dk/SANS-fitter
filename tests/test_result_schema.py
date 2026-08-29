"""The fit_result parameter schema is the same for every engine (issue #30).

Both engines must report one entry per model parameter with the same fields,
so code written against one keeps working against the other. Before this,
bumps reported only the parameters it varied while scipy reported varied and
fixed ones, and neither said which was which.
"""

import contextlib
import io
import os
import tempfile
import unittest

from sans_fitter import SANSFitter
from tests.helpers import create_decay_data_file

ENTRY_FIELDS = {'value', 'stderr', 'formatted', 'fixed', 'linked_to'}


def configure(data_file, **structure_factor):
    """A sphere fit with two varied and three fixed parameters."""
    fitter = SANSFitter()
    with contextlib.redirect_stdout(io.StringIO()):
        fitter.load_data(data_file)
        fitter.set_model('sphere')
        fitter.set_param('radius', value=20.0, min=10.0, max=30.0, vary=True)
        fitter.set_param('scale', value=0.1, min=0.01, max=1.0, vary=True)
        fitter.set_param('background', value=0.01, vary=False)
        fitter.set_param('sld', value=2.0, vary=False)
        fitter.set_param('sld_solvent', value=3.0, vary=False)
        if structure_factor:
            fitter.set_structure_factor(**structure_factor)
    return fitter


def run_fit(fitter, engine, method):
    with contextlib.redirect_stdout(io.StringIO()):
        return fitter.fit(engine=engine, method=method)


class TestSchemaMatchesAcrossEngines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_file = create_decay_data_file(num_points=20)
        cls.results = {
            'bumps': run_fit(configure(cls.data_file), 'bumps', 'amoeba'),
            'lmfit': run_fit(configure(cls.data_file), 'lmfit', 'leastsq'),
        }

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.data_file):
            os.unlink(cls.data_file)

    def test_both_engines_report_the_same_parameters(self):
        self.assertEqual(
            set(self.results['bumps']['parameters']),
            set(self.results['lmfit']['parameters']),
        )

    def test_every_model_parameter_is_reported(self):
        expected = {'radius', 'scale', 'background', 'sld', 'sld_solvent'}
        for engine, result in self.results.items():
            self.assertEqual(set(result['parameters']), expected, engine)

    def test_every_entry_carries_the_same_fields(self):
        for engine, result in self.results.items():
            for name, info in result['parameters'].items():
                self.assertEqual(set(info), ENTRY_FIELDS, f'{engine}/{name}')

    def test_both_engines_agree_on_which_parameters_varied(self):
        varied = {
            engine: {name for name, info in result['parameters'].items() if not info['fixed']}
            for engine, result in self.results.items()
        }
        self.assertEqual(varied['bumps'], {'radius', 'scale'})
        self.assertEqual(varied['bumps'], varied['lmfit'])

    def test_fixed_parameters_keep_their_configured_value(self):
        for engine, result in self.results.items():
            self.assertEqual(result['parameters']['sld']['value'], 2.0, engine)
            self.assertTrue(result['parameters']['sld']['fixed'], engine)
            self.assertIn('fixed', result['parameters']['sld']['formatted'])

    def test_unlinked_parameters_report_no_link(self):
        for engine, result in self.results.items():
            for name, info in result['parameters'].items():
                self.assertIsNone(info['linked_to'], f'{engine}/{name}')

    def test_fixed_parameters_are_exported(self):
        for engine, method in (('bumps', 'amoeba'), ('lmfit', 'leastsq')):
            fitter = configure(self.data_file)
            with contextlib.redirect_stdout(io.StringIO()), tempfile.TemporaryDirectory() as tmpdir:
                fitter.fit(engine=engine, method=method)
                path = os.path.join(tmpdir, 'results.csv')
                fitter.save_results(path)
                with open(path) as handle:
                    content = handle.read()
            self.assertIn('# sld:', content, engine)
            self.assertIn('# radius:', content, engine)


class TestLinkedParametersInResults(unittest.TestCase):
    """A follower must report its target's *fitted* value, not the pre-fit one."""

    def setUp(self):
        self.data_file = create_decay_data_file(num_points=20)

    def tearDown(self):
        if os.path.exists(self.data_file):
            os.unlink(self.data_file)

    def _linked_fitter(self):
        fitter = SANSFitter()
        with contextlib.redirect_stdout(io.StringIO()):
            fitter.load_data(self.data_file)
            fitter.set_models(small='sphere', big='sphere')
            fitter.link_params('big_radius', 'small_radius')
            fitter.set_param('small_radius', value=20.0, min=10.0, max=30.0, vary=True)
        return fitter

    def test_follower_tracks_the_fitted_target(self):
        fitter = self._linked_fitter()
        params = run_fit(fitter, 'bumps', 'amoeba')['parameters']
        self.assertFalse(params['small_radius']['fixed'])
        self.assertTrue(params['big_radius']['fixed'])
        self.assertEqual(params['big_radius']['value'], params['small_radius']['value'])

    def test_follower_names_its_target_in_user_facing_names(self):
        fitter = self._linked_fitter()
        params = run_fit(fitter, 'bumps', 'amoeba')['parameters']
        self.assertEqual(params['big_radius']['linked_to'], 'small_radius')
        self.assertIsNone(params['small_radius']['linked_to'])
        # The canonical A_/B_ spelling must never reach a user-facing string.
        for info in params.values():
            self.assertNotIn('A_', info['formatted'])
            self.assertNotIn('B_', info['formatted'])

    def test_radius_effective_tracks_radius_under_link_radius(self):
        fitter = configure(
            self.data_file,
            structure_factor_name='hardsphere',
            radius_effective_mode='link_radius',
        )
        with contextlib.redirect_stdout(io.StringIO()):
            fitter.set_param('volfraction', value=0.2, min=0.0, max=0.6, vary=True)
        params = run_fit(fitter, 'bumps', 'amoeba')['parameters']
        self.assertEqual(params['radius_effective']['linked_to'], 'radius')
        self.assertTrue(params['radius_effective']['fixed'])
        self.assertEqual(params['radius_effective']['value'], params['radius']['value'])


if __name__ == '__main__':
    unittest.main()
