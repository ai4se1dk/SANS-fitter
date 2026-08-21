"""Tests for P(r) inversion (issue #57).

Layered per the design: exact mathematics at machine precision, physics
round-trips on pinned sphere/cylinder fixtures (simulated with sasmodels,
handed to the module as sasdata Data1D objects), data-preparation edge cases,
estimator behaviour, D_max exploration and plots.
"""

import os
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from sasdata.dataloader.data_info import Data1D
from sasmodels.core import load_model
from sasmodels.data import empty_data1D
from sasmodels.direct_model import DirectModel

from sans_fitter import pr_inversion as pri
from sans_fitter.data.loader import get_fit_index, normalize_sans_data
from sans_fitter.inversion import basis as pr_basis
from sans_fitter.inversion import estimate as pr_estimate
from sans_fitter.inversion import solver as pr_solver
from sans_fitter.plotting import plot_dmax_scan, plot_pr_distribution, plot_pr_fit

SPHERE_RADIUS = 60.0
SPHERE_D_MAX = 2 * SPHERE_RADIUS
SPHERE_Q = np.geomspace(0.005, 0.25, 80)
SPHERE_PARS = {
    'radius': SPHERE_RADIUS,
    'scale': 1.0,
    'background': 0.0,
    'sld': 4.0,
    'sld_solvent': 1.0,
}
SPHERE_RG = np.sqrt(3.0 / 5.0) * SPHERE_RADIUS

CYLINDER_D_MAX = 402.0
CYLINDER_Q = np.geomspace(0.002, 0.2, 120)
CYLINDER_PARS = {
    'radius': 20.0,
    'length': 400.0,
    'scale': 1.0,
    'background': 0.0,
    'sld': 4.0,
    'sld_solvent': 1.0,
}
CYLINDER_RG = np.sqrt(20.0**2 / 2.0 + 400.0**2 / 12.0)

NOISE_LEVEL = 0.02

_KERNELS: dict[str, object] = {}


def _clean_intensity(model_name: str, pars: dict, q: np.ndarray) -> np.ndarray:
    """Evaluate a sasmodels kernel at q (sasmodels used only for evaluation)."""
    if model_name not in _KERNELS:
        _KERNELS[model_name] = load_model(model_name)
    model = DirectModel(empty_data1D(q), _KERNELS[model_name])
    return model(**pars)


def _true_i0(model_name: str, pars: dict) -> float:
    return float(_clean_intensity(model_name, pars, np.array([1e-5]))[0])


def make_synthetic_data(model_name, pars, q, seed=42, noise=NOISE_LEVEL, background=0.0):
    """Simulated dataset as a sasdata Data1D (the class users actually pass).

    Noise comes from a dedicated Generator so tests never mutate global NumPy
    RNG state.
    """
    i_clean = _clean_intensity(model_name, pars, q) + background
    rng = np.random.default_rng(seed)
    sigma = noise * np.abs(i_clean)
    data = Data1D(x=q, y=i_clean + rng.normal(0.0, sigma), dy=sigma)
    return normalize_sans_data(data)


def sphere_pr_oracle(r: np.ndarray) -> np.ndarray:
    """Analytic sphere distance distribution (Glatter), up to normalization."""
    x = r / SPHERE_RADIUS
    return np.where(r <= 2 * SPHERE_RADIUS, r**2 * (1 - 3 * x / 4 + x**3 / 16), 0.0)


def quiet_invert(*args, **kwargs):
    """invert() with warnings suppressed, for tests not asserting on them."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return pri.invert(*args, **kwargs)


class TestBasisFunctions(unittest.TestCase):
    """Exact mathematics of the basis and its Fourier transform."""

    def test_basis_vanishes_at_endpoints(self):
        for n in (1, 3, 12):
            values = pr_basis._ortho(120.0, n, np.array([0.0, 120.0]))
            np.testing.assert_allclose(values, 0.0, atol=1e-12)

    def test_transform_pole_limit(self):
        d_max = 120.0
        for n in (1, 2, 5, 13):
            pole_q = np.pi * n / d_max
            expected = 4.0 * d_max**2 / n
            at_pole = pr_basis._ortho_transformed(d_max, n, np.array([pole_q]))[0]
            self.assertAlmostEqual(at_pole, expected, places=8)
            for eps in (1e-6, -1e-6):
                near = pr_basis._ortho_transformed(d_max, n, np.array([pole_q * (1 + eps)]))[0]
                self.assertAlmostEqual(near / expected, 1.0, places=4)

    def test_transform_q_zero_limit(self):
        d_max = 120.0
        for n in (1, 2, 7):
            expected = 8.0 * d_max**2 * (-1.0) ** (n + 1) / n
            value = pr_basis._ortho_transformed(d_max, n, np.array([0.0]))[0]
            self.assertAlmostEqual(value, expected, places=8)

    def test_transform_vectorized_pole_mixture(self):
        d_max = 120.0
        n = 3
        pole_q = np.pi * n / d_max
        q = np.array([0.0, 0.01, pole_q, 0.11, pole_q * (1 + 5e-9)])
        values = pr_basis._ortho_transformed(d_max, n, q)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertAlmostEqual(values[2], 4.0 * d_max**2 / n, places=8)
        self.assertAlmostEqual(values[4], 4.0 * d_max**2 / n, places=6)

    def test_transform_matches_numerical_fourier_integral(self):
        d_max = 120.0
        r = np.linspace(0.0, d_max, 20001)
        for n, q in ((1, 0.03), (3, 0.11), (6, 0.2)):
            phi = pr_basis._ortho(d_max, n, r)
            integrand = 4.0 * np.pi * phi * np.sinc(q * r / np.pi)
            numeric = np.trapezoid(integrand, r)
            analytic = pr_basis._ortho_transformed(d_max, n, np.array([q]))[0]
            self.assertAlmostEqual(numeric / analytic, 1.0, places=6)

    def test_i0_identity_closed_form(self):
        """Phi~_n(0) = 4 pi * integral(Phi_n dr): exact per basis function."""
        d_max = 120.0
        r = np.linspace(0.0, d_max, 200001)
        for n in (1, 2, 5, 9):
            integral = 4.0 * np.pi * np.trapezoid(pr_basis._ortho(d_max, n, r), r)
            closed_form = 8.0 * d_max**2 * (-1.0) ** (n + 1) / n
            self.assertAlmostEqual(integral / closed_form, 1.0, places=6)


class TestExactSolve(unittest.TestCase):
    """Machine-precision behaviour of the solver on basis-generated data."""

    def test_basis_generated_round_trip(self):
        coefficients = np.array([0.5, -0.2, 0.1, 0.05])
        q = np.geomspace(0.004, 0.3, 50)
        intensity = np.zeros(q.size)
        for j, c in enumerate(coefficients):
            intensity += c * pr_basis._ortho_transformed(120.0, j + 1, q)
        data = normalize_sans_data(Data1D(x=q, y=intensity, dy=np.ones(q.size)))
        result = quiet_invert(data, 120.0, n_terms=4, alpha=0.0, fit_background=False)
        np.testing.assert_allclose(result.coefficients, coefficients, atol=1e-10)
        self.assertAlmostEqual(result.data_chisq, 0.0, places=10)

    def test_alpha_zero_reproduces_plain_lstsq(self):
        data = make_synthetic_data('sphere', SPHERE_PARS, SPHERE_Q, seed=3)
        prep = pr_solver._prepare_data(data)
        a_data, b_data, _ = pr_solver._build_blocks(
            prep.q, prep.intensity, prep.sigma, SPHERE_D_MAX, 8, True, 'corrected'
        )
        direct, *_ = np.linalg.lstsq(a_data, b_data, rcond=None)
        result = quiet_invert(data, SPHERE_D_MAX, n_terms=8, alpha=0.0)
        ours = np.concatenate([[result.background], result.coefficients])
        np.testing.assert_allclose(ours, direct, atol=1e-12)

    def test_rank_deficiency_surfaced_not_raised(self):
        """A narrow q window with many terms is genuinely rank-deficient."""
        q = np.linspace(0.001, 0.002, 30)
        data = normalize_sans_data(Data1D(x=q, y=np.full(30, 100.0), dy=np.ones(30)))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = pri.invert(data, 100.0, n_terms=20, alpha=0.0)
        self.assertLess(result.rank, result.n_terms + 1)
        self.assertTrue(np.all(np.isfinite(result.coefficients)))
        self.assertTrue(any('Rank-deficient' in str(w.message) for w in caught))


class TestSphereRoundTrip(unittest.TestCase):
    """Pinned sphere fixture: R = 60 A, D_max = 120 A, q in [0.005, 0.25]."""

    @classmethod
    def setUpClass(cls):
        cls.data = make_synthetic_data('sphere', SPHERE_PARS, SPHERE_Q, seed=42)
        cls.i0_true = _true_i0('sphere', SPHERE_PARS)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cls.estimate = pri.estimate_n_terms(cls.data, SPHERE_D_MAX)
            cls.result = pri.invert(
                cls.data,
                SPHERE_D_MAX,
                n_terms=cls.estimate.n_terms,
                alpha=cls.estimate.alpha,
            )

    def test_rg_within_two_percent(self):
        self.assertAlmostEqual(self.result.rg / SPHERE_RG, 1.0, delta=0.02)

    def test_i0_within_five_percent(self):
        self.assertAlmostEqual(self.result.i0 / self.i0_true, 1.0, delta=0.05)

    def test_pr_shape_correlation(self):
        oracle = sphere_pr_oracle(self.result.r)
        correlation = np.corrcoef(self.result.pr, oracle)[0, 1]
        self.assertGreater(correlation, 0.99)

    def test_positive_fraction_and_oscillations(self):
        self.assertGreaterEqual(self.result.positive_fraction, 0.95)
        self.assertAlmostEqual(self.result.oscillations, 1.1, delta=0.3)

    def test_result_i0_matches_pr_integral(self):
        r_fine = np.linspace(0.0, SPHERE_D_MAX, 20001)
        integral = 4.0 * np.pi * np.trapezoid(self.result.evaluate_pr(r_fine), r_fine)
        i0_closed = self.result.evaluate_iq(np.array([0.0]))[0] - self.result.background
        self.assertAlmostEqual(integral / i0_closed, 1.0, places=6)

    def test_evaluate_iq_matches_stored_fit(self):
        np.testing.assert_allclose(
            self.result.evaluate_iq(self.result.q_fit), self.result.iq_fit, rtol=1e-12
        )

    def test_fitted_background_recovers_truth(self):
        background_true = 0.05
        data = make_synthetic_data(
            'sphere', SPHERE_PARS, SPHERE_Q, seed=11, background=background_true
        )
        result = quiet_invert(
            data, SPHERE_D_MAX, n_terms=self.estimate.n_terms, alpha=self.estimate.alpha
        )
        self.assertTrue(result.background_fitted)
        self.assertTrue(np.isfinite(result.background_err))
        self.assertAlmostEqual(result.background, background_true, delta=5 * result.background_err)

    def test_fixed_background_matches_presubtracted_and_does_not_mutate(self):
        background = 0.1
        data = make_synthetic_data('sphere', SPHERE_PARS, SPHERE_Q, seed=11, background=background)
        y_before = np.asarray(data.y).copy()
        fixed = quiet_invert(
            data,
            SPHERE_D_MAX,
            n_terms=10,
            alpha=self.estimate.alpha,
            fit_background=False,
            background=background,
        )
        np.testing.assert_array_equal(np.asarray(data.y), y_before)

        subtracted = Data1D(x=data.x, y=np.asarray(data.y) - background, dy=data.dy)
        normalize_sans_data(subtracted)
        reference = quiet_invert(
            subtracted,
            SPHERE_D_MAX,
            n_terms=10,
            alpha=self.estimate.alpha,
            fit_background=False,
            background=0.0,
        )
        np.testing.assert_allclose(fixed.coefficients, reference.coefficients, rtol=1e-10)
        self.assertEqual(fixed.background, background)
        self.assertFalse(np.isfinite(fixed.background_err))

    def test_format_summary_contains_scalars(self):
        summary = self.result.format_summary()
        for label in (
            'D_max',
            'Rg',
            'I(0)',
            'Oscillations',
            'Positive fraction',
            'Data chi-squared',
            'Effective dof',
            'Condition number',
            'Approx. chi2 per residual dof',
        ):
            self.assertIn(label, summary)
        summary.encode('ascii')  # ASCII-safe by design

    def test_save_csv_round_trip(self):
        with tempfile.NamedTemporaryFile(mode='r', suffix='.csv', delete=False) as f:
            filename = f.name
        self.addCleanup(os.unlink, filename)
        self.result.save_csv(filename)
        with open(filename) as f:
            lines = f.read().splitlines()
        header = [line for line in lines if line.startswith('#')]
        self.assertTrue(any('D_max' in line for line in header))
        self.assertTrue(any('Uncertainties fabricated: False' in line for line in header))
        rows = [line for line in lines if line and not line.startswith('#')]
        self.assertEqual(rows[0], 'r,P(r),dP(r)')
        self.assertEqual(len(rows) - 1, self.result.r.size)
        first = [float(v) for v in rows[1].split(',')]
        self.assertAlmostEqual(first[1], self.result.pr[0], places=10)

    def test_regularization_trend(self):
        """Over a wide alpha range, more smoothing does not increase oscillation."""
        prep = pr_solver._prepare_data(self.data)
        alpha_ref = pr_estimate._alpha_suggestion(prep, SPHERE_D_MAX, 10, True, 'corrected')
        osc_smooth = quiet_invert(self.data, SPHERE_D_MAX, n_terms=10, alpha=alpha_ref).oscillations
        osc_loose = quiet_invert(
            self.data, SPHERE_D_MAX, n_terms=10, alpha=alpha_ref * 1e-9
        ).oscillations
        self.assertLessEqual(osc_smooth, osc_loose + 0.05)

    def test_regularization_penalty_matches_integral(self):
        """Corrected-mode penalty is a quadrature of alpha * integral(P''^2 dr)."""
        r_fine = np.linspace(0.0, SPHERE_D_MAX, 4001)
        pr_second = np.zeros_like(r_fine)
        for j, c in enumerate(self.result.coefficients):
            pr_second += c * pr_basis._ortho_second_derivative(SPHERE_D_MAX, j + 1, r_fine)
        expected = self.result.alpha * np.trapezoid(pr_second**2, r_fine)
        self.assertAlmostEqual(self.result.regularization_penalty / expected, 1.0, delta=0.02)


class TestCylinderRoundTrip(unittest.TestCase):
    """Elongated fixture: 20 x 400 A cylinder, q_min chosen so pi/q_min >= D_max."""

    @classmethod
    def setUpClass(cls):
        cls.data = make_synthetic_data('cylinder', CYLINDER_PARS, CYLINDER_Q, seed=1)
        cls.i0_true = _true_i0('cylinder', CYLINDER_PARS)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            estimate = pri.estimate_n_terms(cls.data, CYLINDER_D_MAX)
            cls.result = pri.invert(
                cls.data, CYLINDER_D_MAX, n_terms=estimate.n_terms, alpha=estimate.alpha
            )

    def test_rg_within_five_percent(self):
        self.assertAlmostEqual(self.result.rg / CYLINDER_RG, 1.0, delta=0.05)

    def test_i0_within_ten_percent(self):
        self.assertAlmostEqual(self.result.i0 / self.i0_true, 1.0, delta=0.10)

    def test_pr_shape_is_skewed(self):
        """An elongated particle peaks well below D_max/2 with an extended tail."""
        peak_r = self.result.r[np.argmax(self.result.pr)]
        self.assertLess(peak_r, CYLINDER_D_MAX / 2)
        self.assertGreaterEqual(self.result.positive_fraction, 0.9)


class TestSasViewCompatibility(unittest.TestCase):
    """The 'sasview' regularizer reproduces SasView's matrix, formula for formula."""

    def test_reg_block_matches_sasview_formula(self):
        """Pinned oracle for Invertor._get_matrix's penalty rows.

        The expected values are frozen literals (computed once from SasView's
        formula at d_max=120, 20 penalty rows), so this check is independent
        of the implementation's own operator code — a formula change fails
        against the pinned numbers rather than against a mirrored formula.
        """
        d_max, n_terms = 120.0, 6
        q = np.geomspace(0.01, 0.2, 25)
        sigma = np.full(q.size, 0.5)
        _, _, reg_unit = pr_solver._build_blocks(
            q, np.ones(q.size), sigma, d_max, n_terms, True, 'sasview'
        )
        n_reg = pri.SASVIEW_N_REG
        self.assertEqual(reg_unit.shape, (n_reg, n_terms + 1))
        # The background column is never penalized.
        np.testing.assert_allclose(reg_unit[:, 0], 0.0, atol=0.0)
        expected_rows = {
            0: [
                0.628318530718,
                1.25663706144,
                1.88495559215,
                2.51327412287,
                3.14159265359,
                3.76991118431,
            ],
            1: [
                0.628302618393,
                1.25613037565,
                1.88113952896,
                2.49737810996,
                3.09379949403,
                3.65314018047,
            ],
            7: [
                0.593036510695,
                0.379222144157,
                -2.34809174057,
                -6.03311329968,
                -3.8850647056,
                7.42824171218,
            ],
            19: [
                -0.473907989929,
                0.0361801805728,
                2.1514964419,
                -6.78455455071,
                14.3533610051,
                -25.0916994972,
            ],
        }
        for m, values in expected_rows.items():
            np.testing.assert_allclose(reg_unit[m, 1:], values, rtol=1e-9)

    def test_data_block_matches_sasview_formula(self):
        d_max, n_terms = 120.0, 4
        q = np.geomspace(0.01, 0.2, 15)
        intensity = np.linspace(1.0, 2.0, 15)
        sigma = np.linspace(0.1, 0.3, 15)
        a_data, b_data, _ = pr_solver._build_blocks(
            q, intensity, sigma, d_max, n_terms, True, 'sasview'
        )
        np.testing.assert_allclose(a_data[:, 0], 1.0 / sigma, rtol=1e-12)
        for j in range(n_terms):
            np.testing.assert_allclose(
                a_data[:, j + 1],
                pr_basis._ortho_transformed(d_max, j + 1, q) / sigma,
                rtol=1e-12,
            )
        np.testing.assert_allclose(b_data, intensity / sigma, rtol=1e-12)

    def test_live_sasview_comparison(self):
        """Optional local extra: needs an importable SasView installation."""
        try:
            from sas.sascalc.pr.invertor import Invertor
        except ImportError:
            self.skipTest('SasView (sas.sascalc.pr) is not installed.')

        class _PrLogic:
            def __init__(self, data):
                self.data = data

        data = make_synthetic_data('sphere', SPHERE_PARS, SPHERE_Q, seed=1)
        inv = Invertor(_PrLogic(data))
        inv.dmax = SPHERE_D_MAX
        inv.alpha = 0.001
        inv.est_bck = True
        c_sv, _ = inv.lstsq(10)

        result = quiet_invert(data, SPHERE_D_MAX, n_terms=10, alpha=0.001, regularizer='sasview')
        # SasView's lstsq returns the post-shift array: background moved to
        # inv.background, coefficients shifted down, trailing zero appended.
        scale = float(np.max(np.abs(np.asarray(c_sv)[:10])))
        np.testing.assert_allclose(result.coefficients, np.asarray(c_sv)[:10], atol=1e-8 * scale)
        self.assertAlmostEqual(result.background, inv.background, places=6)


class TestDataPreparation(unittest.TestCase):
    """Validation, masking, sigma policy and warnings."""

    def setUp(self):
        self.data = make_synthetic_data('sphere', SPHERE_PARS, SPHERE_Q, seed=42)

    def test_q_range_restriction_matches_get_fit_index(self):
        self.data.qmin = 0.01
        self.data.qmax = 0.15
        result = quiet_invert(self.data, SPHERE_D_MAX, n_terms=8, alpha=1.0)
        expected = get_fit_index(self.data)
        self.assertEqual(result.n_points_used, int(expected.sum()))
        np.testing.assert_array_equal(result.accepted, expected)

    def test_q_range_via_fitter(self):
        from sans_fitter import SANSFitter

        fitter = SANSFitter()
        fitter.set_data(self.data)
        fitter.set_q_range(qmin=0.01, qmax=0.15)
        result = quiet_invert(fitter.data, SPHERE_D_MAX, n_terms=8, alpha=1.0)
        self.assertEqual(result.n_points_used, int(get_fit_index(fitter.data).sum()))

    def test_raw_data1d_accepted_without_mutation(self):
        raw = Data1D(
            x=SPHERE_Q, y=np.asarray(self.data.y).copy(), dy=np.asarray(self.data.dy).copy()
        )
        y_before = np.asarray(raw.y).copy()
        result = quiet_invert(raw, SPHERE_D_MAX, n_terms=8, alpha=1.0)
        self.assertEqual(result.n_points_used, SPHERE_Q.size)
        np.testing.assert_array_equal(np.asarray(raw.y), y_before)

    def test_fabricated_sigma_flagged_and_exported(self):
        data = Data1D(x=SPHERE_Q, y=np.asarray(self.data.y).copy(), dy=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = pri.invert(data, SPHERE_D_MAX, n_terms=8, alpha=1.0)
        self.assertTrue(result.uncertainties_fabricated)
        self.assertTrue(any('fabricated' in str(w.message) for w in caught))
        with tempfile.NamedTemporaryFile(mode='r', suffix='.csv', delete=False) as f:
            filename = f.name
        self.addCleanup(os.unlink, filename)
        result.save_csv(filename)
        with open(filename) as f:
            self.assertIn('Uncertainties fabricated: True', f.read())

    def test_partially_nan_dy_dropped(self):
        dy = np.asarray(self.data.dy).copy()
        dy[3] = np.nan
        dy[7] = 0.0
        raw = Data1D(x=SPHERE_Q, y=np.asarray(self.data.y).copy(), dy=dy)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = pri.invert(raw, SPHERE_D_MAX, n_terms=8, alpha=1.0)
        self.assertEqual(result.n_dropped_points, 2)
        self.assertEqual(result.n_points_used, SPHERE_Q.size - 2)
        self.assertTrue(any('dropped' in str(w.message) for w in caught))

    def test_infinite_intensity_dropped(self):
        y = np.asarray(self.data.y).copy()
        y[5] = np.inf
        raw = Data1D(x=SPHERE_Q, y=y, dy=np.asarray(self.data.dy).copy())
        result = quiet_invert(raw, SPHERE_D_MAX, n_terms=8, alpha=1.0)
        self.assertEqual(result.n_points_used, SPHERE_Q.size - 1)

    def test_q_zero_dropped_with_warning(self):
        q = np.concatenate([[0.0], SPHERE_Q])
        y = np.concatenate([[1.0], np.asarray(self.data.y)])
        dy = np.concatenate([[0.1], np.asarray(self.data.dy)])
        raw = Data1D(x=q, y=y, dy=dy)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = pri.invert(raw, SPHERE_D_MAX, n_terms=8, alpha=1.0)
        self.assertEqual(result.n_points_used, SPHERE_Q.size)
        self.assertTrue(any('dropped' in str(w.message) for w in caught))

    def test_negative_q_rejected(self):
        raw = Data1D(x=np.array([-0.01, 0.02, 0.03, 0.04]), y=np.ones(4), dy=np.ones(4))
        with self.assertRaises(ValueError):
            quiet_invert(raw, 100.0, n_terms=2, alpha=1.0)

    def test_2d_and_sesans_rejected(self):
        class Fake2D:
            qx_data = np.ones(3)

        with self.assertRaises(TypeError):
            quiet_invert(Fake2D(), 100.0)
        sesans = Data1D(x=SPHERE_Q, y=np.ones(SPHERE_Q.size), dy=np.ones(SPHERE_Q.size))
        sesans.isSesans = True
        with self.assertRaises(TypeError):
            quiet_invert(sesans, 100.0)

    def test_mismatched_lengths_rejected(self):
        raw = Data1D(x=np.array([0.01, 0.02, 0.03]), y=np.ones(3), dy=np.ones(3))
        raw.y = np.ones(5)
        with self.assertRaises(ValueError):
            quiet_invert(raw, 100.0, n_terms=2, alpha=1.0)
        raw = Data1D(x=np.array([0.01, 0.02, 0.03]), y=np.ones(3), dy=np.ones(2))
        with self.assertRaises(ValueError):
            quiet_invert(raw, 100.0, n_terms=2, alpha=1.0)

    def test_all_masked_rejected(self):
        raw = Data1D(x=SPHERE_Q, y=np.full(SPHERE_Q.size, np.nan), dy=np.ones(SPHERE_Q.size))
        with self.assertRaises(pri.InsufficientDataError):
            quiet_invert(raw, SPHERE_D_MAX, n_terms=8, alpha=1.0)

    def test_slit_smearing_warning(self):
        data = Data1D(
            x=SPHERE_Q, y=np.asarray(self.data.y).copy(), dy=np.asarray(self.data.dy).copy()
        )
        data.dxl = np.full(SPHERE_Q.size, 0.05)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pri.invert(data, SPHERE_D_MAX, n_terms=8, alpha=1.0)
        self.assertTrue(any('slit-smearing' in str(w.message) for w in caught))

    def test_argument_validation(self):
        for kwargs in (
            {'d_max': -1.0},
            {'d_max': 0.0},
            {'d_max': np.inf},
            {'d_max': np.nan},
            {'alpha': -0.5},
            {'alpha': np.nan},
            {'alpha': np.inf},
            {'n_terms': 0},
            {'n_terms': 8.5},
            {'n_terms': True},
            {'r_points': 1},
            {'r_points': 101.5},
            {'background': np.nan},
            {'background': np.inf},
            {'regularizer': 'bogus'},
        ):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                quiet_invert(self.data, **{'d_max': SPHERE_D_MAX, **kwargs})

    def test_insufficient_points_for_terms(self):
        raw = Data1D(x=SPHERE_Q[:6], y=np.asarray(self.data.y)[:6], dy=np.asarray(self.data.dy)[:6])
        with self.assertRaises(pri.InsufficientDataError):
            quiet_invert(raw, SPHERE_D_MAX, n_terms=8, alpha=1.0)

    def test_alpha_zero_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pri.invert(self.data, SPHERE_D_MAX, n_terms=8, alpha=0.0)
        self.assertTrue(any('unregularized' in str(w.message) for w in caught))

    def test_shannon_warnings(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pri.invert(self.data, 1000.0, n_terms=8, alpha=1.0)
        self.assertTrue(any('pi/q_min' in str(w.message) for w in caught))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pri.invert(self.data, SPHERE_D_MAX, n_terms=25, alpha=1.0)
        self.assertTrue(any('Shannon channels' in str(w.message) for w in caught))

    def test_no_shannon_warning_at_estimator_selected_n(self):
        """The channel warning uses the same ceiling as the estimator's bound,
        so an auto-selected N never triggers it (regression: n=10 at 9.55
        channels used to warn while ceil admitted it)."""
        import math

        channels = math.ceil(float(np.max(self.data.x)) * SPHERE_D_MAX / np.pi)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pri.invert(self.data, SPHERE_D_MAX, n_terms=channels, alpha=1.0)
        self.assertFalse(any('Shannon channels' in str(w.message) for w in caught))

    def test_explore_dmax_does_not_spam_shannon_warnings(self):
        """A fixed-N scan crosses the channel bound at its low-D end by
        construction; the per-point warning is suppressed inside the scan."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pri.explore_dmax(
                self.data,
                SPHERE_D_MAX,
                n_terms=10,
                alpha=1e5,
                dmin=0.7 * SPHERE_D_MAX,
                dmax=1.1 * SPHERE_D_MAX,
                n_points=9,
            )
        self.assertFalse(any('Shannon channels' in str(w.message) for w in caught))


class TestEstimators(unittest.TestCase):
    """Heuristic behaviour: bands over seeds, typed failures on noise."""

    def test_sphere_heuristics_stable_across_seeds(self):
        for seed in (1, 2, 3, 4, 5):
            data = make_synthetic_data('sphere', SPHERE_PARS, SPHERE_Q, seed=seed)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                estimate = pri.estimate_n_terms(data, SPHERE_D_MAX)
                result = pri.invert(
                    data, SPHERE_D_MAX, n_terms=estimate.n_terms, alpha=estimate.alpha
                )
            self.assertGreaterEqual(estimate.n_terms, 8, msg=f'seed {seed}')
            self.assertLessEqual(estimate.n_terms, 30, msg=f'seed {seed}')
            self.assertGreater(estimate.alpha, 0.0)
            self.assertGreaterEqual(result.sigma_positive_fraction, 0.9, msg=f'seed {seed}')

    def test_estimate_alpha_returns_positive_finite(self):
        data = make_synthetic_data('sphere', SPHERE_PARS, SPHERE_Q, seed=2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            estimate = pri.estimate_alpha(data, SPHERE_D_MAX, n_terms=10)
        self.assertTrue(np.isfinite(estimate.alpha))
        self.assertGreater(estimate.alpha, 0.0)
        self.assertTrue(estimate.message)

    def test_pure_noise_raises_estimation_error(self):
        rng = np.random.default_rng(7)
        q = np.geomspace(0.005, 0.25, 60)
        data = normalize_sans_data(Data1D(x=q, y=rng.normal(0.0, 1.0, q.size), dy=np.ones(q.size)))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with self.assertRaises(pri.PrEstimationError):
                pri.estimate_n_terms(data, 120.0)

    def test_tiny_dataset_raises_insufficient_data(self):
        q = np.geomspace(0.01, 0.1, 5)
        data = normalize_sans_data(Data1D(x=q, y=np.ones(5), dy=np.full(5, 0.1)))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with self.assertRaises(pri.InsufficientDataError):
                pri.estimate_n_terms(data, 120.0)

    def test_estimates_are_frozen(self):
        estimate = pri.AlphaEstimate(alpha=1.0, message='x')
        with self.assertRaises(AttributeError):
            estimate.alpha = 2.0

    def test_auto_invert_uses_scan_alpha(self):
        data = make_synthetic_data('sphere', SPHERE_PARS, SPHERE_Q, seed=4)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            estimate = pri.estimate_n_terms(data, SPHERE_D_MAX)
            result = pri.auto_invert(data, SPHERE_D_MAX)
        self.assertEqual(result.n_terms, estimate.n_terms)
        self.assertEqual(result.alpha, estimate.alpha)

    def test_fixed_background_threads_through_selection(self):
        """Selection with a fixed background equals selection on pre-subtracted
        data — the estimators must see the same problem the final fit solves."""
        background = 0.1
        data = make_synthetic_data('sphere', SPHERE_PARS, SPHERE_Q, seed=5, background=background)
        subtracted = normalize_sans_data(
            Data1D(x=SPHERE_Q, y=np.asarray(data.y) - background, dy=np.asarray(data.dy))
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with_bkg = pri.estimate_n_terms(
                data, SPHERE_D_MAX, fit_background=False, background=background
            )
            pre_sub = pri.estimate_n_terms(subtracted, SPHERE_D_MAX, fit_background=False)
        self.assertEqual(with_bkg.n_terms, pre_sub.n_terms)
        self.assertAlmostEqual(with_bkg.alpha, pre_sub.alpha)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            auto = pri.auto_invert(data, SPHERE_D_MAX, fit_background=False, background=background)
        self.assertEqual(auto.n_terms, with_bkg.n_terms)
        self.assertEqual(auto.alpha, with_bkg.alpha)

    def test_estimate_alpha_accepts_background(self):
        background = 0.1
        data = make_synthetic_data('sphere', SPHERE_PARS, SPHERE_Q, seed=5, background=background)
        subtracted = normalize_sans_data(
            Data1D(x=SPHERE_Q, y=np.asarray(data.y) - background, dy=np.asarray(data.dy))
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with_bkg = pri.estimate_alpha(
                data, SPHERE_D_MAX, n_terms=10, fit_background=False, background=background
            )
            pre_sub = pri.estimate_alpha(subtracted, SPHERE_D_MAX, n_terms=10, fit_background=False)
        self.assertAlmostEqual(with_bkg.alpha, pre_sub.alpha)


class TestExploreDmax(unittest.TestCase):
    """D_max scan: plateau location, failure alignment, alpha policies."""

    @classmethod
    def setUpClass(cls):
        cls.data = make_synthetic_data('sphere', SPHERE_PARS, SPHERE_Q, seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            estimate = pri.estimate_n_terms(cls.data, SPHERE_D_MAX)
        cls.n_terms = estimate.n_terms
        cls.alpha = estimate.alpha

    def test_chisq_minimum_near_true_dmax(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            scan = pri.explore_dmax(
                self.data,
                SPHERE_D_MAX,
                n_terms=self.n_terms,
                alpha=self.alpha,
                dmin=0.75 * SPHERE_D_MAX,
                dmax=1.25 * SPHERE_D_MAX,
                n_points=11,
            )
        best = scan.d_max_values[np.argmin(scan.data_chisq)]
        self.assertAlmostEqual(best / SPHERE_D_MAX, 1.0, delta=0.10)

    def test_arrays_align_with_dmax_values(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            scan = pri.explore_dmax(
                self.data, SPHERE_D_MAX, n_terms=self.n_terms, alpha=self.alpha, n_points=7
            )
        n = scan.d_max_values.size
        for attr in (
            'data_chisq',
            'rg',
            'i0',
            'oscillations',
            'positive_fraction',
            'sigma_positive_fraction',
            'background',
            'alpha',
        ):
            self.assertEqual(getattr(scan, attr).size, n, msg=attr)

    def test_injected_failure_recorded_and_aligned(self):
        original = pr_solver._invert_prepared
        failing_index = {'count': 0}

        def flaky(prep, d_max, *args, **kwargs):
            failing_index['count'] += 1
            if failing_index['count'] == 3:
                raise ValueError('injected failure')
            return original(prep, d_max, *args, **kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with patch.object(pr_estimate, '_invert_prepared', side_effect=flaky):
                scan = pri.explore_dmax(
                    self.data, SPHERE_D_MAX, n_terms=self.n_terms, alpha=self.alpha, n_points=6
                )
        self.assertEqual(len(scan.failures), 1)
        self.assertIn('injected failure', scan.failures[0][1])
        self.assertEqual(scan.d_max_values.size, 5)
        self.assertEqual(scan.rg.size, 5)

    def test_all_failed_scan_raises(self):
        """An entirely failed scan raises instead of returning empty arrays."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with patch.object(
                pr_estimate, '_invert_prepared', side_effect=ValueError('always broken')
            ):
                with self.assertRaises(pri.PrEstimationError) as ctx:
                    pri.explore_dmax(
                        self.data,
                        SPHERE_D_MAX,
                        n_terms=self.n_terms,
                        alpha=self.alpha,
                        n_points=4,
                    )
        self.assertIn('always broken', str(ctx.exception))

    def test_background_threaded_through_scan(self):
        """A scan with a fixed background equals a scan on pre-subtracted data."""
        background = 0.1
        data = make_synthetic_data('sphere', SPHERE_PARS, SPHERE_Q, seed=6, background=background)
        subtracted = normalize_sans_data(
            Data1D(x=SPHERE_Q, y=np.asarray(data.y) - background, dy=np.asarray(data.dy))
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with_bkg = pri.explore_dmax(
                data,
                SPHERE_D_MAX,
                n_terms=self.n_terms,
                alpha=self.alpha,
                n_points=5,
                fit_background=False,
                background=background,
            )
            pre_sub = pri.explore_dmax(
                subtracted,
                SPHERE_D_MAX,
                n_terms=self.n_terms,
                alpha=self.alpha,
                n_points=5,
                fit_background=False,
            )
        np.testing.assert_allclose(with_bkg.rg, pre_sub.rg, rtol=1e-10)
        np.testing.assert_allclose(with_bkg.data_chisq, pre_sub.data_chisq, rtol=1e-10)

    def test_scan_emits_single_support_warning(self):
        """A scan crossing pi/q_min warns once at scan level, not per point."""
        support = np.pi / float(np.min(self.data.x))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pri.explore_dmax(
                self.data,
                0.9 * support,
                n_terms=self.n_terms,
                alpha=self.alpha,
                dmin=0.8 * support,
                dmax=1.2 * support,
                n_points=6,
            )
        support_warnings = [w for w in caught if 'pi/q_min' in str(w.message)]
        self.assertEqual(len(support_warnings), 1)

    def test_refit_alpha_comparable_plateau(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fixed = pri.explore_dmax(
                self.data, SPHERE_D_MAX, n_terms=self.n_terms, alpha=self.alpha, n_points=5
            )
            refit = pri.explore_dmax(
                self.data, SPHERE_D_MAX, n_terms=self.n_terms, n_points=5, refit_alpha=True
            )
        np.testing.assert_allclose(fixed.rg, refit.rg, rtol=0.05)

    def test_format_summary(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            scan = pri.explore_dmax(
                self.data, SPHERE_D_MAX, n_terms=self.n_terms, alpha=self.alpha, n_points=4
            )
        summary = scan.format_summary()
        self.assertIn('D_max scan', summary)
        summary.encode('ascii')


class TestPlots(unittest.TestCase):
    """Trace structure of the three plot functions (never shown)."""

    @classmethod
    def setUpClass(cls):
        cls.data = make_synthetic_data('sphere', SPHERE_PARS, SPHERE_Q, seed=42)
        cls.data.qmin = 0.01  # leave some excluded points
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cls.result = pri.invert(cls.data, SPHERE_D_MAX, n_terms=10, alpha=1e5)
            cls.scan = pri.explore_dmax(cls.data, SPHERE_D_MAX, n_terms=10, alpha=1e5, n_points=4)

    def test_plot_pr_distribution(self):
        fig = plot_pr_distribution(self.result, show=False)
        names = [trace.name for trace in fig.data]
        self.assertIn('P(r)', names)
        self.assertIn('±1σ', names)

    def test_plot_pr_fit_traces(self):
        fig = plot_pr_fit(self.data, self.result, show=False)
        names = [trace.name for trace in fig.data]
        self.assertIn('Experimental Data', names)
        self.assertIn('Excluded Data', names)
        self.assertIn('P(r) Fit', names)
        self.assertEqual(len(fig.data), 4)  # data, excluded, fit, residuals

    def test_result_plot_wrappers(self):
        fig_pr = self.result.plot_pr(show=False)
        fig_fit = self.result.plot_fit(self.data, show=False)
        self.assertGreaterEqual(len(fig_pr.data), 2)
        self.assertGreaterEqual(len(fig_fit.data), 3)

    def test_plot_dmax_scan_single_and_all(self):
        fig = plot_dmax_scan(self.scan, quantity='rg', show=False)
        self.assertEqual(len(fig.data), 1)
        fig_all = plot_dmax_scan(self.scan, quantity='all', show=False)
        self.assertEqual(len(fig_all.data), 7)  # one trace per DMAX_SCAN_QUANTITIES entry

    def test_plot_dmax_scan_unknown_quantity(self):
        with self.assertRaises(ValueError):
            plot_dmax_scan(self.scan, quantity='bogus', show=False)


class TestUncertaintyBand(unittest.TestCase):
    """The P(r) 1-sigma band follows the closed form for a 1-term expansion."""

    def test_band_matches_closed_form_for_one_term(self):
        # With a single coefficient, P(r) = c1*Phi_1(r), so
        # sigma_P(r) = sqrt(Var(c1)) * |Phi_1(r)| exactly.
        data = make_synthetic_data('sphere', SPHERE_PARS, SPHERE_Q)
        result = quiet_invert(data, SPHERE_D_MAX, n_terms=1, alpha=0.0)
        variance = float(np.asarray(result.coefficient_covariance)[0, 0])
        r = np.linspace(0.0, SPHERE_D_MAX, 37)
        basis = 2.0 * r * np.sin(np.pi * r / SPHERE_D_MAX)
        np.testing.assert_allclose(
            result.evaluate_pr_err(r), np.sqrt(variance) * np.abs(basis), rtol=1e-10, atol=1e-30
        )


class TestAutoInvertMultimodalWarning(unittest.TestCase):
    """auto_invert must warn audibly when no scanned N fits the data."""

    def test_warns_on_bimodal_population(self):
        # Two sphere populations (R=60 and R=15): the estimator's unimodal
        # peak criterion over-smooths, no N passes the chi-squared gate, and
        # the fallback pick must surface as a warning, not a debug log.
        q = np.geomspace(0.005, 0.25, 120)
        big = _clean_intensity('sphere', SPHERE_PARS, q)
        # Weight the small population so it contributes comparably to I(q)
        # (I(0) scales with volume squared, so an unweighted R=15 population
        # would be invisible next to R=60).
        small = _clean_intensity('sphere', dict(SPHERE_PARS, radius=15.0, scale=1000.0), q)
        i_clean = np.asarray(big) + np.asarray(small)
        rng = np.random.default_rng(7)
        sigma = 0.02 * np.abs(i_clean)
        data = normalize_sans_data(Data1D(x=q, y=i_clean + rng.normal(0.0, sigma), dy=sigma))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pri.auto_invert(data, d_max=SPHERE_D_MAX)
        self.assertTrue(
            any('fit the data' in str(w.message) for w in caught),
            f'no fallback warning among: {[str(w.message) for w in caught]}',
        )


if __name__ == '__main__':
    unittest.main()
