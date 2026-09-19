"""Simultaneous fitting of several datasets with MultiFitter.

Runs four analyses on simulated data with known answers, so every number
printed can be checked against the truth it was generated from:

1. **Contrast variation** — two contrasts of one particle. The radius and the
   concentration are shared, the solvent SLDs are known constants, and the
   incoherent backgrounds stay separate.
2. **Fitting the curves separately** — the same data, one dataset at a time,
   to show what the joint constraint buys.
3. **Mixed instrument configurations** — three Q ranges with different
   resolution, fitted together without merging or rebinning anything.
4. **An arithmetic constraint** — a concentration series where one scale is a
   known multiple of another.

Run it with:

    python examples/simultaneous_fitting_example.py
"""

import numpy as np

from sans_fitter import MultiFitter, SANSFitter, examples, set_verbosity

# The fitter narrates what it does; this example prints its own summaries.
set_verbosity('warning')

TRUE_RADIUS = 48.0
TRUE_PD = 0.12
TRUE_SCALE = 0.02
TRUE_SLD = 4.0


def heading(text: str) -> None:
    print(f'\n{"=" * 78}\n{text}\n{"=" * 78}')


# =========================================================================
# 1. Contrast variation
# =========================================================================


def contrast_variation():
    """One particle in two solvents: share the geometry, keep the rest local."""
    heading('1. Contrast variation — shared geometry, separate backgrounds')

    solvents = {'h2o': -0.56, 'd2o': 6.34}
    backgrounds = {'h2o': 0.012, 'd2o': 0.035}
    datasets = {
        name: examples.simulate(
            'sphere',
            radius=TRUE_RADIUS,
            radius_pd=TRUE_PD,
            scale=TRUE_SCALE,
            background=backgrounds[name],
            sld=TRUE_SLD,
            sld_solvent=solvent,
            qmin=0.006,
            qmax=0.4,
            npoints=70,
            noise=0.03,
            seed=seed,
        )
        for seed, (name, solvent) in enumerate(solvents.items(), start=101)
    }

    fit = MultiFitter()
    for name, data in datasets.items():
        fit.add(name, data, model='sphere')
        entry = fit[name]
        # Start deliberately away from the truth, and identically in both
        # datasets — share() requires the members to agree before it will make
        # them one quantity.
        entry.set_param('radius', value=35, min=10, max=120, vary=True)
        entry.set_param('scale', value=0.01, min=0.001, max=0.2, vary=True)
        entry.set_param('background', value=0.02, min=0.0, max=0.2, vary=True)
        entry.set_param('sld', value=TRUE_SLD, vary=False)
        entry.enable_polydispersity(True)
        entry.set_pd_param('radius', pd_width=0.05, pd_type='gaussian', vary=True)
        # The solvent contrast is measured, not fitted.
        fit.constrain(f'{name}.sld_solvent', solvents[name])

    # The particle is the same in both solvents, and so is its size
    # distribution and its concentration. The background is not: incoherent
    # scattering follows the solvent.
    fit.share('radius', 'radius_pd', 'scale')

    fit.describe()

    result = fit.fit(method='lm')
    print(fit.get_fit_report())

    print('\nAgainst the truth the data was generated from:')
    for label, fitted, truth in (
        ('radius', result.parameters['d2o.radius'], TRUE_RADIUS),
        ('radius_pd', result.parameters['d2o.radius_pd'], TRUE_PD),
        ('scale', result.parameters['d2o.scale'], TRUE_SCALE),
        ('h2o background', result.parameters['h2o.background'], backgrounds['h2o']),
        ('d2o background', result.parameters['d2o.background'], backgrounds['d2o']),
    ):
        deviation = abs(fitted.value - truth) / fitted.stderr if fitted.stderr else float('nan')
        print(
            f'  {label:<16} {fitted.formatted:>18}   truth {truth:<8g} '
            f'({deviation:.1f} sigma away)'
        )

    return datasets, result


# =========================================================================
# 2. The same data, fitted separately
# =========================================================================


def fit_separately(datasets):
    """What each curve says on its own, for comparison with the joint answer."""
    heading('2. The same two curves, fitted one at a time')

    solvents = {'h2o': -0.56, 'd2o': 6.34}
    print(f'{"Dataset":<10} {"radius":>18} {"scale":>18}')
    print('-' * 48)
    for name, data in datasets.items():
        fitter = SANSFitter()
        fitter.set_data(data)
        fitter.set_model('sphere')
        fitter.set_param('radius', value=35, min=10, max=120, vary=True)
        fitter.set_param('scale', value=0.01, min=0.001, max=0.2, vary=True)
        fitter.set_param('background', value=0.02, min=0.0, max=0.2, vary=True)
        fitter.set_param('sld', value=TRUE_SLD, vary=False)
        fitter.set_param('sld_solvent', value=solvents[name], vary=False)
        fitter.enable_polydispersity(True)
        fitter.set_pd_param('radius', pd_width=0.05, vary=True)
        separate = fitter.fit(engine='bumps', method='lm')
        print(
            f'{name:<10} {separate["parameters"]["radius"]["formatted"]:>18} '
            f'{separate["parameters"]["scale"]["formatted"]:>18}'
        )

    print(
        '\nEach curve gives its own radius, with its own error bar. The joint fit\n'
        'above gives one radius constrained by both — which is the point: the\n'
        'two contrasts see the same particle, so they should not be allowed to\n'
        'disagree about how big it is.'
    )


# =========================================================================
# 3. Several instrument configurations
# =========================================================================


def instrument_configurations():
    """Three detector settings, three Q windows, three resolutions, one fit."""
    heading('3. Three instrument configurations — different Q ranges and resolution')

    settings = {
        'low_q': {'qmin': 0.003, 'qmax': 0.03, 'dq': 0.14, 'npoints': 30},
        'mid_q': {'qmin': 0.02, 'qmax': 0.15, 'dq': 0.10, 'npoints': 40},
        'high_q': {'qmin': 0.1, 'qmax': 0.5, 'dq': 0.06, 'npoints': 35},
    }

    fit = MultiFitter()
    for seed, (name, setting) in enumerate(settings.items(), start=201):
        data = examples.simulate(
            'sphere',
            radius=TRUE_RADIUS,
            radius_pd=TRUE_PD,
            scale=TRUE_SCALE,
            background=0.02,
            sld=TRUE_SLD,
            sld_solvent=6.34,
            qmin=setting['qmin'],
            qmax=setting['qmax'],
            npoints=setting['npoints'],
            dq=setting['dq'],
            noise=0.03,
            seed=seed,
        )
        fit.add(name, data, model='sphere')
        entry = fit[name]
        entry.set_param('radius', value=35, min=10, max=120, vary=True)
        entry.set_param('scale', value=0.02, min=0.001, max=0.2, vary=True)
        entry.set_param('background', value=0.02, min=0.0, max=0.2, vary=True)
        entry.set_param('sld', value=TRUE_SLD, vary=False)
        entry.enable_polydispersity(True)
        entry.set_pd_param('radius', pd_width=0.05, vary=True)
        fit.constrain(f'{name}.sld_solvent', 6.34)
        # Each configuration carries its own simulated dQ column, and the
        # default 'data' mode uses it. Nothing here needs a common Q grid.
        entry.set_resolution('data')

    # It is one sample, so everything about the sample is shared. Only the
    # background is per-configuration, because it depends on the setup.
    fit.share('radius', 'radius_pd', 'scale')

    result = fit.fit(method='lm')

    print(f'{"Configuration":<14} {"Points":>7} {"Q range":>20} {"chi2":>10} {"chi2/N":>9}')
    print('-' * 64)
    for entry in result.datasets.values():
        span = f'{entry.q_range[0]:.4g} - {entry.q_range[1]:.4g}'
        print(
            f'{entry.name:<14} {entry.n_points:>7} {span:>20} '
            f'{entry.chisq:>10.2f} {entry.mean_squared_residual:>9.3f}'
        )
    radius = result.parameters['high_q.radius']
    print(
        f'\nJoint radius: {radius.formatted}   (truth {TRUE_RADIUS})\n'
        f'{result.n_points} points across {result.n_datasets} configurations, '
        f'{result.n_free} free parameters, chi2/dof = {result.reduced_chisq:.3f}.\n'
        'The three curves were fitted on their own grids with their own\n'
        'resolution; only the residual vectors were concatenated.'
    )


# =========================================================================
# 4. An arithmetic constraint
# =========================================================================


def known_dilution():
    """A 2:1 dilution series, where the ratio is known and the scale is not."""
    heading('4. Concentration series — one scale, a known dilution factor')

    fit = MultiFitter()
    for seed, (name, scale) in enumerate(
        (('stock', 0.024), ('diluted', 0.012)), start=301
    ):
        data = examples.simulate(
            'sphere',
            radius=TRUE_RADIUS,
            scale=scale,
            background=0.01,
            sld=TRUE_SLD,
            sld_solvent=6.34,
            qmin=0.006,
            qmax=0.35,
            npoints=60,
            noise=0.03,
            seed=seed,
        )
        fit.add(name, data, model='sphere')
        entry = fit[name]
        entry.set_param('radius', value=40, min=10, max=120, vary=True)
        entry.set_param('background', value=0.01, min=0.0, max=0.1, vary=True)
        entry.set_param('sld', value=TRUE_SLD, vary=False)
        fit.constrain(f'{name}.sld_solvent', 6.34)

    fit.share('radius')
    # The stock scale is what we want; the dilution ratio is known from the
    # preparation, so the diluted scale is not an independent parameter.
    fit['stock'].set_param('scale', value=0.02, min=0.001, max=0.1, vary=True)
    fit['diluted'].set_param('scale', min=0.0005, max=0.05)
    fit.constrain('diluted.scale', '0.5 * stock.scale')

    result = fit.fit(method='lm')

    stock = result.parameters['stock.scale']
    diluted = result.parameters['diluted.scale']
    print(f'  stock.scale    {stock.formatted:>18}   truth 0.024   ({stock.status})')
    print(f'  diluted.scale  {diluted.formatted:>18}   truth 0.012   ({diluted.status})')
    print(f'  ratio held exactly: {diluted.value / stock.value:.6f}')
    print(f'  uncertainty source: {diluted.uncertainty_source}')
    print(
        f'\nThe constraint removed a parameter: {result.n_free} free rather than '
        f'{result.n_free + 1}.\nThe derived error is the root error times the '
        'coefficient, not an independent estimate.'
    )

    # An expression that cannot be kept inside its target's bounds is refused
    # before anything is fitted, rather than quietly losing the bound.
    try:
        fit.constrain('diluted.background', '1 / stock.background')
    except ValueError as error:
        print(f'\nRefused, as intended:\n  {error}')


def main():
    datasets, _result = contrast_variation()
    fit_separately(datasets)
    instrument_configurations()
    known_dilution()

    heading('Done')
    print(
        'Every fit above used one bumps problem with one set of free parameters.\n'
        'See docs/multifit.md for choosing what to share, how constraint bounds\n'
        'are enforced, and what dataset weights do to the reported uncertainties.'
    )


if __name__ == '__main__':
    np.seterr(all='ignore')
    main()
