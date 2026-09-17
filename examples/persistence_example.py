"""
Example: Saving, Reloading and Sharing an Analysis

This script demonstrates how to:
1. Save a complete analysis setup and its fit result as JSON (save_analysis)
2. Reopen it in a fresh fitter and use the result without refitting (load_analysis)
3. Move the analysis to another directory and still find its data
4. See why a result is left out when the setup changed after the fit
5. Save a setup-only template and apply it to the next sample (data=)
6. Write a shareable HTML or Markdown report (report)

save_results() writes the outcome of a fit. save_analysis() writes how the fit
was set up: model, parameter values, bounds and vary flags, polydispersity,
links, structure factor, resolution mode and Q range.
"""

import json
import os
import shutil
import warnings

import numpy as np

from sans_fitter import SANSFitter, examples

# Everything this example writes goes under one directory, recreated on each
# run, so the "move the project" step in Part 4 has something real to move.
WORKDIR = os.path.abspath('persistence_example_output')
if os.path.isdir(WORKDIR):
    shutil.rmtree(WORKDIR)
PROJECT = os.path.join(WORKDIR, 'beamtime_2026')
os.makedirs(os.path.join(PROJECT, 'data'))


def write_csv(data, path):
    """Write Q, I, dI columns, the way a reduction pipeline often delivers them.

    The dQ column is deliberately left out: the resolution then has to be
    stated with set_resolution(), which makes it part of the analysis rather
    than of the file, and so something the saved analysis must remember.
    """
    table = np.column_stack([data.x, data.y, data.dy])
    np.savetxt(path, table, delimiter=',', header='Q,I,dI', comments='')


# ============================================================================
# Part 1: Measure and Fit
# ============================================================================

print('=' * 80)
print('Part 1: Measure and Fit')
print('=' * 80)

# Polydisperse spheres, smeared by a 5% pinhole resolution, with known truth.
measured = examples.simulate(
    'sphere',
    radius=60.0,
    radius_pd=0.10,
    sld=4.0,
    sld_solvent=1.0,
    scale=0.05,
    background=0.001,
    npoints=80,
    dq=0.05,
    seed=3,
)
sample_file = os.path.join(PROJECT, 'data', 'sample_A.csv')
write_csv(measured, sample_file)
print(f'\nWrote {os.path.relpath(sample_file)} (radius = {measured.truth["radius"]} Å)')

fitter = SANSFitter()
fitter.load_data(sample_file)
fitter.set_model('sphere')
fitter.set_param('radius', value=50, min=10, max=200, vary=True)
fitter.set_param('scale', value=0.05, min=1e-4, max=1, vary=True)
fitter.set_param('background', value=0.001, min=0, max=0.1, vary=True)
fitter.set_param('sld', value=4.0, vary=False)
fitter.set_param('sld_solvent', value=1.0, vary=False)

fitter.enable_polydispersity(True)
fitter.set_pd_param('radius', pd_width=0.1, pd_type='gaussian', vary=True)

# The file has no dQ column, so the resolution is our choice.
fitter.set_resolution('pinhole', dq_over_q=0.05)
fitter.set_q_range(qmin=0.01, qmax=0.4)

result = fitter.fit(engine='bumps', method='amoeba')
print(f'\nFitted radius: {result["parameters"]["radius"]["formatted"]} Å')
print(f'χ²/dof:        {result["reduced_chisq"]:.3f}')

# ============================================================================
# Part 2: Save the Analysis
# ============================================================================

print('\n' + '=' * 80)
print('Part 2: Save the Analysis')
print('=' * 80)

analysis_file = os.path.join(PROJECT, 'sample_A.json')
fitter.save_analysis(analysis_file)

with open(analysis_file, encoding='utf-8') as handle:
    document = json.load(handle)

print('\nTop-level sections:', ', '.join(document))
print('Data file (relative):', document['data']['path_relative'])
print('Model:               ', document['configuration']['model_name'])
print('Resolution:          ', document['resolution'])
print('Q range:             ', document['fit_range'])
print('Result included:     ', document['result'] is not None)

# Bounds that are infinite by default (scale and background have no upper
# limit unless you set one) are written as the string "Infinity", so the file
# stays strict JSON that any tool can read.

# ============================================================================
# Part 3: Reopen It, No Refit Needed
# ============================================================================

print('\n' + '=' * 80)
print('Part 3: Reopen It, No Refit Needed')
print('=' * 80)

reopened = SANSFitter.load_analysis(analysis_file)

print(
    f'\nResolution restored: {reopened.get_resolution()["mode"]}, '
    f'dq/q = {reopened.get_resolution()["dq_over_q"]}'
)
print(f'Q range restored:    {reopened.get_q_range()}')
print(
    f'PD restored:         {reopened.get_pd_param("radius")["pd"]:.4f} '
    f'({reopened.get_pd_param("radius")["pd_type"]})'
)

print(f'Result attached:     χ²/dof = {reopened.fit_result["reduced_chisq"]:.3f}')
print('\nFit report of the reopened analysis:')
print(reopened.get_fit_report())
reopened.plot_results(show_residuals=True, log_scale=True)

# Posterior sample chains and data curves are not stored. The curve is rebuilt
# from the saved parameters and checked against the saved χ² before the result
# is attached, so a file whose numbers do not match its setup is reported
# rather than plotted.

# ============================================================================
# Part 4: Move the Project, Keep the Data Link
# ============================================================================

print('\n' + '=' * 80)
print('Part 4: Move the Project, Keep the Data Link')
print('=' * 80)

# The data path is recorded relative to the analysis file as well as
# absolutely. Moving (or zipping and emailing) the project directory as a whole
# keeps the relative path valid.
moved = os.path.join(WORKDIR, 'shared_with_collaborator')
shutil.move(PROJECT, moved)
print(f'\nMoved the project to {os.path.relpath(moved)}')

from_elsewhere = SANSFitter.load_analysis(os.path.join(moved, 'sample_A.json'))
print(f'Data points loaded:  {len(from_elsewhere.data.x)}')
print(f'Result still valid:  {from_elsewhere.fit_result is not None}')

# ============================================================================
# Part 5: A Result Is Saved Only While It Describes the Setup
# ============================================================================

print('\n' + '=' * 80)
print('Part 5: A Result Is Saved Only While It Describes the Setup')
print('=' * 80)

# Editing a parameter does not clear the last fit result. Saving both together
# would pair a χ² with a configuration that never produced it, so the setup is
# written and the result is left out, with the reason.
from_elsewhere.set_param('radius', value=80)
stale_file = os.path.join(moved, 'sample_A_edited.json')
from_elsewhere.save_analysis(stale_file)

with open(stale_file, encoding='utf-8') as handle:
    stale = json.load(handle)
print(f'\nResult included: {stale["result"] is not None}')
print(f'Reason:          {stale["result_omitted"]}')

# The same check applies to the Q range, the resolution mode and the data
# itself. Fit before you save, or save before you experiment.

# ============================================================================
# Part 6: A Template for the Next Sample
# ============================================================================

print('\n' + '=' * 80)
print('Part 6: A Template for the Next Sample')
print('=' * 80)

# include_result=False saves the configured model without the sample-specific
# outcome: the model, bounds, polydispersity and resolution, ready to reuse.
template_file = os.path.join(moved, 'sphere_template.json')
reopened.save_analysis(template_file, include_result=False)

# A second sample with larger particles, measured the same way.
second = examples.simulate(
    'sphere',
    radius=75.0,
    radius_pd=0.10,
    sld=4.0,
    sld_solvent=1.0,
    scale=0.05,
    background=0.001,
    npoints=80,
    dq=0.05,
    seed=4,
)
second_file = os.path.join(moved, 'data', 'sample_B.csv')
write_csv(second, second_file)

# data= points the template at the new file instead of the recorded one. This
# prints a warning that the data differs from what the analysis was saved
# from. Here that is the point; the warning exists for the case where it is a
# mistake, and it is also why no saved result is ever attached to other data.
next_sample = SANSFitter.load_analysis(template_file, data=second_file)
print(
    f'\nTemplate applied to {os.path.basename(second_file)}; '
    f'result attached: {next_sample.fit_result is not None}'
)

result_b = next_sample.fit(engine='bumps', method='amoeba')
print(
    f'Fitted radius: {result_b["parameters"]["radius"]["formatted"]} Å '
    f'(truth {second.truth["radius"]} Å)'
)
next_sample.save_analysis(os.path.join(moved, 'sample_B.json'))

# An analysis whose data came from set_data() (a data_ops result, or a
# simulated dataset) has no file to reopen, so load_analysis needs data= for it
# every time. The error message quotes the dataset's provenance to say which.

# ============================================================================
# Part 7: Reports to Share
# ============================================================================

print('\n' + '=' * 80)
print('Part 7: Reports to Share')
print('=' * 80)

# HTML embeds the interactive plot and needs no extra dependencies.
html_file = os.path.join(moved, 'sample_B_report.html')
next_sample.report(html_file)
print(f'\nWrote {os.path.relpath(html_file)}')

# offline=True inlines the Plotly library, for a file that opens without a
# network connection (a few MB instead of a few tens of kB).

# Markdown suits an issue, a pull request or a logbook. Its figure is a PNG
# next to the report, named after it, which needs the optional renderer:
#     pip install "sans-fitter[report]"
# Without it the report is still written, without the figure, and says so.
markdown_file = os.path.join(moved, 'sample_B_report.md')
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    report = next_sample.report(markdown_file)
figure = next(iter(report.assets), None)
print(
    f'Wrote {os.path.relpath(markdown_file)}; '
    f'figure: {figure or "omitted (no image renderer installed)"}'
)

# Without a filename nothing is written: the returned report renders itself in
# a notebook, and str(report) is the Markdown text.
print('\nFirst lines of the Markdown report:')
print('\n'.join(str(report).splitlines()[:12]))

print('\n' + '=' * 80)
print('Summary: Persistence and Reporting Key Points')
print('=' * 80)
print("""
✓ save_analysis() writes the setup and, while it still matches, the last result
✓ load_analysis() restores the setup exactly and attaches the result, no refit
✓ The data path is stored relative to the analysis file, so projects can move
✓ Editing the setup after a fit leaves the result out of the file, with a reason
✓ include_result=False saves a template; data= applies it to another sample
✓ report('x.html') needs nothing extra; report('x.md') needs the [report] extra
✓ Only built-in models load by default: allow_custom_models=True for plugins
✓ Restoring guarantees the configuration, not that fit() repeats the same numbers
""")

print(f'All files are under {os.path.relpath(WORKDIR)}')
print('\n✓ Persistence example completed successfully!')
