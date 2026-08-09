# Example Data

Getting started with a fitting package usually means finding a data file first.
`sans_fitter.examples` removes that step in two ways: a curated set of **bundled
datasets**, and a **simulator** that generates data.

```python
from sans_fitter import examples

examples.describe()                       # what's available
data = examples.load('silica_spheres')    # fit-ready Data1D
fitter = examples.load_fitter('silica_spheres')   # data + model + parameters
result = fitter.fit()
```

## Bundled datasets

These are the same example datasets SasView ships. They are **not** vendored
into this repository — they live inside the installed `sasdata` package, which
is already a hard dependency. That keeps the wheel small and the collection in
sync with sasdata.

`examples.describe()` prints the whole collection:

| Name | Model | What it is good for |
|---|---|---|
| `cylinder` | cylinder | Noise-free calculated cylinder, R=20 Å, L=400 Å. A verification reference, not a fitting exercise (see the note below). |
| `sphere` | sphere | 100 nm spheres, dI but no dQ. |
| `sphere_smeared` | sphere | The same spheres **with** dQ — pair the two to see what resolution smearing does. |
| `polydisperse_spheres` | sphere | Fit with polydispersity off, then on, and watch the residuals collapse. |
| `silica_spheres` | sphere | Measured Ludox colloidal silica, with both dI and dQ. |
| `sds_micelles` | ellipsoid | Measured charged SDS micelles — needs the `hayter_msa` structure factor. |
| `sds_micelles_salt` | ellipsoid | The same micelles with 0.2 M NaCl; the salt screens the charge, so `hardsphere` suffices. |
| `core_shell` | core_shell_sphere | Partly degenerate core radius and shell thickness — a lesson in correlated parameters. |
| `polymer_micelles` | ellipsoid | Measured 10% Pluronic P123, with a clear structure-factor peak. |
| `protein` | sphere | Measured apoferritin: ~400 points, flat incoherent background, noisy high-Q tail. |
| `canSAS_xml` | sphere | Measured SANS2D data as canSAS-1D XML. |
| `nxcanSAS_h5` | sphere | The identical measurement as NXcanSAS HDF5. |

Filter by tag to find one of a given kind:

```python
examples.list_examples(tag='measured')          # real instrument data
examples.list_examples(tag='structure-factor')  # concentrated samples
examples.list_examples(tag='resolution')        # datasets carrying dQ
```

`examples.describe('name')` prints the full detail for one entry, including live
facts read from the file — point count, Q range, and whether dI and dQ are
present:

```
silica_spheres
==============
Measured Ludox colloidal silica. Carries both dI and dQ, so it exercises
weighted fitting and resolution smearing on real data.

  file              Ludox_silica.xml
  model             sphere
  polydispersity    radius
  tags              measured, colloid, resolution
  points            92
  Q range           0.0071429 to 0.25169 1/A
  dI column         yes
  dQ column         yes
```

### Presets

`load_fitter()` returns a `SANSFitter` with the data, model, structure factor,
polydispersity and starting parameters already set:

```python
fitter = examples.load_fitter('sds_micelles')
result = fitter.fit()
fitter.plot_results()
```

The starting values are coarse — chosen to put the optimiser in the right basin,
not published results. Check `examples.get_example(name).truth` to see whether
ground truth is known at all; for measured data it is `None`, because there is
none.

!!! note "`cylinder` is a verification dataset, not a fitting exercise"
    `cyl_400_20.txt` is a noise-free calculation: evaluating the model at its
    truth reproduces all 20 points exactly, which makes it the right thing to
    check the model pipeline against. It cannot be *fitted*, though — it has no
    dI column, so the bumps engine refuses it, and its intensity spans five
    decades unweighted, which the scipy engine cannot descend. For a cylinder
    you can actually fit, use `simulate('cylinder', radius=20, length=400)`.

## Simulated data

`examples.simulate()` computes a dataset from any sasmodels model and attaches
the generating parameters as `data.truth`:

```python
data = examples.simulate('sphere', radius=50, noise=0.02, seed=0)
data.truth['radius']    # 50.0
```

This is often the better teaching tool: you can state the answer up front and
have the reader check that the fit recovers it. It works for any of the ~100
sasmodels models, needs no files, and lets you dial noise, Q range and
resolution independently.

```python
# wider Q range, more points, 5% noise
examples.simulate('cylinder', radius=20, length=400,
                  qmin=0.001, qmax=0.7, npoints=200, noise=0.05)

# with instrument resolution — the intensity is smeared, not just labelled
examples.simulate('sphere', radius=50, dq=0.05)

# polydisperse; the _pd_n/_pd_type companions are filled in for you
examples.simulate('sphere', radius=50, radius_pd=0.15)

# onto the Q grid of a real dataset
real = examples.load('silica_spheres')
examples.simulate('sphere', radius=100, q=real.x)
```

### How the noise is generated

Uncertainties follow counting statistics, scaled so a point at the median
intensity gets exactly the requested relative error:

```
dI = noise * sqrt(|I| * median(|I|))
```

The scatter is drawn from that same width, so the error bars honestly describe
the noise and reduced χ² lands near 1 for a correct model.

A purely relative `dI = noise * I` would be simpler but is wrong here: it drives
the uncertainty to zero inside the form-factor minima, where the intensity
itself goes to zero. Those points then carry runaway weight and pull the fit
away from the truth — measurably so, a simulated 50 Å sphere recovers as 88 Å.
Counting statistics keep the *absolute* uncertainty from collapsing while
letting the *relative* uncertainty grow in the dim minima and the high-Q tail,
which is what real SANS data does.

### Sample and background pairs

`simulate_pair()` returns a matched sample and background on an identical Q
grid, which is what [`data_ops`](usage.md) requires:

```python
from sans_fitter import SANSFitter, data_ops, examples

sample, background = examples.simulate_pair('sphere', radius=50,
                                            background_level=0.5)
subtracted = data_ops.subtract(sample, background)

fitter = SANSFitter()
fitter.set_data(subtracted)
fitter.set_model('sphere')
fitter.set_param('radius', value=30, min=5, max=300)
result = fitter.fit()      # recovers radius ≈ 50
```

## If the bundled files are missing

`load()` raises `FileNotFoundError` naming the expected location if your
`sasdata` build excludes its example data or has moved it. `simulate()` needs no
files at all and is unaffected.

## API

::: sans_fitter.examples
    options:
      show_root_heading: true
      show_source: true
      members:
        - list_examples
        - describe
        - get_example
        - example_path
        - load
        - load_fitter
        - simulate
        - simulate_pair
        - Example
