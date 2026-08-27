# Custom Models

SANS Fitter is not limited to the models shipped with SasModels. `set_model()` passes the
model string straight to `sasmodels.core.load_model()` without checking it against the
built-in list, so any SasModels *plugin model* you write can be loaded and fitted exactly
like a built-in one.

## Writing a Plugin Model

A plugin model is a plain Python file that declares the model metadata, its parameters, and
a scattering function `Iq(q, ...)`.

```python
# my_power_law.py
import numpy as np
from numpy import inf

name = "my_power_law"
title = "Custom power law"
description = "I(q) = scale * q^-power + background"
category = "shape-independent"

# name, units, default, [min, max], type, description
parameters = [
    ["power", "", 4.0, [-inf, inf], "", "Power law exponent"],
]

def Iq(q, power):
    return q**-power

Iq.vectorized = True
```

Notes:

-   The arguments of `Iq` must match the parameter names, in the order they appear in
    `parameters`.
-   `scale` and `background` are added automatically by SasModels — do not list them in
    `parameters`.
-   `Iq.vectorized = True` tells SasModels that `Iq` accepts the whole `q` array at once.
    Omit it if your function handles one `q` value at a time.

See the [SasModels plugin documentation](https://www.sasview.org/docs/user/qtgui/Perspectives/Fitting/plugin.html)
for the full specification, including 2D models (`Iqxy`), `form_volume`, and polydispersity
support.

## Loading a Custom Model

There are three ways to reach your file. All of them are just a string passed to
`set_model()`.

### By file path

Any string ending in `.py` is treated as a path to a plugin file.

```python
from sans_fitter import SANSFitter

fitter = SANSFitter()
fitter.set_model('path/to/my_power_law.py')
```

### With the `custom.` prefix

Place the file in the SasModels custom-model directory
(`~/.sasmodels/custom_models/` on Linux/macOS, `C:\Users\<you>\.sasmodels\custom_models\`
on Windows) and refer to it by file name:

```python
fitter.set_model('custom.MyPowerLaw')     # loads MyPowerLaw.py
```

### Via `SAS_MODELPATH`

If the `SAS_MODELPATH` environment variable points at a directory of plugin files, a bare
model name resolves against it:

```bash
export SAS_MODELPATH=/path/to/my/models
```

```python
fitter.set_model('my_power_law')
```

## Fitting a Custom Model

Nothing else changes — parameter configuration, fitting, plotting, and result export all
work as usual:

```python
fitter = SANSFitter()
fitter.load_data('my_sans_data.csv')
fitter.set_model('my_power_law.py')

fitter.set_param('power', value=3.0, min=1, max=6, vary=True)
fitter.set_param('scale', value=1e-3, min=1e-6, max=1, vary=True)
fitter.set_param('background', value=0.01, min=0, max=1, vary=True)

result = fitter.fit(engine='bumps')
fitter.plot_results()
```

## C Kernel Models

For performance-critical models you can supply a C kernel instead of a Python `Iq`. The
loading routes above are unchanged; only the file contents differ:

```python
name = "my_fast_model"
parameters = [...]
source = ["my_fast_model.c"]     # C file sitting next to the .py file
```

## Limitations

!!! note "Custom models are not listed by `get_all_models()`"

    `get_all_models()` calls `sasmodels.core.list_models()`, which only returns the built-in
    models. Your custom model will not appear in that list, nor in any notebook dropdown
    populated from it. Pass the model string to `set_model()` directly.

!!! warning "Structure factors need extra model attributes"

    `apply_structure_factor()` builds the product model as `'<model_name>@<structure_factor>'`,
    which parses correctly for custom models. However, SasModels requires a form factor used
    in a product model to define `form_volume` and an effective radius (`radius_effective`).
    A custom model without them cannot be combined with a structure factor.
