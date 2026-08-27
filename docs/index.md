# SANS Model Fitter

A flexible, model-agnostic Python template for fitting Small-Angle Neutron Scattering (SANS) data using the SasModels library.

## Overview

SANS Fitter provides a unified interface for fitting SANS data using different optimization engines (BUMPS, LMFit) with any model from the SasModels library. It is designed to be flexible and easy to use, allowing researchers to focus on data analysis rather than writing fitting code.

## Key Features

- **Model-Agnostic Design**: Works with any model from the SasModels library (cylinder, sphere, core_shell, etc.)
- **Custom Models**: Loads your own SasModels plugin models by path or `custom.` prefix (see [Custom Models](custom_models.md))
- **Multiple Fitting Engines**: Supports both BUMPS (default) and LMFit optimization engines
- **Flexible Data Loading**: Reads CSV, XML, and HDF5 formats via sasdata
- **User-Friendly Parameter Management**: Easy-to-use interface for setting parameter values, bounds, and fitting flags
- **Comprehensive Visualization**: Automatic plotting of data, fitted model, and residuals
- **Result Export**: Save fitted parameters and curves to CSV files

## Quick Example

```python
from sans_fitter import SANSFitter

# Create fitter instance
fitter = SANSFitter()

# Load your data
fitter.load_data('my_sans_data.csv')

# Set the model
fitter.set_model('cylinder')

# Configure parameters
fitter.set_param('radius', value=20, min=1, max=100, vary=True)

# Perform the fit
result = fitter.fit()

# Visualize results
fitter.plot_results()
```

## License

This project is licensed under the BSD 3-Clause License.
