# Quick Start Guide - SANS Data Analysis Web App

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/rozyczko/SANS-fitter.git
cd SANS-fitter
```

### 2. Install Dependencies
```bash
# Install the SANS-fitter package with web app dependencies
pip install -e ".[web]"
```

### 3. Run the Application
```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`.

## Quick Demo

To see a demonstration of the app's capabilities without opening a browser:

```bash
python demo_app.py
```

This will show:
- Model selection
- Data loading
- AI-assisted suggestions
- Parameter configuration
- Fitting process
- Results visualization

## Testing

Run the test suite to verify functionality:

```bash
python test_app.py
```

## File Structure

```
├── app.py                      # Main Streamlit application
├── pyproject.toml             # Package config with [web] optional deps
├── demo_app.py                # Command-line demo
├── test_app.py                # Test suite
├── simulated_sans_data.csv    # Example dataset (200 points)
├── example_sans_data.dat      # Alternative example (70 points)
├── Dockerfile                 # For Docker deployment
├── Procfile                   # For Heroku deployment
├── setup.sh                   # Heroku setup script
├── WEBAPP_README.md          # Detailed web app documentation
└── src/sans_fitter/          # Core fitting library
    ├── __init__.py
    └── sans_fitter.py
```

## Usage Workflow

### 1. Upload Data
- Click "Browse files" in sidebar
- Or click "Load Example Data" button
- Supported formats: CSV, .dat
- Required columns: Q, I(Q), dI(Q)

### 2. Select Model
**Manual Mode:**
- Choose from 79+ models in dropdown
- Click "Load Model"

**AI-Assisted Mode:**
- Click "Get AI Suggestions"
- Optional: Enter OpenAI API key for enhanced suggestions
- Select from suggested models
- Click "Load Model"

### 3. Configure Parameters
- Set initial values, bounds (min/max)
- Check "Fit?" to vary parameter during optimization
- Use quick presets:
  - "Fit Scale & Background" - Common starting point
  - "Fit All Parameters" - Full optimization
  - "Fix All Parameters" - Reset all to fixed
- Click "Update Parameters"

### 4. Run Fit
- Select engine: BUMPS (recommended) or LMFit
- Choose method: 
  - BUMPS: amoeba, lm, newton, de
  - LMFit: leastsq, least_squares, differential_evolution
- Click "🚀 Run Fit"

### 5. View Results
- Interactive Plotly plot with:
  - Data points (with error bars)
  - Fitted curve overlay
  - Zoom, pan, export tools
- Fitted parameters table
- Download results as CSV

## Common Models

| Category | Models |
|----------|--------|
| **Spherical** | sphere, core_shell_sphere, fuzzy_sphere, vesicle |
| **Cylindrical** | cylinder, core_shell_cylinder, flexible_cylinder |
| **Ellipsoidal** | ellipsoid, core_shell_ellipsoid, triaxial_ellipsoid |
| **Flat** | lamellar, parallelepiped, core_shell_lamellar |
| **Complex** | fractal, pearl_necklace, pringle |

## Deployment Options

### Streamlit Cloud (Free)
1. Push to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Deploy with one click

### Docker
```bash
docker build -t sans-app .
docker run -p 8501:8501 sans-app
```

### Heroku
```bash
heroku create my-sans-app
git push heroku main
```

## API Key (Optional)

For enhanced AI model suggestions, get an API key from [OpenAI](https://platform.openai.com):

1. Create account at platform.openai.com
2. Generate API key
3. Enter in sidebar when using AI-Assisted mode

**Note**: The app works without an API key using built-in heuristics.

## Troubleshooting

**Data won't load?**
- Check file has columns: Q, I, dI (or similar)
- Ensure no missing values
- Try example data first

**Fit fails?**
- At least one parameter must have "Fit?" checked
- Check parameter bounds (min < value < max)
- Try "amoeba" method first (most robust)
- Start with fewer varying parameters

**Slow performance?**
- Large datasets (>1000 points) take longer
- Try downsampling data
- Use faster methods (amoeba vs de)

## Support

- **Documentation**: See [README.md](README.md) and [WEBAPP_README.md](WEBAPP_README.md)
- **Issues**: [GitHub Issues](https://github.com/rozyczko/SANS-fitter/issues)
- **Examples**: Check `examples/` directory

## License

BSD 3-Clause License - See [LICENSE](LICENSE)
