"""
SANS Data Analysis Web Application

A Streamlit-based web application for Small Angle Neutron Scattering (SANS) data analysis.
Features include data upload, model selection (manual and AI-assisted), parameter fitting,
and interactive visualization.
"""

import os
import tempfile
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sasmodels import core
from sasmodels.direct_model import DirectModel

from sans_fitter import SANSFitter

# Maximum value that Streamlit's number_input can handle
MAX_FLOAT_DISPLAY = 1e300
MIN_FLOAT_DISPLAY = -1e300


def clamp_for_display(value: float) -> float:
    """
    Clamp a value to a range that Streamlit's number_input can handle.
    Converts inf/-inf to displayable bounds.
    
    Args:
        value: The value to clamp
        
    Returns:
        The clamped value
    """
    if np.isinf(value):
        return MAX_FLOAT_DISPLAY if value > 0 else MIN_FLOAT_DISPLAY
    return value


def get_all_models() -> List[str]:
    """
    Fetch all available models from sasmodels.

    Returns:
        List of model names
    """
    try:
        all_models = core.list_models()
        return sorted(all_models)
    except Exception as e:
        st.error(f'Error fetching models: {str(e)}')
        return []


def analyze_data_for_ai_suggestion(q_data: np.ndarray, i_data: np.ndarray) -> str:
    """
    Analyze SANS data to create a description for AI model suggestion.

    Args:
        q_data: Q values (scattering vector)
        i_data: Intensity values

    Returns:
        String description of the data characteristics
    """
    # Calculate key features
    log_i = np.log10(i_data + 1e-10)  # Avoid log(0)
    log_q = np.log10(q_data + 1e-10)

    # Slope in log-log space (power law exponent)
    slope = np.polyfit(log_q, log_i, 1)[0]

    # Intensity ratio (high Q to low Q)
    low_q_intensity = np.mean(i_data[:len(i_data) // 10])
    high_q_intensity = np.mean(i_data[-len(i_data) // 10 :])
    intensity_ratio = low_q_intensity / (high_q_intensity + 1e-10)

    # Q range
    q_min, q_max = q_data.min(), q_data.max()

    description = f"""Data Analysis:
- Q range: {q_min:.4f} to {q_max:.4f} Å⁻¹
- Power law slope: {slope:.2f}
- Intensity decay: {intensity_ratio:.1f}x from low to high Q
- Data points: {len(q_data)}
"""
    return description


def suggest_models_simple(q_data: np.ndarray, i_data: np.ndarray) -> List[str]:
    """
    Simple heuristic-based model suggestion.

    This is a placeholder for AI-based suggestion. Based on data characteristics,
    suggests appropriate SANS models.

    Args:
        q_data: Q values
        i_data: Intensity values

    Returns:
        List of suggested model names
    """
    log_i = np.log10(i_data + 1e-10)
    log_q = np.log10(q_data + 1e-10)

    # Calculate slope
    slope = np.polyfit(log_q, log_i, 1)[0]

    suggestions = []

    # Heuristic rules based on slope and shape
    if slope < -3.5:
        # Steep decay - likely spherical particles
        suggestions = ['sphere', 'core_shell_sphere', 'fuzzy_sphere']
    elif -3.5 <= slope < -2:
        # Moderate decay - could be cylindrical or ellipsoidal
        suggestions = ['cylinder', 'ellipsoid', 'core_shell_cylinder']
    elif -2 <= slope < -1:
        # Gentle decay - possibly flat structures or aggregates
        suggestions = ['parallelepiped', 'lamellar', 'flexible_cylinder']
    else:
        # Flat or increasing - unusual, suggest common models
        suggestions = ['sphere', 'cylinder', 'ellipsoid']

    return suggestions[:5]  # Return top 5 suggestions


def suggest_models_ai(
    q_data: np.ndarray, i_data: np.ndarray, api_key: Optional[str] = None
) -> List[str]:
    """
    AI-powered model suggestion using OpenAI API.

    Args:
        q_data: Q values
        i_data: Intensity values
        api_key: OpenAI API key

    Returns:
        List of suggested model names
    """
    if not api_key:
        st.warning('No API key provided. Using simple heuristic suggestion instead.')
        return suggest_models_simple(q_data, i_data)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        # Get all available models
        all_models = get_all_models()

        # Create data description
        data_description = analyze_data_for_ai_suggestion(q_data, i_data)

        # Create prompt
#         prompt = f"""You are a SANS (Small Angle Neutron Scattering) data analysis expert.
# Analyze the following SANS data characteristics and suggest 3 most appropriate models
# from the sasmodels library.

# {data_description}

# Available models include: {', '.join(all_models[:50])}... (and more)

# Common model types:
# - Spherical: sphere, core_shell_sphere, fuzzy_sphere
# - Cylindrical: cylinder, core_shell_cylinder, flexible_cylinder
# - Ellipsoidal: ellipsoid, core_shell_ellipsoid
# - Lamellar: lamellar, core_shell_lamellar
# - Complex: parallelepiped, pringle, fractal

# Based on the data characteristics (slope, Q range, intensity decay), suggest 3 models
# that would fit the provided data. Return ONLY the model names, one per line, no explanations."""

        prompt = f"""You are a SANS (Small Angle Neutron Scattering) data analysis expert.
Analyze the following SANS data and suggest 3 most appropriate models
from the sasmodels library.

The data:
Q (Å⁻¹), I(Q) (cm⁻¹)

{chr(10).join([f"{q_data[i]:.6f}, {i_data[i]:.6f}" for i in range(len(q_data))])}

Data description:

{data_description}

Available models include all models in the sasmodels library.

Based on the data characteristics (slope, Q range, intensity decay), suggest 3 models
that would fit the provided data. Return ONLY the model names, one per line, no explanations."""

        response = client.chat.completions.create(
            model='gpt-4o',
            max_tokens=500,
            messages=[{'role': 'user', 'content': prompt}],
        )

        # Parse response
        suggestions = []
        response_text = response.choices[0].message.content
        for line in response_text.strip().split('\n'):
            model_name = line.strip().lower()
            # Remove numbering, bullets, etc.
            model_name = model_name.lstrip('0123456789.-• ')
            if model_name in all_models:
                suggestions.append(model_name)

        return suggestions if suggestions else suggest_models_simple(q_data, i_data)

    except Exception as e:
        st.warning(f'AI suggestion failed: {str(e)}. Using simple heuristic instead.')
        return suggest_models_simple(q_data, i_data)


def plot_data_and_fit(
    fitter: SANSFitter, show_fit: bool = False, fit_q: Optional[np.ndarray] = None, fit_i: Optional[np.ndarray] = None
) -> go.Figure:
    """
    Create an interactive Plotly figure with data and optionally fitted curve.

    Args:
        fitter: SANSFitter instance with loaded data
        show_fit: Whether to show fitted curve
        fit_q: Q values for fitted curve
        fit_i: Intensity values for fitted curve

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Plot original data with error bars
    fig.add_trace(
        go.Scatter(
            x=fitter.data.x,
            y=fitter.data.y,
            error_y=dict(type='data', array=fitter.data.dy, visible=True),
            mode='markers',
            name='Data',
            marker=dict(size=6, color='blue', symbol='circle'),
        )
    )

    # Plot fitted curve if available
    if show_fit and fit_q is not None and fit_i is not None:
        fig.add_trace(
            go.Scatter(
                x=fit_q,
                y=fit_i,
                mode='lines',
                name='Fitted Model',
                line=dict(color='red', width=2),
            )
        )

    # Update layout
    fig.update_layout(
        title='SANS Data Analysis',
        xaxis_title='Q (Å⁻¹)',
        yaxis_title='Intensity (cm⁻¹)',
        xaxis_type='log',
        yaxis_type='log',
        hovermode='closest',
        template='plotly_white',
        height=600,
        showlegend=True,
    )

    return fig


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title='SANS Data Analysis',
        page_icon='🔬',
        layout='wide',
        initial_sidebar_state='expanded',
    )

    st.title('🔬 SANS Data Analysis Web Application')
    st.markdown(
        """
    Analyze Small Angle Neutron Scattering (SANS) data with model fitting and AI-assisted model selection.
    """
    )

    # Initialize session state
    if 'fitter' not in st.session_state:
        st.session_state.fitter = SANSFitter()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'model_selected' not in st.session_state:
        st.session_state.model_selected = False
    if 'fit_completed' not in st.session_state:
        st.session_state.fit_completed = False

    # Sidebar for controls
    st.sidebar.header('Data Upload')

    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        'Upload SANS data file (CSV or .dat)',
        type=['csv', 'dat'],
        help='File should contain columns: Q, I(Q), dI(Q)',
    )

    # Example data button
    if st.sidebar.button('Load Example Data'):
        example_file = 'simulated_sans_data.csv'
        if os.path.exists(example_file):
            try:
                st.session_state.fitter.load_data(example_file)
                st.session_state.data_loaded = True
                st.sidebar.success('✓ Example data loaded successfully!')
            except Exception as e:
                st.sidebar.error(f'Error loading example data: {str(e)}')
        else:
            st.sidebar.error('Example data file not found!')

    # Process uploaded file
    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # Load data
            st.session_state.fitter.load_data(tmp_file_path)
            st.session_state.data_loaded = True
            st.sidebar.success('✓ Data uploaded successfully!')

            # Clean up temp file
            os.unlink(tmp_file_path)

        except Exception as e:
            st.sidebar.error(f'Error loading data: {str(e)}')
            st.session_state.data_loaded = False

    # Main content area
    if not st.session_state.data_loaded:
        st.info('👆 Please upload a SANS data file or load example data from the sidebar.')
        st.markdown(
            """
        ### Expected Data Format

        Your data file should be a CSV or .dat file with three columns:
        - **Q**: Scattering vector (Å⁻¹)
        - **I(Q)**: Intensity (cm⁻¹)
        - **dI(Q)**: Error/uncertainty in intensity

        Example:
        ```
        Q,I,dI
        0.001,1.035,0.020
        0.006,0.990,0.020
        0.011,1.038,0.020
        ...
        ```
        """
        )
        return

    # Show data preview
    st.subheader('📊 Data Preview')
    col1, col2 = st.columns([2, 1])

    with col1:
        # Plot data
        fig = plot_data_and_fit(st.session_state.fitter, show_fit=False)
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown('**Data Statistics**')
        data = st.session_state.fitter.data
        st.metric('Data Points', len(data.x))
        st.metric('Q Range', f'{data.x.min():.4f} - {data.x.max():.4f} Å⁻¹')
        st.metric('Max Intensity', f'{data.y.max():.4e} cm⁻¹')

        # Show data table
        if st.checkbox('Show data table'):
            df = pd.DataFrame({'Q': data.x, 'I(Q)': data.y, 'dI(Q)': data.dy})
            st.dataframe(df.head(20), height=300)

    # Model Selection
    st.sidebar.header('Model Selection')

    selection_method = st.sidebar.radio(
        'Selection Method', ['Manual', 'AI-Assisted'], help='Choose how to select the fitting model'
    )

    selected_model = None

    if selection_method == 'Manual':
        all_models = get_all_models()
        selected_model = st.sidebar.selectbox(
            'Select Model',
            options=all_models,
            index=all_models.index('sphere') if 'sphere' in all_models else 0,
            help='Choose a model from the sasmodels library',
        )

    else:  # AI-Assisted
        st.sidebar.markdown('**AI-Assisted Model Suggestion**')

        # API key input (optional)
        api_key = st.sidebar.text_input(
            'OpenAI API Key (optional)',
            type='password',
            help='Enter your OpenAI API key for AI-powered suggestions. Leave empty for heuristic-based suggestions.',
        )

        if st.sidebar.button('Get AI Suggestions'):
            with st.spinner('Analyzing data...'):
                data = st.session_state.fitter.data
                suggestions = suggest_models_ai(data.x, data.y, api_key if api_key else None)

                if suggestions:
                    st.sidebar.success(f'✓ Found {len(suggestions)} suggestions')
                    st.session_state.ai_suggestions = suggestions
                else:
                    st.sidebar.warning('No suggestions found')

        # Show suggestions if available
        if 'ai_suggestions' in st.session_state and st.session_state.ai_suggestions:
            st.sidebar.markdown('**Suggested Models:**')
            selected_model = st.sidebar.selectbox(
                'Choose from suggestions', options=st.session_state.ai_suggestions
            )

    # Load selected model
    if selected_model:
        if st.sidebar.button('Load Model'):
            try:
                st.session_state.fitter.set_model(selected_model)
                st.session_state.model_selected = True
                st.session_state.current_model = selected_model
                st.session_state.fit_completed = False  # Reset fit status
                st.sidebar.success(f'✓ Model "{selected_model}" loaded!')
            except Exception as e:
                st.sidebar.error(f'Error loading model: {str(e)}')

    # Parameter Configuration
    if st.session_state.model_selected:
        st.subheader(f'⚙️ Model Parameters: {st.session_state.current_model}')

        fitter = st.session_state.fitter
        params = fitter.params

        # Apply pending preset action before widgets are rendered
        if 'pending_preset' in st.session_state:
            preset = st.session_state.pending_preset
            del st.session_state.pending_preset
            
            for param_name in params.keys():
                if preset == 'scale_background':
                    vary = param_name in ('scale', 'background')
                elif preset == 'fit_all':
                    vary = True
                elif preset == 'fix_all':
                    vary = False
                else:
                    vary = False
                fitter.set_param(param_name, vary=vary)

        st.markdown(
            """
        Configure the model parameters below. Set initial values, bounds, and whether each parameter
        should be fitted (vary) or held constant.
        """
        )

        # Create parameter configuration UI
        param_cols = st.columns([2, 1, 1, 1, 1])
        param_cols[0].markdown('**Parameter**')
        param_cols[1].markdown('**Value**')
        param_cols[2].markdown('**Min**')
        param_cols[3].markdown('**Max**')
        param_cols[4].markdown('**Fit?**')

        # Store parameter updates
        param_updates = {}

        for param_name, param_info in params.items():
            cols = st.columns([2, 1, 1, 1, 1])

            # Parameter name and description
            with cols[0]:
                st.text(param_name)
                if param_info.get('description'):
                    st.caption(param_info['description'][:50])

            # Value input
            with cols[1]:
                value = st.number_input(
                    'Value',
                    value=clamp_for_display(float(param_info['value'])),
                    format='%g',
                    key=f'value_{param_name}',
                    label_visibility='collapsed',
                )

            # Min bound
            with cols[2]:
                min_val = st.number_input(
                    'Min',
                    value=clamp_for_display(float(param_info['min'])),
                    format='%g',
                    key=f'min_{param_name}',
                    label_visibility='collapsed',
                )

            # Max bound
            with cols[3]:
                max_val = st.number_input(
                    'Max',
                    value=clamp_for_display(float(param_info['max'])),
                    format='%g',
                    key=f'max_{param_name}',
                    label_visibility='collapsed',
                )

            # Vary checkbox
            with cols[4]:
                vary = st.checkbox(
                    'Fit',
                    value=param_info['vary'],
                    key=f'vary_{param_name}',
                    label_visibility='collapsed',
                )

            # Store updates
            param_updates[param_name] = {
                'value': value,
                'min': min_val,
                'max': max_val,
                'vary': vary,
            }

        # Apply parameter updates
        if st.button('Update Parameters'):
            for param_name, updates in param_updates.items():
                fitter.set_param(
                    param_name,
                    value=updates['value'],
                    min=updates['min'],
                    max=updates['max'],
                    vary=updates['vary'],
                )
            st.success('✓ Parameters updated!')

        # Quick parameter presets
        st.markdown('**Quick Presets:**')
        preset_cols = st.columns(4)

        with preset_cols[0]:
            if st.button('Fit Scale & Background'):
                st.session_state.pending_preset = 'scale_background'
                st.rerun()

        with preset_cols[1]:
            if st.button('Fit All Parameters'):
                st.session_state.pending_preset = 'fit_all'
                st.rerun()

        with preset_cols[2]:
            if st.button('Fix All Parameters'):
                st.session_state.pending_preset = 'fix_all'
                st.rerun()

        # Fitting Section
        st.sidebar.header('Fitting')

        engine = st.sidebar.selectbox(
            'Optimization Engine', ['bumps', 'lmfit'], help='Choose the fitting engine'
        )

        if engine == 'bumps':
            method = st.sidebar.selectbox(
                'Method',
                ['amoeba', 'lm', 'newton', 'de'],
                help='Optimization method for BUMPS',
            )
        else:
            method = st.sidebar.selectbox(
                'Method',
                ['leastsq', 'least_squares', 'differential_evolution'],
                help='Optimization method for LMFit',
            )

        if st.sidebar.button('🚀 Run Fit', type='primary'):
            # Apply current parameter settings before fitting
            for param_name, updates in param_updates.items():
                fitter.set_param(
                    param_name,
                    value=updates['value'],
                    min=updates['min'],
                    max=updates['max'],
                    vary=updates['vary'],
                )
            
            with st.spinner(f'Fitting with {engine}/{method}...'):
                try:
                    # Check if any parameters are set to vary
                    any_vary = any(p['vary'] for p in fitter.params.values())
                    if not any_vary:
                        st.warning('⚠️ No parameters are set to vary. Please enable at least one parameter to fit.')
                    else:
                        result = fitter.fit(engine=engine, method=method)
                        st.session_state.fit_completed = True
                        st.session_state.fit_result = result
                        st.sidebar.success('✓ Fit completed successfully!')
                except Exception as e:
                    st.sidebar.error(f'Fitting error: {str(e)}')

        # Display fit results
        if st.session_state.fit_completed:
            st.subheader('📈 Fit Results')

            # Get fitted parameters
            col1, col2 = st.columns([2, 1])

            with col1:
                # Plot data with fit
                try:
                    # Get current parameter values
                    param_values = {name: info['value'] for name, info in fitter.params.items()}

                    # Calculate model using DirectModel (uses data's Q values)
                    calculator = DirectModel(fitter.data, fitter.kernel)
                    fit_i = calculator(**param_values)

                    # Use data's Q values for the fitted curve
                    q_plot = fitter.data.x

                    # Plot
                    fig = plot_data_and_fit(fitter, show_fit=True, fit_q=q_plot, fit_i=fit_i)
                    st.plotly_chart(fig, width='stretch')

                except Exception as e:
                    st.error(f'Error plotting results: {str(e)}')

            with col2:
                st.markdown('**Fitted Parameters**')

                # Display fitted parameters in a table
                fitted_params = []
                for name, info in fitter.params.items():
                    if info['vary']:
                        fitted_params.append({'Parameter': name, 'Value': f"{info['value']:.4g}"})

                if fitted_params:
                    df_fitted = pd.DataFrame(fitted_params)
                    st.dataframe(df_fitted, hide_index=True, width='stretch')
                else:
                    st.info('No parameters were fitted')

                # Export results
                st.markdown('**Export Results**')
                if st.button('Save Results to CSV'):
                    try:
                        # Create results dataframe
                        results_data = []
                        for name, info in fitter.params.items():
                            results_data.append(
                                {
                                    'Parameter': name,
                                    'Value': info['value'],
                                    'Min': info['min'],
                                    'Max': info['max'],
                                    'Fitted': info['vary'],
                                }
                            )

                        df_results = pd.DataFrame(results_data)

                        # Convert to CSV
                        csv = df_results.to_csv(index=False)

                        # Download button
                        st.download_button(
                            label='Download CSV',
                            data=csv,
                            file_name='fit_results.csv',
                            mime='text/csv',
                        )
                    except Exception as e:
                        st.error(f'Error saving results: {str(e)}')


if __name__ == '__main__':
    main()
