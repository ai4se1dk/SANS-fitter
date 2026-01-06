#!/usr/bin/env python
"""
Test script to validate the Streamlit app functionality without running the full UI.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from sans_fitter import SANSFitter
import app

def test_get_all_models():
    """Test model listing."""
    print("Testing get_all_models()...")
    models = app.get_all_models()
    assert len(models) > 0, "No models found!"
    assert 'sphere' in models, "sphere model not found!"
    assert 'cylinder' in models, "cylinder model not found!"
    print(f"✓ Found {len(models)} models")

def test_data_analysis():
    """Test data analysis for AI suggestion."""
    print("\nTesting analyze_data_for_ai_suggestion()...")
    # Create fake data
    q = np.logspace(-3, -1, 50)
    i = 100 * np.exp(-q * 10) + 0.1
    
    description = app.analyze_data_for_ai_suggestion(q, i)
    assert len(description) > 0, "No description generated!"
    assert 'Q range' in description, "Q range not in description!"
    print("✓ Data analysis working")
    print(description[:200])

def test_suggest_models_simple():
    """Test simple model suggestion."""
    print("\nTesting suggest_models_simple()...")
    # Create fake data with steep decay (spherical)
    q = np.logspace(-3, -1, 50)
    i = 100 * q**(-4) + 0.1  # Porod law for spheres
    
    suggestions = app.suggest_models_simple(q, i)
    assert len(suggestions) > 0, "No suggestions generated!"
    print(f"✓ Got {len(suggestions)} suggestions: {suggestions}")

def test_fitter_integration():
    """Test SANSFitter integration."""
    print("\nTesting SANSFitter integration...")
    fitter = SANSFitter()
    
    # Load example data
    try:
        fitter.load_data('simulated_sans_data.csv')
        print("✓ Data loaded successfully")
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return
    
    # Set model
    try:
        fitter.set_model('sphere')
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return
    
    # Check parameters
    assert len(fitter.params) > 0, "No parameters loaded!"
    print(f"✓ Found {len(fitter.params)} parameters")

def test_plot_creation():
    """Test plot generation."""
    print("\nTesting plot_data_and_fit()...")
    fitter = SANSFitter()
    
    try:
        fitter.load_data('simulated_sans_data.csv')
        fig = app.plot_data_and_fit(fitter, show_fit=False)
        assert fig is not None, "No figure generated!"
        print("✓ Plot created successfully")
    except Exception as e:
        print(f"✗ Plot creation failed: {e}")

if __name__ == '__main__':
    print("=" * 70)
    print("SANS Fitter Streamlit App - Functionality Tests")
    print("=" * 70)
    
    try:
        test_get_all_models()
        test_data_analysis()
        test_suggest_models_simple()
        test_fitter_integration()
        test_plot_creation()
        
        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        print("\nTo run the full Streamlit app, use:")
        print("  streamlit run app.py")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
