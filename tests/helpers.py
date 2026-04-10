import os
import sys
import tempfile

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def create_loading_test_data_file(num_points=10):
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    temp_file.write('Q,I,dI\n')
    for index in range(num_points):
        q_value = 0.01 * (index + 1)
        intensity = 100 * np.exp(-q_value * 10) + 0.1
        error = intensity * 0.1
        temp_file.write(f'{q_value},{intensity},{error}\n')
    temp_file.close()
    return temp_file.name


def create_decay_data_file(num_points=30):
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    temp_file.write('Q,I,dI\n')

    q_values = np.logspace(-2, 0, num_points)
    intensity = 0.1 * (1 / (1 + q_values**2)) + 0.01
    d_intensity = intensity * 0.1

    for q_value, intensity_value, d_intensity_value in zip(q_values, intensity, d_intensity):
        temp_file.write(f'{q_value},{intensity_value},{d_intensity_value}\n')

    temp_file.close()
    return temp_file.name


def create_concentrated_sphere_data_file(num_points=30):
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    temp_file.write('Q,I,dI\n')

    q_values = np.logspace(-2, 0, num_points)
    intensity = (
        0.01
        * (1 / (1 + (q_values * 50) ** 2))
        * (1 - 0.2 * np.sin(q_values * 100) / (q_values * 100 + 1e-10))
    )
    intensity = np.maximum(intensity, 0.001)
    d_intensity = intensity * 0.1

    for q_value, intensity_value, d_intensity_value in zip(q_values, intensity, d_intensity):
        temp_file.write(f'{q_value},{intensity_value},{d_intensity_value}\n')

    temp_file.close()
    return temp_file.name
