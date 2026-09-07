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


def create_loading_test_data_file_with_resolution(num_points=10):
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    temp_file.write('Q,I,dI,dQ\n')
    for index in range(num_points):
        q_value = 0.01 * (index + 1)
        intensity = 100 * np.exp(-q_value * 10) + 0.1
        error = intensity * 0.1
        resolution = q_value * 0.05
        temp_file.write(f'{q_value},{intensity},{error},{resolution}\n')
    temp_file.close()
    return temp_file.name


def create_multi_dataset_xml_file(num_points=3):
    """Write a CanSAS XML containing two datasets and return its path.

    Dataset 0 has title "alpha sample" and run id "alpha_run"; dataset 1 has
    title "beta sample" and run id "beta_run". Both share the same Q grid.
    """
    points = '\n'.join(
        f'<Idata><Q unit="1/A">{q}</Q><I unit="1/cm">{i}</I><Idev unit="1/cm">{di}</Idev></Idata>'
        for q, i, di in (
            ('0.01', '100', '10'),
            ('0.02', '50', '5'),
            ('0.03', '20', '2'),
        )[:num_points]
    )
    xml = f"""<?xml version="1.0"?>
<SASroot version="1.0"
    xmlns="cansas1d/1.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="cansas1d/1.0 http://svn.smallangles.net/svn/canSAS/1dwg/trunk/cansas1d.xsd">
<SASentry>
  <Title>alpha sample</Title>
  <Run>alpha_run</Run>
  <SASdata>
    {points}
  </SASdata>
</SASentry>
<SASentry>
  <Title>beta sample</Title>
  <Run>beta_run</Run>
  <SASdata>
    {points}
  </SASdata>
</SASentry>
</SASroot>
"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
    temp_file.write(xml)
    temp_file.close()
    return temp_file.name


def create_background_data_file(num_points=10):
    """Flat background on the same Q grid as create_loading_test_data_file."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    temp_file.write('Q,I,dI\n')
    for index in range(num_points):
        q_value = 0.01 * (index + 1)
        temp_file.write(f'{q_value},0.5,0.05\n')
    temp_file.close()
    return temp_file.name


def create_offset_grid_data_file(num_points=12):
    """Dataset on a Q grid that does not match create_loading_test_data_file."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    temp_file.write('Q,I,dI\n')
    for index in range(num_points):
        q_value = 0.017 * (index + 1)
        intensity = 50 * np.exp(-q_value * 8) + 0.2
        temp_file.write(f'{q_value},{intensity},{intensity * 0.1}\n')
    temp_file.close()
    return temp_file.name


def create_decay_data_file(num_points=30):
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    temp_file.write('Q,I,dI\n')

    q_values = np.logspace(-2, 0, num_points)
    intensity = 0.1 * (1 / (1 + q_values**2)) + 0.01
    d_intensity = intensity * 0.1

    for q_value, intensity_value, d_intensity_value in zip(
        q_values, intensity, d_intensity, strict=True
    ):
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

    for q_value, intensity_value, d_intensity_value in zip(
        q_values, intensity, d_intensity, strict=True
    ):
        temp_file.write(f'{q_value},{intensity_value},{d_intensity_value}\n')

    temp_file.close()
    return temp_file.name
