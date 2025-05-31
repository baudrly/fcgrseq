# -*- coding: utf-8 -*-
"""
Unit tests for the feature_extraction module.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

# Module containing functions to test
from fcgr_analyzer import feature_extraction as fe
from fcgr_analyzer.config import FCGR_K as CFG_FCGR_K, FCGR_DIM as CFG_FCGR_DIM, EPSILON
from fcgr_analyzer.utils import NUMBA_AVAILABLE

# Fixtures and constants from conftest
from .conftest import sample_fcgr_matrix_k6, sample_fcgr_matrix_k4 # k6 matches default config, k4 for other tests

# Default dimension for tests (matches sample_fcgr_matrix_k6 fixture)
TEST_DIM = CFG_FCGR_DIM


# --- Test calculate_statistical_features ---
def test_stats_basic(sample_fcgr_matrix_k6):
    features = fe.calculate_statistical_features(sample_fcgr_matrix_k6)
    assert isinstance(features, dict)
    stat_keys = ['mean', 'variance', 'skewness', 'kurtosis', 'shannon_entropy']
    assert all(key in features for key in stat_keys)
    assert features['mean'] > 0
    assert features['variance'] > 0
    assert np.isfinite(features['skewness'])
    assert np.isfinite(features['kurtosis'])
    if fe.SKIMAGE_AVAILABLE:
        assert features['shannon_entropy'] > 0
    else:
        assert_almost_equal(features['shannon_entropy'], 0.0, decimal=7)

def test_stats_constant_matrix():
    matrix = np.full((TEST_DIM, TEST_DIM), 5.0, dtype=np.float32)
    features = fe.calculate_statistical_features(matrix)
    assert_almost_equal(features['mean'], 5.0)
    assert_almost_equal(features['variance'], 0.0)
    assert_almost_equal(features['skewness'], 0.0)
    assert_almost_equal(features['kurtosis'], -3.0) # Fisher's definition
    if fe.SKIMAGE_AVAILABLE: # skimage.measure.shannon_entropy handles this well
        # For a constant matrix, probs are 1/(N*M) for each cell. Sum(p log p)
        # For skimage, it takes the image directly. If all values are same, entropy should be 0
        # For normalized prob_matrix where all p_i = 1/N, entropy is log(N)
        # fe.calculate_statistical_features normalizes the matrix as probabilities
        # If matrix sum is > EPSILON, prob_matrix is matrix / sum(matrix)
        # So prob_matrix entries are all 1 / (TEST_DIM*TEST_DIM)
        # entropy = - sum (p_i * log2(p_i)) = - N_cells * (1/N_cells * log2(1/N_cells)) = -log2(1/N_cells) = log2(N_cells)
        expected_entropy = np.log2(TEST_DIM * TEST_DIM)
        assert_almost_equal(features['shannon_entropy'], expected_entropy, decimal=5)
    else:
         assert_almost_equal(features['shannon_entropy'], 0.0, decimal=7) # Default if skimage missing

def test_stats_zero_matrix():
    matrix = np.zeros((TEST_DIM, TEST_DIM), dtype=np.float32)
    features = fe.calculate_statistical_features(matrix)
    assert_almost_equal(features['mean'], 0.0)
    assert_almost_equal(features['variance'], 0.0)
    assert_almost_equal(features['skewness'], 0.0)
    assert_almost_equal(features['kurtosis'], -3.0)
    assert_almost_equal(features['shannon_entropy'], 0.0) # Sum is 0, so entropy is 0


# --- Test calculate_haralick_textures ---
HARALICK_KEYS = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
HARALICK_DEFAULTS = {'contrast': 0.0, 'dissimilarity': 0.0, 'homogeneity': 1.0, 'energy': 1.0, 'correlation': 1.0, 'ASM': 1.0}

@pytest.mark.skipif(not fe.SKIMAGE_AVAILABLE, reason="scikit-image not available")
def test_haralick_basic(sample_fcgr_matrix_k6):
    features = fe.calculate_haralick_textures(sample_fcgr_matrix_k6)
    assert isinstance(features, dict)
    assert all(key in features for key in HARALICK_KEYS)
    assert all(np.isfinite(v) for v in features.values())
    assert features['contrast'] > HARALICK_DEFAULTS['contrast'] + EPSILON
    assert features['homogeneity'] < HARALICK_DEFAULTS['homogeneity'] - EPSILON

@pytest.mark.skipif(not fe.SKIMAGE_AVAILABLE, reason="scikit-image not available")
@pytest.mark.parametrize("matrix_val, expected_features", [
    (5.0, HARALICK_DEFAULTS), # Constant non-zero
    (0.0, HARALICK_DEFAULTS)  # Constant zero
])
def test_haralick_constant_or_zero_matrix(matrix_val, expected_features):
    matrix = np.full((TEST_DIM, TEST_DIM), matrix_val, dtype=np.float32)
    features = fe.calculate_haralick_textures(matrix)
    for key in HARALICK_KEYS:
        assert_almost_equal(features[key], expected_features[key], decimal=6)

def test_haralick_unavailable():
    if fe.SKIMAGE_AVAILABLE: pytest.skip("Skipping as scikit-image is available")
    matrix = np.random.rand(TEST_DIM, TEST_DIM).astype(np.float32)
    features = fe.calculate_haralick_textures(matrix)
    assert features == HARALICK_DEFAULTS # Should return defaults if lib missing


# --- Test calculate_hu_moments ---
HU_KEYS = [f'hu_moment_{i}' for i in range(7)]
HU_DEFAULTS = {key: 0.0 for key in HU_KEYS}

@pytest.mark.skipif(not fe.OPENCV_AVAILABLE, reason="OpenCV (cv2) not available")
def test_hu_moments_basic(sample_fcgr_matrix_k6):
    features = fe.calculate_hu_moments(sample_fcgr_matrix_k6)
    assert isinstance(features, dict)
    assert all(key in features for key in HU_KEYS)
    assert all(np.isfinite(v) for v in features.values())
    # At least one moment should be non-zero for a non-trivial image
    assert any(abs(features[key]) > EPSILON for key in HU_KEYS)

@pytest.mark.skipif(not fe.OPENCV_AVAILABLE, reason="OpenCV (cv2) not available")
@pytest.mark.parametrize("matrix_val, expected_features", [
    (5.0, HU_DEFAULTS), (0.0, HU_DEFAULTS)
])
def test_hu_moments_constant_or_zero_matrix(matrix_val, expected_features):
    matrix = np.full((TEST_DIM, TEST_DIM), matrix_val, dtype=np.float32)
    features = fe.calculate_hu_moments(matrix)
    for key in HU_KEYS:
        assert_almost_equal(features[key], expected_features[key], decimal=6)

def test_hu_moments_unavailable():
    if fe.OPENCV_AVAILABLE: pytest.skip("Skipping as OpenCV is available")
    matrix = np.random.rand(TEST_DIM, TEST_DIM).astype(np.float32)
    features = fe.calculate_hu_moments(matrix)
    assert features == HU_DEFAULTS


# --- Test calculate_fractal_dimension ---
# Helper for simple fractal patterns
def create_simple_fractal(dim, pattern_type="checkerboard"):
    matrix = np.zeros((dim, dim), dtype=np.float32)
    if pattern_type == "checkerboard":
        for r in range(dim):
            for c in range(dim):
                if (r + c) % 2 == 0: matrix[r, c] = 1.0
    elif pattern_type == "hline":
        matrix[dim // 2, :] = 1.0
    elif pattern_type == "vline":
        matrix[:, dim // 2] = 1.0
    elif pattern_type == "solid":
        matrix[:, :] = 1.0
    elif pattern_type == "sparse_points": # Few points, should have FD near 0
        matrix[0,0] = 1.0
        matrix[dim-1, dim-1] = 1.0
    return matrix

@pytest.mark.parametrize("pattern_type, dim_size, expected_fd_range", [
    ("solid", 32, (1.9, 2.0)),        # Solid square
    ("hline", 32, (0.9, 1.1)),        # Horizontal line
    ("vline", 32, (0.9, 1.1)),        # Vertical line
    ("checkerboard", 32, (1.9, 2.0)), # Checkerboard (space-filling at large scale)
    ("sparse_points", 32, (0.0, 0.5)) # Very sparse points
])
def test_fd_patterns(pattern_type, dim_size, expected_fd_range):
    matrix = create_simple_fractal(dim_size, pattern_type)
    fd = fe.calculate_fractal_dimension(matrix)
    assert isinstance(fd, float)
    assert np.isfinite(fd)
    assert expected_fd_range[0] <= fd <= expected_fd_range[1], \
        f"FD for {pattern_type} ({fd:.3f}) not in range {expected_fd_range}"

def test_fd_zero_matrix_fd():
    matrix = np.zeros((32, 32), dtype=np.float32)
    fd = fe.calculate_fractal_dimension(matrix)
    assert_almost_equal(fd, 0.0, decimal=7)

def test_fd_small_matrix():
    matrix = np.ones((4,4)) # Max power log2(4)=2. Scales: 2. Only one scale pair.
    fd = fe.calculate_fractal_dimension(matrix)
    # Regression with < 2 points returns default (0.0)
    assert_almost_equal(fd, 0.0, decimal=7) # Insufficient scales

@pytest.mark.skipif(not NUMBA_AVAILABLE or fe._boxcount_loop_numba is None, reason="Numba or Numba FD func unavailable")
def test_fd_numba_vs_python_fallback(mocker):
    matrix = create_simple_fractal(64, "checkerboard") # Use a larger matrix for better test

    # Call normally (should use Numba if available and compiled)
    fd_numba_attempt = fe.calculate_fractal_dimension(matrix)

    # Force Python fallback by mocking the Numba function to None
    mocker.patch('fcgr_analyzer.feature_extraction._boxcount_loop_numba', None)
    fd_python = fe.calculate_fractal_dimension(matrix)

    assert np.isfinite(fd_numba_attempt)
    assert np.isfinite(fd_python)
    # They might not be exactly equal due to implementation differences, but should be close
    assert_allclose(fd_numba_attempt, fd_python, atol=0.1,
                    err_msg=f"FD mismatch: Numba-like={fd_numba_attempt}, Python={fd_python}")


# --- Test extract_all_features ---
def test_extract_all_features_structure_and_defaults(sample_fcgr_matrix_k6):
    all_features = fe.extract_all_features(sample_fcgr_matrix_k6)
    default_template = fe.extract_all_features(np.zeros((CFG_FCGR_DIM, CFG_FCGR_DIM))) # Get keys and default types

    assert isinstance(all_features, dict)
    assert all_features.keys() == default_template.keys() # Ensure all expected features are present

    for key, value in all_features.items():
        assert isinstance(value, float), f"Feature '{key}' type mismatch (is {type(value)})"
        assert np.isfinite(value), f"Feature '{key}' is not finite ({value})"

def test_extract_all_features_on_zero_matrix():
    matrix = np.zeros((CFG_FCGR_DIM, CFG_FCGR_DIM), dtype=np.float32)
    features = fe.extract_all_features(matrix)
    
    # Check some specific default values
    assert_almost_equal(features['mean'], 0.0)
    assert_almost_equal(features['variance'], 0.0)
    assert_almost_equal(features['kurtosis'], -3.0)
    assert_almost_equal(features['fractal_dimension'], 0.0)
    assert_almost_equal(features['fcgr_entropy'], 0.0)
    assert_almost_equal(features['quadrant_ratio_AA_GC'], 1.0) # EPSILON/EPSILON
    assert_almost_equal(features['center_mass_x'], 0.5) # Default for empty/uniform
    assert_almost_equal(features['center_mass_y'], 0.5)

    if fe.SKIMAGE_AVAILABLE:
        assert_almost_equal(features['homogeneity'], 1.0)
        assert_almost_equal(features['energy'], 1.0)
    if fe.OPENCV_AVAILABLE:
        assert all(abs(features[f'hu_moment_{i}']) < EPSILON for i in range(7))


@pytest.mark.parametrize("invalid_matrix_input", [
    None, np.array([]), np.random.rand(CFG_FCGR_DIM), np.random.rand(CFG_FCGR_DIM, CFG_FCGR_DIM-1)
])
def test_extract_all_features_invalid_inputs(invalid_matrix_input):
    default_values = fe.extract_all_features(np.zeros((CFG_FCGR_DIM, CFG_FCGR_DIM)))
    features = fe.extract_all_features(invalid_matrix_input)
    assert features == default_values

def test_extract_all_features_wrong_dim_vs_config(sample_fcgr_matrix_k4, caplog):
    # sample_fcgr_matrix_k4 is 16x16, default CFG_FCGR_DIM is 64x64
    features = fe.extract_all_features(sample_fcgr_matrix_k4)
    assert "FCGR matrix shape (16, 16) != expected (64x64)" in caplog.text
    # Features should still be calculated, e.g. mean
    assert 'mean' in features
    assert np.isfinite(features['mean'])


# Test new features in extract_all_features
def test_fcgr_entropy_calculation(sample_fcgr_matrix_k6):
    # Assuming calculate_feature_entropy is from .analysis,
    # if it's meant to be local, adjust import.
    # For now, let's test the one in .analysis
    from fcgr_analyzer.analysis import calculate_feature_entropy as an_calc_entropy
    
    matrix_norm = sample_fcgr_matrix_k6 / (np.sum(sample_fcgr_matrix_k6) + EPSILON)
    matrix_norm = np.maximum(matrix_norm, 0) # Ensure non-negative
    
    # Calculate expected entropy manually or using scipy if available
    expected_entropy = an_calc_entropy(matrix_norm) # Using the one from analysis
    
    features = fe.extract_all_features(sample_fcgr_matrix_k6)
    assert_allclose(features['fcgr_entropy'], expected_entropy, atol=1e-5)

def test_quadrant_ratios_and_center_mass(sample_fcgr_matrix_k6):
    features = fe.extract_all_features(sample_fcgr_matrix_k6)
    
    mid = sample_fcgr_matrix_k6.shape[0] // 2
    aa_sum = np.sum(sample_fcgr_matrix_k6[:mid, :mid])
    gc_sum = np.sum(sample_fcgr_matrix_k6[mid:, mid:])
    at_sum = np.sum(sample_fcgr_matrix_k6[:mid, mid:])
    cg_sum = np.sum(sample_fcgr_matrix_k6[mid:, :mid])

    expected_q_aa_gc = (aa_sum + EPSILON) / (gc_sum + EPSILON)
    expected_q_at_cg = (at_sum + EPSILON) / (cg_sum + EPSILON)
    assert_allclose(features['quadrant_ratio_AA_GC'], expected_q_aa_gc, atol=1e-5)
    assert_allclose(features['quadrant_ratio_AT_CG'], expected_q_at_cg, atol=1e-5)

    total = np.sum(sample_fcgr_matrix_k6) + EPSILON
    y_indices, x_indices = np.mgrid[0:sample_fcgr_matrix_k6.shape[0], 0:sample_fcgr_matrix_k6.shape[1]]
    expected_cm_x = np.sum(x_indices * sample_fcgr_matrix_k6) / total / sample_fcgr_matrix_k6.shape[1]
    expected_cm_y = np.sum(y_indices * sample_fcgr_matrix_k6) / total / sample_fcgr_matrix_k6.shape[0]
    assert_allclose(features['center_mass_x'], expected_cm_x, atol=1e-5)
    assert_allclose(features['center_mass_y'], expected_cm_y, atol=1e-5)

def test_quadrant_ratios_zero_denominator():
    matrix = np.zeros((CFG_FCGR_DIM, CFG_FCGR_DIM))
    matrix[0,0] = 1 # Only AA quadrant has value
    features = fe.extract_all_features(matrix)
    # gc_sum, at_sum, cg_sum will be 0. Ratios become (X+EPSILON)/EPSILON or EPSILON/EPSILON
    assert features['quadrant_ratio_AA_GC'] > 1.0 # (1+EPSILON)/EPSILON
    assert_allclose(features['quadrant_ratio_AT_CG'], 1.0) # EPSILON/EPSILON
