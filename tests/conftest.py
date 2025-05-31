# -*- coding: utf-8 -*-
"""
Pytest configuration file (fixtures).
"""
import pytest
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# Define constants used across tests
TEST_SEQ_VALID_SHORT = "ATGC" * 10  # Length 40
TEST_SEQ_VALID_MIN_LEN = "ATGC" * 50  # Length 200 (assuming MIN_SEQ_LEN is 200 for tests)
TEST_SEQ_VALID_LONG = "ATGC" * 1500  # Length 6000 (assuming MAX_SEQ_LEN is 5000 for tests)
TEST_SEQ_INVALID_CHARS = "ATGC" * 50 + "XYZ-" + "CGTA" * 50  # Length 204 + 4 non-ATGC
TEST_SEQ_EMPTY = ""
TEST_SEQ_NON_ASCII = "ATGC你好"

# Minimal config for tests that don't need full pipeline setup
MINIMAL_TEST_CONFIG = {
    'MIN_SEQ_LEN': 200,
    'MAX_SEQ_LEN': 5000,
    'FCGR_K': 6, # Default k from config.py
    'FCGR_DIM': 1 << 6,
    'ENTREZ_EMAIL': 'test@example.com', # Critical for data_acquisition
    'ENTREZ_API_KEY': None,
    'NCBI_FETCH_DELAY': 0.01, # Faster for tests
    'REQUEST_RETRIES': 1,
    'REQUEST_BACKOFF_FACTOR': 0.1,
    'REQUEST_STATUS_FORCELIST': (500, 503),
    'PLOTTING_ENABLED': True, # Assume plotting is tested unless skipped by specific test
    'PANDOC_CHECK_ENABLED': True,
    'N_JOBS': 1, # Default to 1 for tests to avoid complexity unless testing parallelism
    'DEFAULT_CACHE_DIR': None, # Disable caching by default for unit tests
    'FIGURES_SUBDIR': 'figures',
    'DATA_SUBDIR': 'data',
}

@pytest.fixture(scope="session")
def test_output_dir_session():
    """Creates a temporary directory for test outputs (session-scoped)."""
    temp_dir = tempfile.mkdtemp(prefix="fcgr_test_output_session_")
    print(f"Created temporary session test output directory: {temp_dir}")
    yield temp_dir
    print(f"Removing temporary session test output directory: {temp_dir}")
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(scope="function")
def test_output_dir(test_output_dir_session):
    """Creates a temporary directory for test outputs (function-scoped for isolation)."""
    # Use a subdirectory within the session dir for easier cleanup if a test fails mid-creation
    func_temp_dir = tempfile.mkdtemp(prefix="fcgr_test_func_", dir=test_output_dir_session)
    yield func_temp_dir
    shutil.rmtree(func_temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def figures_dir(test_output_dir):
    """Creates a figures subdirectory within the test output directory."""
    figs_dir = os.path.join(test_output_dir, MINIMAL_TEST_CONFIG['FIGURES_SUBDIR'])
    os.makedirs(figs_dir, exist_ok=True)
    return figs_dir

@pytest.fixture(scope="function")
def data_dir(test_output_dir):
    """Creates a data subdirectory within the test output directory."""
    dat_dir = os.path.join(test_output_dir, MINIMAL_TEST_CONFIG['DATA_SUBDIR'])
    os.makedirs(dat_dir, exist_ok=True)
    return dat_dir

@pytest.fixture
def mock_entrez_efetch(mocker):
    """Mocks Biopython Entrez.efetch specifically."""
    def mock_efetch_func(*args, **kwargs):
        identifier = kwargs.get('id', '')
        rettype = kwargs.get('rettype', 'fasta')
        # print(f"Mock Entrez.efetch called with: id={identifier}, rettype={rettype}") # For debugging

        mock_fasta_record = f">NM_12345_mock Mock sequence BRCA1\n{TEST_SEQ_VALID_MIN_LEN}\n"
        mock_gb_record_simple = f"""LOCUS       NM_12345_mock   {len(TEST_SEQ_VALID_MIN_LEN)} bp DNA linear PRI 01-JAN-2024
DEFINITION  Mock sequence BRCA1.
ACCESSION   NM_12345_mock
VERSION     NM_12345_mock.1
SOURCE      Homo sapiens
FEATURES             Location/Qualifiers
     source          1..{len(TEST_SEQ_VALID_MIN_LEN)}
     CDS             1..{len(TEST_SEQ_VALID_MIN_LEN)}
                     /gene="BRCA1_mock"
ORIGIN
        1 {TEST_SEQ_VALID_MIN_LEN.lower()}
//"""
        mock_gb_record_with_rrna = f"""LOCUS       NC_67890_mock   {len(TEST_SEQ_VALID_MIN_LEN)} bp DNA linear BAC 01-JAN-2024
DEFINITION  Mock E. coli genome section.
ACCESSION   NC_67890_mock
VERSION     NC_67890_mock.1
SOURCE      Escherichia coli
FEATURES             Location/Qualifiers
     source          1..{len(TEST_SEQ_VALID_MIN_LEN)}
     rRNA            complement(100..150)
                     /product="16S ribosomal RNA"
                     /gene="rrsA_mock"
ORIGIN
        1 {TEST_SEQ_VALID_MIN_LEN.lower()}
//"""
        mock_error_response = "<error>Identifier not found</error>" # More realistic error
        
        handle = MagicMock()
        if identifier == "NM_12345" and rettype == 'fasta':
            handle.read.return_value = mock_fasta_record.encode('utf-8')
        elif identifier == "NM_12345" and rettype == 'gb':
            handle.read.return_value = mock_gb_record_simple.encode('utf-8')
        elif identifier == "NC_67890" and rettype == 'gb':
            handle.read.return_value = mock_gb_record_with_rrna.encode('utf-8')
        elif identifier == "ERROR_ID":
            handle.read.return_value = mock_error_response.encode('utf-8')
        elif identifier == "EMPTY_SEQ_ACC": # Fasta record with empty sequence
             handle.read.return_value = f">EMPTY_SEQ_ACC\n\n".encode('utf-8')
        else:
            handle.read.return_value = "".encode('utf-8') # Empty for unknown

        handle.close.return_value = None
        return handle

    mock = mocker.patch('Bio.Entrez.efetch', side_effect=mock_efetch_func)
    mocker.patch('Bio.Entrez.email', MINIMAL_TEST_CONFIG['ENTREZ_EMAIL'])
    mocker.patch('Bio.Entrez.tool', "pytest_fcgr_analyzer")
    mocker.patch('Bio.Entrez.api_key', None)
    mocker.patch('time.sleep', return_value=None) # Prevent actual sleeping
    return mock


@pytest.fixture
def sample_fcgr_matrix_k6():
    """Provides a sample, non-trivial FCGR matrix for k=6 (64x64)."""
    k = 6 # Matches default config
    dim = 1 << k
    matrix = np.zeros((dim, dim), dtype=np.float32)
    matrix[0, 0] = 50
    matrix[dim-1, dim-1] = 30
    matrix[dim//2, dim//2] = 20
    matrix[1:5, 1:5] = np.random.rand(4, 4) * 10 # Small block
    matrix = matrix + np.random.rand(dim, dim) * 0.5 # Low-level noise
    if np.sum(matrix) > 0:
        matrix /= np.sum(matrix)
    return matrix.astype(np.float32)

@pytest.fixture
def sample_fcgr_matrix_k4():
    """Provides a sample, non-trivial FCGR matrix for k=4 (16x16)."""
    k = 4
    dim = 1 << k
    matrix = np.zeros((dim, dim), dtype=np.float32)
    matrix[0, 0] = 50
    matrix[dim-1, dim-1] = 30
    matrix[dim//2, dim//2] = 20
    matrix[1, 1] = 10
    matrix[2, 3] = 5
    matrix = matrix + np.random.rand(dim, dim) * 2
    if np.sum(matrix) > 0:
        matrix /= np.sum(matrix)
    return matrix.astype(np.float32)


@pytest.fixture
def sample_dataframe_for_analysis(sample_fcgr_matrix_k6):
    """Provides a DataFrame suitable for analysis tests."""
    n_samples = 30  # Increased for better stat testing
    # Make FCGR_K from MINIMAL_TEST_CONFIG effective here
    k_test = MINIMAL_TEST_CONFIG['FCGR_K']
    dim_test = 1 << k_test
    
    data = {
        'id': [f'hash_{i}' for i in range(n_samples)],
        'original_id': [f'ID_{chr(65+(i%26))}{(i//26)}' for i in range(n_samples)],
        'species': ['Human'] * (n_samples // 3) + ['Mouse'] * (n_samples // 3) + ['Bacteria'] * (n_samples - 2 * (n_samples // 3)),
        'biotype': (['Coding'] * (n_samples // 2) + ['rRNA'] * (n_samples - n_samples // 2)),
        'sequence': [TEST_SEQ_VALID_MIN_LEN] * n_samples, # Dummy sequence
        'length': np.random.randint(MINIMAL_TEST_CONFIG['MIN_SEQ_LEN'], MINIMAL_TEST_CONFIG['MAX_SEQ_LEN'] + 1, n_samples),
        'fcgr': [np.random.rand(dim_test, dim_test).astype(np.float32) / (dim_test**2) for _ in range(n_samples)],
        'mean': np.random.rand(n_samples) * 0.01,
        'variance': np.random.rand(n_samples) * 0.001,
        'skewness': np.random.randn(n_samples) * 0.5,
        'kurtosis': np.random.randn(n_samples) * 1.0 - 1.0, # Center around typical values
        'shannon_entropy': np.random.rand(n_samples) * k_test * 2, # Max entropy for 2^k * 2^k is log2((2^k)^2) = 2k
        'fractal_dimension': np.random.rand(n_samples) * 0.8 + 1.1, # Range [1.1, 1.9]
        'contrast': np.random.rand(n_samples) * 10,
        'dissimilarity': np.random.rand(n_samples) * 2,
        'homogeneity': np.random.rand(n_samples) * 0.3 + 0.7, # Range [0.7, 1.0]
        'energy': np.random.rand(n_samples) * 0.3 + 0.2,   # Range [0.2, 0.5]
        'correlation': np.random.rand(n_samples) * 0.4 + 0.3, # Range [0.3, 0.7]
        'ASM': np.random.rand(n_samples) * 0.1 + 0.01,
        **{f'hu_moment_{i}': np.random.randn(n_samples) * 2.0 for i in range(7)}, # Hu moments vary widely
        'fcgr_entropy': np.random.rand(n_samples) * k_test * 1.5,
        'quadrant_ratio_AA_GC': np.random.rand(n_samples) * 2.0 + 0.1,
        'quadrant_ratio_AT_CG': np.random.rand(n_samples) * 2.0 + 0.1,
        'center_mass_x': np.random.rand(n_samples) * 0.6 + 0.2, # Range [0.2, 0.8]
        'center_mass_y': np.random.rand(n_samples) * 0.6 + 0.2,
    }
    df = pd.DataFrame(data)
    # Ensure fcgr column uses the specific sample_fcgr_matrix if only one is needed by a test
    # For general analysis tests, the random ones above are fine.
    return df

@pytest.fixture
def default_pipeline_config():
    """Returns a copy of the default configuration settings."""
    # Import dynamically to get a fresh copy each time and avoid module-level state issues in tests
    from fcgr_analyzer import config as app_config
    
    # Create a dictionary from the module's attributes
    cfg_dict = {k: getattr(app_config, k) for k in dir(app_config) if not k.startswith('__')}
    
    # Override critical/user-specific settings for test environment
    cfg_dict['ENTREZ_EMAIL'] = MINIMAL_TEST_CONFIG['ENTREZ_EMAIL']
    cfg_dict['ENTREZ_API_KEY'] = MINIMAL_TEST_CONFIG['ENTREZ_API_KEY']
    cfg_dict['DEFAULT_CACHE_DIR'] = None # Disable caching by default
    cfg_dict['N_JOBS'] = 1 # Force single job for test predictability
    cfg_dict['NCBI_FETCH_DELAY'] = 0.01
    return cfg_dict