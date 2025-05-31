# -*- coding: utf-8 -*-
"""
Unit tests for the analysis module.
"""
import pytest
import os
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, PropertyMock
from numpy.testing import assert_almost_equal, assert_allclose

# Module containing functions to test
from fcgr_analyzer import analysis as an
from fcgr_analyzer.config import PLOTTING_ENABLED, N_JOBS as CFG_N_JOBS, RANDOM_STATE as CFG_RANDOM_STATE
from fcgr_analyzer.utils import IS_PYODIDE

# Fixtures and constants from conftest
from .conftest import sample_dataframe_for_analysis, sample_fcgr_matrix_k6, \
                      figures_dir, test_output_dir, MINIMAL_TEST_CONFIG

# Skip all tests in this module if plotting is disabled globally or libs not available
pytestmark = [
    pytest.mark.skipif(not PLOTTING_ENABLED, reason="Plotting is disabled in config"),
    pytest.mark.skipif(not an.PLOTTING_LIBS_AVAILABLE, reason="Matplotlib/Seaborn not available for analysis tests")
]

@pytest.fixture
def mock_plt_save_and_close(mocker):
    """Mocks common matplotlib savefig and close calls."""
    mock_savefig = mocker.patch('matplotlib.figure.Figure.savefig')
    mock_close = mocker.patch('matplotlib.pyplot.close')
    return mock_savefig, mock_close

# --- Test _save_plot_to_file / _save_plot_to_base64 ---
def test_save_plot_to_file_native(figures_dir, mock_plt_save_and_close, mocker):
    if IS_PYODIDE: pytest.skip("Native file saving test")
    
    mock_fig = MagicMock()
    mock_savefig, mock_close = mock_plt_save_and_close
    
    save_path = os.path.join(figures_dir, "test_save.png")
    success = an._save_plot_to_file(mock_fig, save_path, dpi=150)
    
    assert success is True
    mock_savefig.assert_called_once()
    assert mock_savefig.call_args[0][0] == save_path
    assert mock_savefig.call_args[1]['dpi'] == 150
    mock_close.assert_called_once_with(mock_fig)

def test_save_plot_to_file_pyodide(mock_plt_save_and_close, mocker):
    if not IS_PYODIDE: pytest.skip("Pyodide specific test")
    
    mocker.patch('fcgr_analyzer.analysis.IS_PYODIDE', True) # Force Pyodide mode for this test
    mock_fig = MagicMock()
    mock_savefig, mock_close = mock_plt_save_and_close
    
    success = an._save_plot_to_file(mock_fig, "/dummy/path.png")
    assert success is False
    mock_savefig.assert_not_called()
    mock_close.assert_called_once_with(mock_fig) # Should still close
    mocker.patch('fcgr_analyzer.analysis.IS_PYODIDE', IS_PYODIDE) # Restore

def test_save_plot_to_base64(mock_plt_save_and_close, mocker):
    mock_fig = MagicMock()
    mock_savefig, mock_close = mock_plt_save_and_close
    
    # Mock BytesIO buffer behavior
    mock_buf_instance = MagicMock()
    mock_buf_instance.read.return_value = b"fakedata"
    mocker.patch('io.BytesIO', return_value=mock_buf_instance)
    
    b64_string = an._save_plot_to_base64(mock_fig, dpi=90)
    
    assert b64_string == "ZmFrZWRhdGE=" # base64 of "fakedata"
    mock_savefig.assert_called_once()
    assert mock_savefig.call_args[1]['format'] == 'png'
    assert mock_savefig.call_args[1]['dpi'] == 90
    mock_close.assert_called_once_with(mock_fig)


# --- Test plot_fcgr ---
def test_plot_fcgr_file_output(figures_dir, sample_fcgr_matrix_k6, mock_plt_save_and_close):
    if IS_PYODIDE: pytest.skip("File output test")
    mock_savefig, _ = mock_plt_save_and_close
    
    save_target = os.path.join(figures_dir, "fcgr_render.png")
    result_path = an.plot_fcgr(sample_fcgr_matrix_k6, "Test FCGR", save_target, output_format='file')
    
    assert result_path == save_target
    mock_savefig.assert_called_once()

def test_plot_fcgr_base64_output(sample_fcgr_matrix_k6, mock_plt_save_and_close, mocker):
    mock_savefig, _ = mock_plt_save_and_close
    mocker.patch('fcgr_analyzer.analysis._save_plot_to_base64', return_value="base64string")
    
    result_b64 = an.plot_fcgr(sample_fcgr_matrix_k6, "Test FCGR", "dummy", output_format='base64')
    
    assert result_b64 == "base64string"
    an._save_plot_to_base64.assert_called_once() # Check the helper was called

def test_plot_fcgr_empty_matrix_returns_none(mock_plt_save_and_close):
    assert an.plot_fcgr(np.array([]), "Empty", "dummy.png") is None
    mock_plt_save_and_close[0].assert_not_called() # savefig

# --- Test run_feature_analysis ---
@pytest.mark.parametrize("output_fmt", ['file', 'base64'])
def test_run_feature_analysis_logic_and_plot(sample_dataframe_for_analysis, figures_dir, output_fmt, mocker):
    if IS_PYODIDE and output_fmt == 'file': pytest.skip("File output test in Pyodide")
        
    mock_savefig, _ = mock_plt_save_and_close(mocker) # Get specific mocks
    
    # Mock statistical functions
    mocker.patch('scipy.stats.kruskal', return_value=(12.0, 0.001))
    mocker.patch('scipy.stats.normaltest', return_value=(1.0, 0.5)) # Assume normal for ANOVA path
    mocker.patch('scipy.stats.f_oneway', return_value=(10.0, 0.002))
    mocker.patch('scipy.stats.mannwhitneyu', return_value=(5.0, 0.03))
    mocker.patch('statsmodels.stats.multitest.multipletests', return_value=([True, False], [0.03, 0.1], 1,1,1)) # Corrected return for multipletests

    # Mock base64 helper if needed
    if output_fmt == 'base64':
        mocker.patch('fcgr_analyzer.analysis._save_plot_to_base64', return_value="feature_dist_b64")

    df = sample_dataframe_for_analysis
    feature = 'fractal_dimension'
    group_by = 'species' # 3 groups: Human, Mouse, Bacteria
    
    results, plot_output = an.run_feature_analysis(
        df, feature, group_by, figures_dir, 
        output_format=output_fmt,
        statistical_tests=['kruskal', 'anova', 'mannwhitney']
    )

    assert results['feature'] == feature
    assert results['group_by'] == group_by
    assert 'descriptive_stats' in results
    assert 'kruskal_wallis' in results['tests']
    assert results['tests']['kruskal_wallis']['p_value'] < 0.05
    assert 'anova' in results['tests'] # Should run as normaltest was mocked to pass
    assert results['tests']['anova']['p_value'] < 0.05
    assert 'pairwise_comparisons' in results['tests']
    assert len(results['tests']['pairwise_comparisons']) == 3 # 3 pairs from 3 groups
    assert results['tests']['pairwise_comparisons'][0]['significant_corrected'] is True
    assert 'effect_size' in results
    
    if output_fmt == 'file':
        assert isinstance(plot_output, str) and plot_output.startswith(os.path.basename(figures_dir))
        assert results.get('plot_path') == plot_output
        mock_savefig.assert_called_once()
    else: # base64
        assert plot_output == "feature_dist_b64"
        assert 'plot_path' not in results # No file path for base64
        an._save_plot_to_base64.assert_called_once()


# --- Test plot_sequence_length_distribution ---
def test_plot_sequence_length_distribution(sample_dataframe_for_analysis, figures_dir, mock_plt_save_and_close):
    if IS_PYODIDE: pytest.skip("File output test")
    mock_savefig, _ = mock_plt_save_and_close
    
    plot_path = an.plot_sequence_length_distribution(sample_dataframe_for_analysis, figures_dir, output_format='file')
    
    assert plot_path is not None
    assert "sequence_length_analysis.png" in plot_path
    mock_savefig.assert_called_once()


# --- Test run_length_adjusted_analysis ---
@patch('sklearn.linear_model.LinearRegression')
def test_run_length_adjusted_analysis(MockLinearRegression, sample_dataframe_for_analysis, figures_dir, mock_plt_save_and_close, mocker):
    if IS_PYODIDE: pytest.skip("File output test")
    if not an.SKLEARN_AVAILABLE: pytest.skip("Sklearn needed for LinearRegression")

    mock_savefig, _ = mock_plt_save_and_close
    mock_lr_instance = MagicMock()
    mock_lr_instance.fit.return_value = None
    mock_lr_instance.predict.return_value = np.random.rand(len(sample_dataframe_for_analysis))
    MockLinearRegression.return_value = mock_lr_instance
    
    mocker.patch('scipy.stats.pearsonr', return_value=(0.6, 0.001)) # r, p
    mocker.patch('scipy.stats.kruskal', return_value=(8.0, 0.005))
    
    feature = 'fractal_dimension'
    results, plot_output = an.run_length_adjusted_analysis(sample_dataframe_for_analysis, feature, figures_dir, output_format='file')
    
    assert results['feature'] == feature
    assert 'length_correlation' in results
    assert results['length_correlation']['pearson_r'] == 0.6
    assert 'length_adjusted_comparisons' in results
    assert 'species' in results['length_adjusted_comparisons']
    
    assert plot_output is not None
    assert f"length_adjusted_{feature}.png" in plot_output
    mock_savefig.assert_called_once()


# --- Test plot_pairwise_comparisons_heatmap ---
def test_plot_pairwise_comparisons_heatmap(figures_dir, mock_plt_save_and_close):
    if IS_PYODIDE: pytest.skip("File output test")
    mock_savefig, _ = mock_plt_save_and_close
    
    sample_stats_results = [
        {'feature': 'feat1', 'group_by': 'species', 'tests': {'pairwise_comparisons': [
            {'group1': 'A', 'group2': 'B', 'p_value_corrected': 0.01, 'effect_size': 0.5},
            {'group1': 'A', 'group2': 'C', 'p_value_corrected': 0.1, 'effect_size': 0.2},
            {'group1': 'B', 'group2': 'C', 'p_value_corrected': 0.04, 'effect_size': -0.3}
        ]}}
    ]
    plot_path = an.plot_pairwise_comparisons_heatmap(sample_stats_results, figures_dir, output_format='file')
    
    assert plot_path is not None
    assert "pairwise_comparisons_heatmap.png" in plot_path
    mock_savefig.assert_called_once()

# --- Test calculate_feature_entropy ---
@pytest.mark.parametrize("matrix_data, expected_entropy", [
    (np.array([[0.25, 0.25], [0.25, 0.25]]), 2.0), # Max entropy for 2x2
    (np.array([[1, 0], [0, 0]]), 0.0),           # Min entropy
    (np.array([[0.5, 0], [0.5, 0]]), 1.0),
    (np.array([[0.0, 0.0], [0.0, 0.0]]), 0.0),   # All zeros
    (np.array([[0.5, 0.25], [0.125, 0.125]]), -(0.5*np.log2(0.5) + 0.25*np.log2(0.25) + 2*0.125*np.log2(0.125)))
])
def test_calculate_feature_entropy(matrix_data, expected_entropy):
    matrix = matrix_data.astype(np.float32)
    # Ensure matrix is normalized if it represents probabilities already for this test
    if np.sum(matrix) > 0 and not np.isclose(np.sum(matrix), 1.0):
        matrix_for_calc = matrix / np.sum(matrix)
    else:
        matrix_for_calc = matrix
        
    entropy = an.calculate_feature_entropy(matrix_for_calc)
    assert_almost_equal(entropy, expected_entropy, decimal=6)

# --- Test plot_feature_correlations_network ---
@pytest.mark.skipif(not getattr(an, 'NETWORKX_AVAILABLE', False), reason="NetworkX not available")
def test_plot_feature_correlations_network(sample_dataframe_for_analysis, figures_dir, mock_plt_save_and_close):
    if IS_PYODIDE: pytest.skip("File output test")
    mock_savefig, _ = mock_plt_save_and_close
    
    df = sample_dataframe_for_analysis
    # Select a few features that are likely to have some correlation
    features = ['mean', 'variance', 'fractal_dimension', 'contrast', 'shannon_entropy']
    # Ensure these features actually exist
    features = [f for f in features if f in df.columns]
    if len(features) < 2 : pytest.skip("Not enough common features for correlation network")
    
    # Ensure some correlation above threshold to create edges
    df_copy = df.copy()
    if 'mean' in features and 'variance' in features:
      df_copy['variance'] = df_copy['mean'] * 2 + np.random.rand(len(df_copy)) * 0.0001 # Create correlation
    
    plot_path = an.plot_feature_correlations_network(df_copy, features, figures_dir, output_format='file', correlation_threshold=0.1)
    
    if plot_path: # Plot is only generated if edges exist
        assert "feature_correlation_network.png" in plot_path
        mock_savefig.assert_called_once()
    else: # No edges, no plot
        mock_savefig.assert_not_called()

# --- Test run_dimensionality_reduction ---
@pytest.mark.skipif(not an.SKLEARN_AVAILABLE, reason="scikit-learn not available")
@pytest.mark.parametrize("method_list", [['PCA'], ['t-SNE'], ['PCA', 't-SNE', 'UMAP']]) # UMAP tested if available
def test_run_dimensionality_reduction_methods(method_list, sample_dataframe_for_analysis, figures_dir, mock_plt_save_and_close, mocker):
    if IS_PYODIDE: pytest.skip("File output test")
    mock_savefig, mock_close = mock_plt_save_and_close

    df = sample_dataframe_for_analysis
    features = [f for f in df.columns if df[f].dtype in [np.float64, np.int64] and f not in ['length']]
    if len(features) < 2: pytest.skip("Not enough numeric features for DR")

    X = df[features].values
    y = df['species'].values
    
    # Mock DR methods
    mock_scaler_inst = MagicMock()
    mock_scaler_inst.fit_transform.return_value = X
    mocker.patch('fcgr_analyzer.analysis.StandardScaler', return_value=mock_scaler_inst)
    
    n_samples = len(X)
    
    mock_pca_inst = MagicMock()
    mock_pca_inst.fit_transform.return_value = np.random.rand(n_samples, 2)
    type(mock_pca_inst).explained_variance_ratio_ = PropertyMock(return_value=np.array([0.6, 0.2]))
    mocker.patch('fcgr_analyzer.analysis.PCA', return_value=mock_pca_inst)
    
    mock_tsne_inst = MagicMock()
    mock_tsne_inst.fit_transform.return_value = np.random.rand(n_samples, 2)
    mocker.patch('fcgr_analyzer.analysis.TSNE', return_value=mock_tsne_inst)

    if 'UMAP' in method_list:
        if an.UMAP is None: # If UMAP is not actually available, skip this param set
            pytest.skip("UMAP not available for this parameterization")
        mock_umap_inst = MagicMock()
        mock_umap_inst.fit_transform.return_value = np.random.rand(n_samples, 2)
        mocker.patch('fcgr_analyzer.analysis.UMAP', return_value=mock_umap_inst)
    
    plot_output_paths = an.run_dimensionality_reduction(
        X, y, "Species", figures_dir, 
        methods=method_list, output_format='file'
    )
    
    num_expected_plots = 0
    if 'PCA' in method_list: num_expected_plots +=1
    if 't-SNE' in method_list: num_expected_plots +=1
    if 'UMAP' in method_list and an.UMAP is not None: num_expected_plots +=1
    
    assert mock_savefig.call_count == num_expected_plots
    assert mock_close.call_count == num_expected_plots
    
    if num_expected_plots > 0:
        # The function was changed to return a list of plot paths, one per method.
        # For simplicity, the original test structure checked a single path.
        # Now, we'd check if plot_output_paths (which is a dict) has the expected keys.
        # The test in the provided code checks a single string - this behavior has changed.
        # The current code returns the path of the *first successful* plot.
        assert isinstance(plot_output_paths, str) and "dim_reduction_Species" in plot_output_paths
    else: # Should not happen with current param if methods are valid
        assert plot_output_paths is None


# --- Test plot_feature_heatmap_normalized ---
def test_plot_feature_heatmap_normalized(sample_dataframe_for_analysis, figures_dir, mock_plt_save_and_close):
    if IS_PYODIDE: pytest.skip("File output test")
    mock_savefig, _ = mock_plt_save_and_close
    
    df = sample_dataframe_for_analysis
    features = ['mean', 'variance', 'fractal_dimension']
    
    plot_path = an.plot_feature_heatmap_normalized(df, features, 'species', figures_dir, output_format='file')
    
    assert plot_path is not None
    assert "feature_heatmap_normalized_species.png" in plot_path
    mock_savefig.assert_called_once()

# --- Test plot_correlation_heatmap (already tested, but can add more specific variant if needed) ---

# --- Test plot_feature_importance ---
def test_plot_feature_importance(figures_dir, mock_plt_save_and_close):
    if IS_PYODIDE: pytest.skip("File output test")
    mock_savefig, _ = mock_plt_save_and_close
        
    feature_names = [f'feat_{i}' for i in range(15)]
    importances = np.random.rand(15)
    std_devs = np.random.rand(15) * 0.1
    
    plot_path = an.plot_feature_importance(
        feature_names, importances, std_devs, 
        "Test Importances", figures_dir, output_format='file', top_n=10
    )
    assert plot_path is not None
    assert "test_importances.png" in plot_path
    mock_savefig.assert_called_once()


# --- Test plot_learning_curves ---
def test_plot_learning_curves(figures_dir, mock_plt_save_and_close):
    if IS_PYODIDE: pytest.skip("File output test")
    mock_savefig, _ = mock_plt_save_and_close
    
    train_sizes = np.array([10, 20, 30, 40])
    train_scores = np.random.rand(4, 5) * 0.2 + 0.7 # Scores between 0.7 and 0.9
    val_scores = np.random.rand(4, 5) * 0.2 + 0.6   # Scores between 0.6 and 0.8
    
    plot_path = an.plot_learning_curves(
        train_sizes, train_scores, val_scores, 
        "Test Learning Curves", figures_dir, output_format='file'
    )
    assert plot_path is not None
    assert "learning_curves.png" in plot_path
    mock_savefig.assert_called_once()


# --- Test run_comprehensive_statistics ---
def test_run_comprehensive_statistics(sample_dataframe_for_analysis, mocker):
    df = sample_dataframe_for_analysis
    features = ['mean', 'variance', 'fractal_dimension']
    grouping_vars = ['species', 'biotype']
    
    # Mock stat functions to ensure they are called and control their output slightly
    mocker.patch('scipy.stats.normaltest', return_value=(MagicMock(statistic=1.0), MagicMock(pvalue=0.5)))
    mocker.patch('scipy.stats.pearsonr', return_value=(MagicMock(statistic=0.5), MagicMock(pvalue=0.01)))
    mocker.patch('scipy.stats.spearmanr', return_value=(MagicMock(correlation=0.4), MagicMock(pvalue=0.02)))
    mocker.patch('scipy.stats.chi2_contingency', return_value=(5.0, 0.02, 2, np.array([[1,1],[1,1]]))) # chi2, p, dof, expected
    
    results = an.run_comprehensive_statistics(df, features, grouping_vars)
    
    assert 'normality_tests' in results
    assert len(results['normality_tests']) == len(features)
    assert 'mean' in results['normality_tests']
    
    assert 'correlation_tests' in results
    num_pairs = len(features) * (len(features) - 1) // 2
    assert len(results['correlation_tests']) == num_pairs
    assert 'mean_vs_variance' in results['correlation_tests']
    
    assert 'chi_square_tests' in results
    assert len(results['chi_square_tests']) == 1 # species_vs_biotype
    assert 'species_vs_biotype' in results['chi_square_tests']
    
    assert 'summary_stats' in results
    assert len(results['summary_stats']) == len(features)
    assert 'mean' in results['summary_stats']
    assert 'cv' in results['summary_stats']['mean'] # Check for coefficient of variation
