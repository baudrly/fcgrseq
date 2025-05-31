# -*- coding: utf-8 -*-
"""
Unit tests for the reporting module.
"""
import pytest
import os
import json
import numpy as np
import subprocess # For CalledProcessError
from unittest.mock import patch, mock_open, MagicMock

# Module containing functions to test
from fcgr_analyzer import reporting as rp
from fcgr_analyzer.utils import convert_numpy_for_json, IS_PYODIDE
from fcgr_analyzer.config import PANDOC_CHECK_ENABLED

# Fixtures and constants from conftest
from .conftest import test_output_dir, MINIMAL_TEST_CONFIG

@pytest.fixture
def sample_report_summary_data():
    """Creates a more comprehensive sample results summary for reporting tests."""
    k_val = MINIMAL_TEST_CONFIG['FCGR_K']
    dim_val = 1 << k_val
    summary = {
        'config': {
            **MINIMAL_TEST_CONFIG, # Base minimal config
            'FCGR_K': k_val, 'FCGR_DIM': dim_val,
            'output_dir': '/test/output', 'figures_dir': '/test/output/figures', 
            'data_dir': '/test/output/data', 'N_JOBS': 2, 'gpu_available': True,
            'mixed_precision': True, 'PLOTTING_ENABLED': True, 'targets_source': 'test_list.csv',
        },
        'data_summary': {
            'targets_requested': 25, 'sequences_processed': 20, 'fcgrs_generated': 18,
            'features_extracted_count': 15,
            'species_counts': {'Human': 10, 'Mouse': 5, 'Rat': 3},
            'biotype_counts': {'Coding': 12, 'rRNA': 4, 'tRNA': 2},
            'input_source': 'test_targets.csv',
            'sequence_summary_path': 'data/seq_summary.csv', 'features_path': 'data/features.csv',
        },
        'timings': {
            'data_acquisition': 10.1, 'fcgr_generation': 5.2, 'feature_extraction': 8.3,
            'analysis_plotting': 12.4, 'machine_learning': 30.5, 'reporting': 1.6, 'total_pipeline': 68.1
        },
        'fcgr_examples_b64': ["fcgr_b64_1", "fcgr_b64_2"] if IS_PYODIDE else None,
        'fcgr_examples': ["figures/fcgr1.png", "figures/fcgr2.png"] if not IS_PYODIDE else None,
        'feature_analysis': {
            'stats_tests': [
                {'feature': 'mean', 'group_by': 'species', 'tests':{'kruskal_wallis':{'statistic': 5.6, 'p_value': 0.06, 'significant': False}}, 'plot_path': 'figures/dist_mean_sp.png'},
                {'feature': 'fd', 'group_by': 'biotype', 'tests':{'kruskal_wallis':{'statistic': 12.1, 'p_value': 0.001, 'significant': True}}, 'plot_path': 'figures/dist_fd_bt.png'}
            ],
            'correlation_plot_b64': "corr_b64" if IS_PYODIDE else None,
            'correlation_plot': "figures/corr.png" if not IS_PYODIDE else None,
            'correlation_network_b64': "corr_net_b64" if IS_PYODIDE else None,
            'correlation_network': "figures/corr_net.png" if not IS_PYODIDE else None,
            'length_distribution_plot_b64': "len_dist_b64" if IS_PYODIDE else None,
            'length_distribution_plot': "figures/len_dist.png" if not IS_PYODIDE else None,
            'pairwise_comparisons_heatmap_b64': "pair_hm_b64" if IS_PYODIDE else None,
            'pairwise_comparisons_heatmap': "figures/pair_hm.png" if not IS_PYODIDE else None,
            'feature_heatmap_species_b64': "fhs_b64" if IS_PYODIDE else None,
            'feature_heatmap_species': "figures/fhs.png" if not IS_PYODIDE else None,
            'feature_heatmap_biotype_b64': "fhb_b64" if IS_PYODIDE else None,
            'feature_heatmap_biotype': "figures/fhb.png" if not IS_PYODIDE else None,
            'length_adjusted_analyses': [{'feature':'fd', 'length_correlation':{'pearson_r':0.5, 'p_value':0.01, 'significant':True}, 'plot_path':'figures/len_adj_fd.png'}],
            'comprehensive_stats': {'normality_tests': {'fd': {'statistic':1.0, 'p_value':0.5, 'is_normal':True}}},
            'example_dist_plots_b64': {'fd_species': 'b64_fd_sp_dist'} if IS_PYODIDE else None,
            'dim_reduction_plots': {
                'species_PCA_b64': "sp_pca_b64" if IS_PYODIDE else None, 'species_PCA': "figures/sp_pca.png" if not IS_PYODIDE else None,
                'biotype_tSNE_b64': "bt_tsne_b64" if IS_PYODIDE else None, 'biotype_tSNE': "figures/bt_tsne.png" if not IS_PYODIDE else None,
            },
            'rf_importance_species_b64': "rf_imp_sp_b64" if IS_PYODIDE else None,
            'rf_importance_species': "figures/rf_imp_sp.png" if not IS_PYODIDE else None,
            'learning_curves_species_b64': "lc_sp_b64" if IS_PYODIDE else None,
            'learning_curves_species': "figures/lc_sp.png" if not IS_PYODIDE else None,
        },
        'ml_results': {
            'Species': {
                'target': 'Species', 'accuracy': 0.95, 'loss': 0.15,
                'report': {'Human': {'precision': 1.0, 'recall': 0.9, 'f1-score': 0.95, 'support': 2}, 'accuracy': 0.95, 'macro avg': {}, 'weighted avg': {}},
                'history_plot_b64': "hist_sp_b64" if IS_PYODIDE else None, 'history_plot': "figures/hist_sp.png" if not IS_PYODIDE else None,
                'cm_plot_b64': "cm_sp_b64" if IS_PYODIDE else None, 'cm_plot': "figures/cm_sp.png" if not IS_PYODIDE else None,
            }
        },
        'performance_metrics': {
            'stage_times': {'data_acquisition': 10.1, 'fcgr_generation':5.2},
            'processing_rates': {'data_acquisition': 2.47, 'fcgr_generation':3.46},
            'memory_usage': {'data_acquisition_start': 50.0, 'data_acquisition_end': 60.0}
        },
        'report_paths': { # Native mode paths
            'json_summary': 'data/summary.json', 'markdown_report': 'report.md', 
            'pdf_report': 'report.pdf', 'log_file': 'analysis.log'
        } if not IS_PYODIDE else {},
        'report_md_content': "Markdown content here" if IS_PYODIDE else None,
        'logs': "Log messages here" if IS_PYODIDE else None,
    }
    return json.loads(json.dumps(summary, default=convert_numpy_for_json)) # Ensure clean types

# --- Test format_clf_report_markdown ---
def test_format_clf_report_markdown_detailed(sample_report_summary_data):
    report_dict = sample_report_summary_data['ml_results']['Species']['report']
    md = rp.format_clf_report_markdown(report_dict, "Species")
    assert "Overall Accuracy: 0.9500" in md
    assert "| Human | 1.000 | 0.900 | 0.950 | 2 |" in md
    assert "| **Accuracy** |   |   | **0.950** | 2 |" in md # Support should be total of class supports

# --- Test format_statistical_tests_markdown ---
def test_format_statistical_tests_markdown_content(sample_report_summary_data):
    stats_data = sample_report_summary_data['feature_analysis']['comprehensive_stats']
    # Add more data for testing
    stats_data['correlation_tests'] = {
        'fd_vs_mean': {'pearson_r': 0.75, 'pearson_p': 0.0001, 'spearman_r': 0.7, 'spearman_p': 0.0002, 'n_samples': 18}
    }
    stats_data['chi_square_tests'] = {
        'species_vs_biotype': {'chi2': 10.2, 'p_value': 0.005, 'cramers_v': 0.4, 'dof': 2, 'n_samples': 18}
    }
    md = rp.format_statistical_tests_markdown(stats_data)
    assert "Normality Tests" in md
    assert "| fd | 1.000 | 0.500 | Yes |" in md
    assert "Feature Correlations" in md
    assert "| fd vs mean | 0.750 | 1.000e-04 | 0.700 | 2.000e-04 | 18 |" in md
    assert "Chi-Square Tests" in md
    assert "| species vs biotype | 10.20 | 5.000e-03 | 0.400 | 2 | 18 |" in md

# --- Test format_length_adjusted_markdown ---
def test_format_length_adjusted_markdown_content(sample_report_summary_data):
    length_adj_data = sample_report_summary_data['feature_analysis']['length_adjusted_analyses']
    md = rp.format_length_adjusted_markdown(length_adj_data)
    assert "Length-Adjusted Feature Analysis" in md
    assert "| fd | 0.500 | 1.000e-02 | Yes |" in md
    assert "**Features significantly correlated with length:** fd" in md

# --- Test format_performance_metrics_markdown ---
def test_format_performance_metrics_markdown_content(sample_report_summary_data):
    perf_data = sample_report_summary_data['performance_metrics']
    md = rp.format_performance_metrics_markdown(perf_data)
    assert "Stage Execution Times" in md
    assert "| Data Acquisition | 10.10 (66.0%) | 2.5 |" in md # (10.1 / (10.1+5.2)) * 100
    assert "Memory Usage" in md
    assert "| Data Acquisition Start | 50.0 | N/A |" in md
    assert "| Data Acquisition End | 60.0 | +10.0 |" in md


# --- Test generate_markdown_report ---
@patch('fcgr_analyzer.reporting.IS_PYODIDE', False) # Test native mode
def test_generate_markdown_report_native_mode(mock_is_pyodide_false, sample_report_summary_data, test_output_dir):
    report_filepath = os.path.join(test_output_dir, "native_report.md")
    sample_report_summary_data['config']['output_dir'] = test_output_dir # Ensure correct path for relative links

    md_content = rp.generate_markdown_report(sample_report_summary_data, report_filepath)
    
    assert os.path.exists(report_filepath)
    with open(report_filepath, 'r') as f: content = f.read()
    
    assert "# FCGR Analysis Report (Native Mode)" in content
    assert f"*Output Directory: `{os.path.abspath(test_output_dir)}`" in content
    assert "![FCGR Example 1](figures/fcgr1.png)" in content
    assert "![Random Forest Feature Importances - Species](figures/rf_imp_sp.png)" in content
    assert "![SVM Learning Curves](figures/lc_sp.png)" in content
    assert "![Species Training History](figures/hist_sp.png)" in content
    assert "Generated Files" in content # Native specific section
    assert "- `fcgr_analysis_report.pdf`: PDF version" in content # Assumes PDF path is in summary

@patch('fcgr_analyzer.reporting.IS_PYODIDE', True) # Test web/Pyodide mode
def test_generate_markdown_report_web_mode(mock_is_pyodide_true, sample_report_summary_data):
    # In web mode, report_filepath is None
    md_content = rp.generate_markdown_report(sample_report_summary_data, None)
    
    assert "# FCGR Analysis Report (Web Mode)" in content
    assert "Output Directory:" not in content # Native specific
    assert "*[Plot generated: FCGR Example 1 (See web interface)]*" in content
    assert "*[Plot generated: Random Forest Feature Importances - Species (See web interface)]*" in content
    assert "Generated Files" not in content # Native specific section

def test_generate_markdown_report_handles_error(sample_report_summary_data, test_output_dir, caplog):
    # Introduce an error by making some data unformattable for the report
    faulty_summary = sample_report_summary_data.copy()
    faulty_summary['data_summary'] = "this will break formatting" # Invalid type
    
    report_filepath = os.path.join(test_output_dir, "error_report.md")
    md_content = rp.generate_markdown_report(faulty_summary, report_filepath)
    
    assert "Report Generation Failed" in md_content
    assert "Error: TypeError" in md_content # Or whatever error it causes
    assert "Failed to generate Markdown report content" in caplog.text
    if not IS_PYODIDE: # Error file only saved in native mode
        assert os.path.exists(os.path.join(test_output_dir, "error_report_error.md"))


# --- Test generate_pdf_report ---
@patch('fcgr_analyzer.reporting.IS_PYODIDE', False)
@patch('fcgr_analyzer.reporting.SUBPROCESS_AVAILABLE', True)
@patch('fcgr_analyzer.reporting.subprocess') # Mock subprocess module
@patch('fcgr_analyzer.reporting.check_pandoc_exists') # Mock pandoc check
def test_generate_pdf_report_success(mock_check_pandoc, mock_subprocess, mock_is_pyodide_false, test_output_dir):
    mock_check_pandoc.return_value = True # Assume pandoc exists
    
    # Mock subprocess.run behavior
    mock_proc_result = MagicMock()
    mock_proc_result.returncode = 0
    mock_proc_result.stdout = ""
    mock_proc_result.stderr = "" # No errors from pandoc
    mock_subprocess.run.return_value = mock_proc_result
    
    md_filepath = os.path.join(test_output_dir, "input.md")
    pdf_filepath = os.path.join(test_output_dir, "output.pdf")
    with open(md_filepath, "w") as f: f.write("# Test MD") # Create dummy MD

    # Mock os.path.exists and os.path.getsize for the PDF file after generation
    original_exists = os.path.exists
    original_getsize = os.path.getsize
    def mock_os_ops(path):
        if path == pdf_filepath: return True # Simulate PDF created
        return original_exists(path)
    
    with patch('os.path.exists', side_effect=mock_os_ops):
        with patch('os.path.getsize', return_value=1024): # Simulate non-empty PDF
            success = rp.generate_pdf_report(md_filepath, pdf_filepath)

    assert success is True
    mock_subprocess.run.assert_called_once()
    cmd_args = mock_subprocess.run.call_args[0][0]
    assert "pandoc" in cmd_args[0]
    assert md_filepath in cmd_args
    assert pdf_filepath in cmd_args
    assert f"--resource-path={os.path.dirname(md_filepath)}:{os.path.join(os.path.dirname(md_filepath), 'figures')}" in cmd_args


@patch('fcgr_analyzer.reporting.IS_PYODIDE', False)
@patch('fcgr_analyzer.reporting.SUBPROCESS_AVAILABLE', True)
@patch('fcgr_analyzer.reporting.subprocess')
@patch('fcgr_analyzer.reporting.check_pandoc_exists')
def test_generate_pdf_report_pandoc_command_fails(mock_check_pandoc, mock_subprocess, mock_is_pyodide_false, test_output_dir):
    mock_check_pandoc.return_value = True
    mock_subprocess.run.side_effect = subprocess.CalledProcessError(1, "pandoc", stderr="Pandoc Error!")
    
    md_filepath = os.path.join(test_output_dir, "input_fail.md")
    pdf_filepath = os.path.join(test_output_dir, "output_fail.pdf")
    with open(md_filepath, "w") as f: f.write("# Test MD Fail")

    success = rp.generate_pdf_report(md_filepath, pdf_filepath)
    assert success is False

@patch('fcgr_analyzer.reporting.IS_PYODIDE', False)
def test_generate_pdf_report_no_pandoc(mock_is_pyodide_false, test_output_dir, caplog):
    with patch('fcgr_analyzer.reporting.check_pandoc_exists', return_value=False):
        # Ensure PANDOC_CHECK_ENABLED is True for this test path
        with patch('fcgr_analyzer.reporting.PANDOC_CHECK_ENABLED', True):
            success = rp.generate_pdf_report("input.md", "output.pdf")
            assert success is True # Returns True because it's skipped gracefully
            assert "Pandoc executable not found" in caplog.text # Warning logged by check_pandoc_exists

# --- Test save_results_summary ---
@patch('fcgr_analyzer.reporting.IS_PYODIDE', False)
def test_save_results_summary_native(mock_is_pyodide_false, sample_report_summary_data, test_output_dir):
    json_filepath = os.path.join(test_output_dir, "summary_out.json")
    
    # Use mock_open to avoid actual file writing but check calls
    m = mock_open()
    with patch('builtins.open', m):
        with patch('json.dump') as mock_json_dump:
            rp.save_results_summary(sample_report_summary_data, json_filepath)
    
    m.assert_called_once_with(json_filepath, 'w', encoding='utf-8')
    mock_json_dump.assert_called_once()
    # Check some args of json.dump
    assert mock_json_dump.call_args[1]['indent'] == 4
    assert mock_json_dump.call_args[1]['default'] == convert_numpy_for_json

@patch('fcgr_analyzer.reporting.IS_PYODIDE', True)
def test_save_results_summary_pyodide(mock_is_pyodide_true, sample_report_summary_data, caplog):
    with patch('builtins.open') as mock_open_pyodide: # Should not be called
        rp.save_results_summary(sample_report_summary_data, "/dummy.json")
    
    mock_open_pyodide.assert_not_called()
    assert "JSON file saving skipped in Pyodide" in caplog.text
