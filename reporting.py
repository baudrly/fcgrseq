# -*- coding: utf-8 -*-
"""
Enhanced Report Generation Module.
Creates comprehensive Markdown and PDF reports with improved statistics and visualizations.
"""
import os
import time
import logging
import json
import pandas as pd
import numpy as np
import traceback
from typing import Dict, List, Any, Optional, Union

# Import framework components and check flags
from .utils import IS_PYODIDE, check_pandoc_exists, \
                   safe_filename, convert_numpy_for_json
from .config import PANDOC_CHECK_ENABLED

# --- Conditional Imports ---
SUBPROCESS_AVAILABLE = False
subprocess = None
if not IS_PYODIDE:
    try:
        import subprocess as sp
        subprocess = sp
        SUBPROCESS_AVAILABLE = True
        logging.debug("Subprocess module loaded.")
    except ImportError:
        logging.debug("Subprocess module not available.")


# --- Enhanced Report Formatting Functions ---

def format_clf_report_markdown(report_dict: Optional[dict], target_name: str) -> str:
    """Enhanced classification report formatting with additional metrics."""
    if not isinstance(report_dict, dict):
        logging.warning(f"Invalid/missing report data for {target_name}.")
        return f"### Classification Report ({target_name})\n\n*Report data not available or invalid.*\n\n"

    md = f"### Classification Report ({target_name})\n\n"
    
    # Extract macro/weighted averages for summary
    macro_avg = report_dict.get('macro avg', {})
    weighted_avg = report_dict.get('weighted avg', {})
    accuracy = report_dict.get('accuracy')
    
    # Summary box
    if accuracy is not None or macro_avg or weighted_avg:
        md += "**Summary Metrics:**\n"
        md += "```\n"
        if accuracy is not None:
            md += f"Overall Accuracy: {accuracy:.4f}\n"
        if macro_avg:
            md += f"Macro F1-Score:   {macro_avg.get('f1-score', 0.0):.4f}\n"
        if weighted_avg:
            md += f"Weighted F1-Score: {weighted_avg.get('f1-score', 0.0):.4f}\n"
        md += "```\n\n"
    
    # Detailed table
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    md += "| " + " | ".join(headers) + " |\n"
    md += "|" + "|".join(['-----'] * len(headers)) + "|\n"

    support_total_check = 0
    # Class rows
    for class_name, metrics in report_dict.items():
        if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
             support = metrics.get('support', 0)
             p = f"{metrics.get('precision', 0.0):.3f}"
             r = f"{metrics.get('recall', 0.0):.3f}"
             f1 = f"{metrics.get('f1-score', 0.0):.3f}"
             s = f"{int(support):d}"
             md += f"| {class_name} | {p} | {r} | {f1} | {s} |\n"
             support_total_check += int(support)

    # Separator
    md += "|" + "|".join(['-----'] * len(headers)) + "|\n"

    # Summary rows
    if accuracy is not None:
        acc = f"{accuracy:.3f}"
        total_support = support_total_check or \
                        get_nested(report_dict, ['macro avg', 'support']) or \
                        get_nested(report_dict, ['weighted avg', 'support']) or 'N/A'
        if isinstance(total_support, (int, float)): 
            total_support = f"{int(total_support)}"
        md += f"| **Accuracy** |   |   | **{acc}** | {total_support} |\n"

    # Average rows
    for avg_type in ['macro avg', 'weighted avg']:
        metrics = report_dict.get(avg_type)
        if isinstance(metrics, dict):
            p = f"{metrics.get('precision', 0.0):.3f}"
            r = f"{metrics.get('recall', 0.0):.3f}"
            f1 = f"{metrics.get('f1-score', 0.0):.3f}"
            s = f"{int(metrics.get('support', 0))}"
            md += f"| **{avg_type.title()}** | {p} | {r} | {f1} | {s} |\n"

    return md + "\n"


def format_statistical_tests_markdown(stats_dict: dict) -> str:
    """Format comprehensive statistical test results."""
    md = "### Statistical Analysis Results\n\n"
    
    # Normality tests
    if 'normality_tests' in stats_dict and stats_dict['normality_tests']:
        md += "#### Normality Tests (D'Agostino-Pearson)\n\n"
        md += "| Feature | Statistic | p-value | Normal? |\n"
        md += "|---------|-----------|---------|--------|\n"
        
        for feat, results in stats_dict['normality_tests'].items():
            stat = f"{results['statistic']:.3f}"
            p = f"{results['p_value']:.3e}" if results['p_value'] < 0.001 else f"{results['p_value']:.3f}"
            normal = "Yes" if results['is_normal'] else "No"
            md += f"| {feat} | {stat} | {p} | {normal} |\n"
        md += "\n"
    
    # Correlation tests
    if 'correlation_tests' in stats_dict and stats_dict['correlation_tests']:
        md += "#### Feature Correlations\n\n"
        md += "| Feature Pair | Pearson r | Pearson p | Spearman rho | Spearman p | n |\n"
        md += "|--------------|-----------|-----------|------------|------------|---|\n"
        
        # Limit to top 10 correlations by absolute r value
        sorted_corrs = sorted(stats_dict['correlation_tests'].items(), 
                            key=lambda x: abs(x[1]['pearson_r']), reverse=True)[:10]
        
        for pair, results in sorted_corrs:
            pr = f"{results['pearson_r']:.3f}"
            pp = f"{results['pearson_p']:.3e}" if results['pearson_p'] < 0.001 else f"{results['pearson_p']:.3f}"
            sr = f"{results['spearman_r']:.3f}"
            sp = f"{results['spearman_p']:.3e}" if results['spearman_p'] < 0.001 else f"{results['spearman_p']:.3f}"
            n = results['n_samples']
            md += f"| {pair.replace('_', ' ')} | {pr} | {pp} | {sr} | {sp} | {n} |\n"
        
        if len(stats_dict['correlation_tests']) > 10:
            md += f"\n*Showing top 10 correlations out of {len(stats_dict['correlation_tests'])} total pairs*\n"
        md += "\n"
    
    # Chi-square tests
    if 'chi_square_tests' in stats_dict and stats_dict['chi_square_tests']:
        md += "#### Chi-Square Tests of Independence\n\n"
        md += "| Variable Pair | chi² | p-value | Cramér's V | DoF | n |\n"
        md += "|---------------|-----|---------|------------|-----|---|\n"
        
        for pair, results in stats_dict['chi_square_tests'].items():
            chi2 = f"{results['chi2']:.2f}"
            p = f"{results['p_value']:.3e}" if results['p_value'] < 0.001 else f"{results['p_value']:.3f}"
            v = f"{results['cramers_v']:.3f}"
            dof = results['dof']
            n = results['n_samples']
            md += f"| {pair.replace('_', ' ')} | {chi2} | {p} | {v} | {dof} | {n} |\n"
        md += "\n"
    
    return md


def format_length_adjusted_markdown(length_adjusted_results: List[dict]) -> str:
    """Format length-adjusted analysis results."""
    if not length_adjusted_results:
        return ""
    
    md = "### Length-Adjusted Feature Analysis\n\n"
    md += "To account for potential biases due to sequence length variation, features were adjusted using linear regression.\n\n"
    
    # Summary table
    md += "| Feature | Length Correlation (r) | p-value | Significant? |\n"
    md += "|---------|----------------------|---------|-------------|\n"
    
    significant_features = []
    
    for result in length_adjusted_results:
        if result.get('length_correlation'):
            feat = result['feature']
            corr = result['length_correlation']
            r = f"{corr['pearson_r']:.3f}"
            p = f"{corr['p_value']:.3e}" if corr['p_value'] < 0.001 else f"{corr['p_value']:.3f}"
            sig = "Yes" if corr['significant'] else "No"
            md += f"| {feat} | {r} | {p} | {sig} |\n"
            
            if corr['significant']:
                significant_features.append(feat)
    
    md += "\n"
    
    if significant_features:
        md += f"**Features significantly correlated with length:** {', '.join(significant_features)}\n\n"
        md += "*Length-adjusted analyses were performed for these features. See visualizations for details.*\n\n"
    
    return md


def format_performance_metrics_markdown(perf_metrics: dict) -> str:
    """Format performance metrics in a readable table."""
    md = "### Performance Metrics\n\n"
    
    if 'stage_times' in perf_metrics:
        md += "#### Stage Execution Times\n\n"
        md += "| Stage | Duration (s) | Rate (items/s) |\n"
        md += "|-------|-------------|----------------|\n"
        
        total_time = sum(perf_metrics['stage_times'].values())
        
        for stage, duration in perf_metrics['stage_times'].items():
            rate = perf_metrics.get('processing_rates', {}).get(stage, None)
            rate_str = f"{rate:.1f}" if rate else "N/A"
            pct = (duration / total_time * 100) if total_time > 0 else 0
            md += f"| {stage.replace('_', ' ').title()} | {duration:.2f} ({pct:.1f}%) | {rate_str} |\n"
        md += "\n"
    
    if 'memory_usage' in perf_metrics:
        md += "#### Memory Usage\n\n"
        md += "| Checkpoint | Memory (MB) | Delta (MB) |\n"
        md += "|------------|-------------|------------|\n"
        
        prev_mem = None
        for checkpoint, mem in perf_metrics['memory_usage'].items():
            delta = f"+{mem - prev_mem:.1f}" if prev_mem else "N/A"
            md += f"| {checkpoint.replace('_', ' ').title()} | {mem:.1f} | {delta} |\n"
            prev_mem = mem
        md += "\n"
    
    return md


# --- Helper Functions ---

def get_nested(data: Any, keys: List[str], default: Any = None) -> Any:
    """Safely retrieve nested dictionary values."""
    val = data
    if data is None: 
        return default
    try:
        for key in keys:
            if isinstance(val, dict):
                val = val.get(key)
                if val is None: 
                    return default
            else:
                return default
        return val
    except Exception:
        return default


def format_plot_ref(alt_text: str, base_key: str, results_dict: dict, web_mode: bool) -> str:
    """Enhanced plot reference formatting."""
    plot_data = None
    plot_key = ""
    
    if web_mode:
        plot_key = base_key + "_b64" if not base_key.endswith("_b64") else base_key
        # Check common locations
        plot_data = get_nested(results_dict, ['feature_analysis', plot_key]) or \
                    get_nested(results_dict, [plot_key]) or \
                    get_nested(results_dict, ['ml_results', 'Species', plot_key]) or \
                    get_nested(results_dict, ['ml_results', 'Biotype', plot_key]) or \
                    get_nested(results_dict, ['feature_analysis', 'dim_reduction_plots', plot_key])
    else:
        plot_key = base_key if not base_key.endswith("_b64") else base_key.replace("_b64", "")
        plot_data = get_nested(results_dict, ['feature_analysis', plot_key]) or \
                    get_nested(results_dict, [plot_key]) or \
                    get_nested(results_dict, ['ml_results', 'Species', plot_key]) or \
                    get_nested(results_dict, ['ml_results', 'Biotype', plot_key]) or \
                    get_nested(results_dict, ['feature_analysis', 'dim_reduction_plots', plot_key])

    if plot_data:
        if web_mode: 
            return f"*[Plot generated: {alt_text} (See web interface)]*\n"
        elif isinstance(plot_data, str): 
            plot_data_clean = plot_data.replace('\\', '/')
            return f"![{alt_text}]({plot_data_clean})\n"
        else: 
            return f"*{alt_text} plot generated but path invalid.*\n"
    else: 
        return f"*{alt_text} plot not generated or available.*\n"


# --- Main Report Generation Function ---

def generate_markdown_report(summary_data: dict, report_filepath: Optional[str] = None) -> Optional[str]:
    """
    Enhanced Markdown report generation with comprehensive statistics and visualizations.
    """
    is_web_mode = IS_PYODIDE or (report_filepath is None)
    mode_str = "(Web Mode)" if is_web_mode else "(Native Mode)"
    logging.info(f"Generating enhanced Markdown report content {mode_str}...")

    # --- Build Markdown String ---
    md = ""
    try:
        # Header
        md += f"# FCGR Analysis Report {mode_str}\n\n"
        md += f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        
        # Get output dir path safely for native mode display
        native_output_dir = None
        if not is_web_mode and report_filepath:
             try: 
                 native_output_dir = os.path.abspath(os.path.dirname(report_filepath))
             except Exception: 
                 pass
        if native_output_dir: 
            md += f"*Output Directory: `{native_output_dir}`*\n\n"

        # Table of Contents
        md += "## Table of Contents\n\n"
        md += "1. [Configuration Summary](#configuration-summary)\n"
        md += "2. [Data Summary](#data-summary)\n"
        md += "3. [FCGR Examples](#fcgr-examples)\n"
        md += "4. [Feature Analysis](#feature-analysis)\n"
        md += "5. [Machine Learning Results](#machine-learning-results)\n"
        md += "6. [Performance Metrics](#performance-metrics)\n\n"

        # 1. Configuration Summary
        md += "## 1. Configuration Summary {#configuration-summary}\n\n"
        cfg = get_nested(summary_data, ['config'], {})
        
        md += "| Parameter | Value |\n"
        md += "|-----------|-------|\n"
        md += f"| Input Source | `{get_nested(summary_data, ['data_summary', 'input_source'], 'N/A')}` |\n"
        md += f"| FCGR k-mer size | {get_nested(cfg, ['FCGR_K'], 'N/A')} |\n"
        md += f"| FCGR Dimension | {get_nested(cfg, ['FCGR_DIM'], 'N/A')} × {get_nested(cfg, ['FCGR_DIM'], 'N/A')} |\n"
        md += f"| Sequence Length Range | {get_nested(cfg, ['MIN_SEQ_LEN'], '?')}-{get_nested(cfg, ['MAX_SEQ_LEN'], '?')} bp |\n"
        
        if not is_web_mode:
            md += f"| Parallel Jobs | {get_nested(cfg, ['N_JOBS'], 'N/A')} |\n"
            md += f"| GPU Available | {get_nested(cfg, ['gpu_available'], False)} |\n"
            md += f"| Mixed Precision | {get_nested(cfg, ['mixed_precision'], False)} |\n"
            md += f"| Cache Directory | {get_nested(cfg, ['DEFAULT_CACHE_DIR'], 'Disabled') or 'Disabled'} |\n"
        
        md += f"| Plotting Enabled | {get_nested(cfg, ['PLOTTING_ENABLED'], False)} |\n"
        md += f"| Target Source | `{get_nested(cfg, ['targets_source'], 'N/A')}` |\n\n"

        # 2. Data Summary
        md += "## 2. Data Summary {#data-summary}\n\n"
        ds = get_nested(summary_data, ['data_summary'], {})
        
        md += "### Processing Overview\n\n"
        md += f"- **Targets Provided:** {get_nested(ds, ['targets_requested'], 'N/A')}\n"
        md += f"- **Sequences Processed:** {get_nested(ds, ['sequences_processed'], 'N/A')}\n"
        md += f"- **Valid FCGRs Generated:** {get_nested(ds, ['fcgrs_generated'], 'N/A')}\n"
        md += f"- **Features Extracted:** {get_nested(ds, ['features_extracted_count'], 'N/A')}\n\n"
        
        # Species distribution
        species_counts = get_nested(ds, ['species_counts'])
        if isinstance(species_counts, dict) and species_counts:
            md += "### Species Distribution\n\n"
            md += "| Species | Count | Percentage |\n"
            md += "|---------|-------|------------|\n"
            total = sum(species_counts.values())
            for sp, co in sorted(species_counts.items(), key=lambda x: x[1], reverse=True):
                pct = (co / total * 100) if total > 0 else 0
                md += f"| {sp} | {co} | {pct:.1f}% |\n"
            md += "\n"
        
        # Biotype distribution
        biotype_counts = get_nested(ds, ['biotype_counts'])
        if isinstance(biotype_counts, dict) and biotype_counts:
            md += "### Biotype Distribution\n\n"
            md += "| Biotype | Count | Percentage |\n"
            md += "|---------|-------|------------|\n"
            total = sum(biotype_counts.values())
            for bt, co in sorted(biotype_counts.items(), key=lambda x: x[1], reverse=True):
                pct = (co / total * 100) if total > 0 else 0
                md += f"| {bt} | {co} | {pct:.1f}% |\n"
            md += "\n"
        
        # Processing times
        tm = get_nested(summary_data, ['timings'])
        if tm:
            md += "### Processing Times\n\n"
            md += "| Stage | Duration (s) | Percentage |\n"
            md += "|-------|-------------|------------|\n"
            total_time = tm.get('total_pipeline', 1)
            for k, v in tm.items():
                if isinstance(v, (int, float)) and k != 'total_pipeline':
                    pct = (v / total_time * 100) if total_time > 0 else 0
                    md += f"| {k.replace('_', ' ').title()} | {v:.2f} | {pct:.1f}% |\n"
            md += f"| **Total** | **{total_time:.2f}** | **100.0%** |\n\n"

        # 3. FCGR Examples
        md += "## 3. FCGR Examples {#fcgr-examples}\n\n"
        fcgr_examples_key = 'fcgr_examples_b64' if is_web_mode else 'fcgr_examples'
        fcgr_examples_data = get_nested(summary_data, [fcgr_examples_key], [])
        
        if fcgr_examples_data:
            if is_web_mode:
                md += f"*{len(fcgr_examples_data)} FCGR visualization plots generated (see web interface).*\n\n"
            else:
                for i, p in enumerate(fcgr_examples_data):
                    md += format_plot_ref(f"FCGR Example {i+1}", p, {}, False)
                md += "\n"
        else:
            md += "*No FCGR example plots generated.*\n\n"

        # 4. Feature Analysis
        md += "## 4. Feature Analysis {#feature-analysis}\n\n"
        fa = get_nested(summary_data, ['feature_analysis'], {})
        
        # 4.1 Sequence Length Distribution
        md += "### 4.1. Sequence Length Analysis\n\n"
        md += format_plot_ref("Sequence Length Distribution", 'length_distribution_plot', summary_data, is_web_mode) + "\n"
        
        # 4.2 Feature Correlations
        md += "### 4.2. Feature Correlations\n\n"
        md += format_plot_ref("Feature Correlation Heatmap", 'correlation_plot', summary_data, is_web_mode)
        md += format_plot_ref("Feature Correlation Network", 'correlation_network', summary_data, is_web_mode) + "\n"
        
        # 4.3 Feature Heatmaps
        md += "### 4.3. Feature Value Heatmaps\n\n"
        md += "#### By Species\n"
        md += format_plot_ref("Normalized Feature Values by Species", 'feature_heatmap_species', summary_data, is_web_mode)
        md += "\n#### By Biotype\n"
        md += format_plot_ref("Normalized Feature Values by Biotype", 'feature_heatmap_biotype', summary_data, is_web_mode) + "\n"
        
        # 4.4 Dimensionality Reduction
        md += "### 4.4. Dimensionality Reduction\n\n"
        dr_plots = get_nested(fa, ['dim_reduction_plots'], {})
        if dr_plots:
            md += "#### By Species\n"
            # Handle dict of methods
            for method in ['PCA', 't-SNE', 'UMAP']:
                method_key = f'species_{method}'
                if method_key in dr_plots:
                    md += f"##### {method}\n"
                    md += format_plot_ref(f"Species {method}", method_key, 
                                        {'feature_analysis': {'dim_reduction_plots': dr_plots}}, is_web_mode)
                    md += "\n"
            
            md += "\n#### By Biotype\n"
            for method in ['PCA', 't-SNE', 'UMAP']:
                method_key = f'biotype_{method}'
                if method_key in dr_plots:
                    md += f"##### {method}\n"
                    md += format_plot_ref(f"Biotype {method}", method_key,
                                        {'feature_analysis': {'dim_reduction_plots': dr_plots}}, is_web_mode)
                    md += "\n"
        

        # 4.5 Comprehensive Statistics
        comp_stats = get_nested(fa, ['comprehensive_stats'])
        if comp_stats:
            md += "### 4.5. Comprehensive Statistical Analysis\n\n"
            md += format_statistical_tests_markdown(comp_stats)
        
        # 4.6 Statistical Test Summary
        md += "### 4.6. Statistical Test Summary\n\n"
        
        # Pairwise comparisons heatmap
        md += "#### Pairwise Comparisons\n"
        md += format_plot_ref("Pairwise Comparisons Heatmap", 'pairwise_comparisons_heatmap', summary_data, is_web_mode)
        md += "\n*Detailed statistical test results are available in the CSV files saved in the data directory.*\n\n"
        
        # 4.7 Length-adjusted analyses
        length_adj = get_nested(fa, ['length_adjusted_analyses'], [])
        if length_adj:
            md += format_length_adjusted_markdown(length_adj)
        
        # 4.8 Example distribution plots note
        if is_web_mode and get_nested(fa, ['example_dist_plots_b64']):
            md += "*Individual feature distribution plots generated (see web interface).*\n\n"
        elif not is_web_mode:
            md += "*Individual feature distribution plots saved in figures directory.*\n\n"

        # 4.8 Feature Importance Analysis
        rf_importance_keys = [k for k in fa.keys() if 'rf_importance' in k]
        if rf_importance_keys:
            md += "### 4.8. Feature Importance Analysis\n\n"
            md += "Random Forest feature importances show which features contribute most to classification:\n\n"
            
            for key in rf_importance_keys:
                if 'species' in key:
                    md += "#### Species Classification\n"
                    md += format_plot_ref("Random Forest Feature Importances - Species", key, 
                                        {'feature_analysis': fa}, is_web_mode)
                elif 'biotype' in key:
                    md += "#### Biotype Classification\n"
                    md += format_plot_ref("Random Forest Feature Importances - Biotype", key,
                                        {'feature_analysis': fa}, is_web_mode)
            md += "\n"

        # 4.9 Learning Curves
        learning_curves_keys = [k for k in fa.keys() if 'learning_curves' in k]
        if learning_curves_keys:
            md += "### 4.9. Model Learning Curves\n\n"
            md += "Learning curves show how model performance improves with more training data:\n\n"
            
            for key in learning_curves_keys:
                md += format_plot_ref("SVM Learning Curves", key, 
                                    {'feature_analysis': fa}, is_web_mode)
            md += "\n"

 
        # 5. Machine Learning Results
        md += "## 5. Machine Learning Results {#machine-learning-results}\n\n"
        
        if IS_PYODIDE:
            md += "*Machine learning classification is not available in web environment.*\n\n"
        else:
            ml_res = get_nested(summary_data, ['ml_results'], {})
            
            for i, target in enumerate(['Species', 'Biotype']):
                md += f"### 5.{i+1}. {target} Classification\n\n"
                res = get_nested(ml_res, [target])
                
                if isinstance(res, dict) and get_nested(res, ['accuracy']) is not None:
                    # Summary metrics
                    acc = res['accuracy']
                    loss = res.get('loss', 'N/A')
                    
                    md += "#### Summary Metrics\n\n"
                    md += f"- **Test Accuracy:** {acc:.4f}\n"
                    md += f"- **Test Loss:** {loss if loss == 'N/A' else f'{loss:.4f}'}\n\n"
                    
                    # Training history plot
                    md += "#### Training History\n"
                    md += format_plot_ref(f"{target} Training History", 'history_plot', res, False) + "\n"
                    
                    # Confusion matrix
                    md += "#### Confusion Matrix\n"
                    md += format_plot_ref(f"{target} Confusion Matrix", 'cm_plot', res, False) + "\n"
                    
                    # Classification report
                    report = get_nested(res, ['report'])
                    if report:
                        md += format_clf_report_markdown(report, target)
                else:
                    md += f"*{target} classification results not available.*\n\n"
            
            # Feature importance plot if available
            importance_plot_sp = get_nested(ml_res, ['feature_importance_plot_sp'])
            if importance_plot_sp:
                md += "### 5.3. Feature Importance Analysis\n\n"
                md += format_plot_ref("Feature Importances (for species)", 'feature_importance_plot_sp', ml_res, False) + "\n"
            importance_plot_bt = get_nested(ml_res, ['feature_importance_plot_bt'])
            if importance_plot_bt:

                md += format_plot_ref("Feature Importances (for biotype)", 'feature_importance_plot_bt', ml_res, False) + "\n"
 
 
        # 6. Performance Metrics
        perf_metrics = get_nested(summary_data, ['performance_metrics'])
        if perf_metrics:
            md += "## 6. Performance Metrics {#performance-metrics}\n\n"
            md += format_performance_metrics_markdown(perf_metrics)
 
        # 7. Files Generated
        if not is_web_mode:
            md += "## 7. Generated Files\n\n"
            md += "The following files have been generated in the output directory:\n\n"
            
            # Data files
            md += "### Data Files\n"
            md += "- `data/processed_sequences_summary.csv`: Summary of processed sequences\n"
            md += "- `data/extracted_features.csv`: All extracted features\n"
            md += "- `data/feature_matrix.npz`: NumPy archive of feature matrix\n"
            md += "- `data/statistical_tests_results.csv`: Detailed statistical test results\n"
            md += "- `data/fcgr_analysis_summary.json`: Complete analysis results in JSON format\n\n"
            
            # Report files
            md += "### Reports\n"
            md += "- `fcgr_analysis_report.md`: This Markdown report\n"
            if get_nested(summary_data, ['report_paths', 'pdf_report']):
                md += "- `fcgr_analysis_report.pdf`: PDF version of this report\n"
            if get_nested(summary_data, ['report_paths', 'log_file']):
                md += f"- `{get_nested(summary_data, ['report_paths', 'log_file'])}`: Detailed execution log\n"
            md += "\n"
            
            # Figure files
            md += "### Figures\n"
            md += "All generated plots are saved in the `figures/` subdirectory.\n\n"
 
        # Footer
        md += "\n---\n\n"
        md += f"*Report generated by FCGR Analyzer v2.0*\n"
        md += f"*Total pipeline execution time: {get_nested(summary_data, ['timings', 'total_pipeline'], 0):.2f} seconds*\n"
        
        if not is_web_mode:
            md += f"\n*For questions or issues, please refer to the documentation or contact the developers.*\n"
 
        # --- Save or Return ---
        if not is_web_mode and report_filepath:
            try:
                # Ensure directory exists before writing
                report_dir = os.path.dirname(report_filepath)
                if report_dir: 
                    os.makedirs(report_dir, exist_ok=True)
                
                with open(report_filepath, 'w', encoding='utf-8') as f:
                    f.write(md)
                
                logging.info(f"Markdown report generated: {report_filepath}")
            except IOError as e:
                logging.error(f"Failed to write Markdown report to {report_filepath}: {e}")
        else:
            logging.info("Markdown report string generated.")
        
        return md
 
    except Exception as e:
        logging.error(f"Failed to generate Markdown report content: {e}", exc_info=True)
        error_md = f"# Report Generation Failed\n\nError: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        
        # Try writing error file if native
        if not is_web_mode and report_filepath:
             try:
                 with open(report_filepath.replace(".md", "_error.md"), 'w', encoding='utf-8') as f:
                     f.write(error_md)
             except Exception:
                 pass
        
        return error_md
 
 
def generate_pdf_report(markdown_filepath: str, pdf_filepath: str) -> bool:
    """Enhanced PDF generation with better formatting and image handling."""
    if IS_PYODIDE:
        logging.warning("PDF generation skipped: Pyodide env.")
        return True
    
    if not SUBPROCESS_AVAILABLE or not subprocess:
        logging.warning("PDF generation skipped: subprocess unavailable.")
        return True
   
    logging.info(f"Attempting PDF generation from '{os.path.basename(markdown_filepath)}'...")
    
    if not os.path.exists(markdown_filepath):
        logging.error(f"PDF skip: MD file missing: {markdown_filepath}")
        return False
    
    if PANDOC_CHECK_ENABLED and not check_pandoc_exists():
        return True  # Skipped gracefully
 
    try:
        report_title = os.path.splitext(os.path.basename(markdown_filepath))[0].replace('_', ' ').title()
        
        # Get the directory containing the markdown file
        markdown_dir = os.path.dirname(os.path.abspath(markdown_filepath))
        
        # Enhanced pandoc command with better formatting options
        command = [
            "pandoc", 
            markdown_filepath, 
            "-o", pdf_filepath,
            "--standalone",
            "--metadata", f"title={report_title}",
            "--metadata", "author=FCGR Analyzer",
            "--metadata", f"date={time.strftime('%Y-%m-%d')}",
            "--from", "markdown+yaml_metadata_block+implicit_figures+table_captions+grid_tables+pipe_tables+raw_html",
            "--to", "pdf",
            "--pdf-engine=xelatex",  # Better unicode support
            "--highlight-style=tango",  # Code highlighting
            "--toc",  # Table of contents
            "--toc-depth=3",
            "-V", "geometry:margin=1in",
            "-V", "fontsize=11pt",
            "-V", "documentclass=report",
            "-V", "colorlinks=true",
            "-V", "linkcolor=blue",
            "-V", "urlcolor=blue",
            "-V", "toccolor=gray",
            # FIXED: Better image handling
            f"--resource-path={markdown_dir}:{os.path.join(markdown_dir, 'figures')}",
            "--embed-resources",
            "--standalone",
            # Allow larger images
            "-V", "graphics=true",
            "-V", "float=true"
        ]
        
        logging.info(f"Executing Pandoc with resource path: {markdown_dir}")
 
        # Ensure the output directory exists
        pdf_dir = os.path.dirname(pdf_filepath)
        if pdf_dir:
            os.makedirs(pdf_dir, exist_ok=True)
 
        # Change to markdown directory for relative paths
        original_cwd = os.getcwd()
        try:
            os.chdir(markdown_dir)
            
            proc = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                check=True, 
                encoding='utf-8', 
                errors='ignore', 
                timeout=300
            )
            
        finally:
            os.chdir(original_cwd)
        
        # Check stderr for warnings even on success
        if proc.stderr:
            # Filter out expected warnings
            stderr_lines = proc.stderr.strip().split('\n')
            important_warnings = []
            for line in stderr_lines:
                if not any(skip in line for skip in ['Missing character:', 'Hyper reference']):
                    important_warnings.append(line)
            
            if important_warnings:
                logging.warning(f"Pandoc warnings:\n" + '\n'.join(important_warnings))
 
        if os.path.exists(pdf_filepath) and os.path.getsize(pdf_filepath) > 0:
             logging.info(f"PDF report generated successfully: {pdf_filepath}")
             return True
        else:
             logging.error(f"Pandoc command seemed successful, but PDF file is missing or empty: {pdf_filepath}")
             return False
             
    except Exception as e:
        logging.error(f"PDF generation error: {e}", exc_info=True)
        return False

# --- Save Results Summary (Native focused) ---
def save_results_summary(summary_data: dict, json_filepath: str):
    """Enhanced JSON summary saving with pretty printing."""
    if IS_PYODIDE:
        logging.info("JSON file saving skipped in Pyodide.")
        return
    
    if not json_filepath:
        logging.error("Cannot save JSON summary: No filepath provided.")
        return
 
    logging.info(f"Saving results summary JSON to: {json_filepath}")
    
    try:
        # Ensure directory exists before writing
        json_dir = os.path.dirname(json_filepath)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
        
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(
                summary_data, 
                f, 
                indent=4, 
                default=convert_numpy_for_json,
                sort_keys=True
            )
        
        logging.info(f"Results summary JSON saved successfully ({os.path.getsize(json_filepath) / 1024:.1f} KB).")
        
    except TypeError as e:
        logging.error(f"Failed JSON serialization: {e}", exc_info=False)
    except IOError as e:
        logging.error(f"Failed JSON file write to {json_filepath}: {e}", exc_info=False)
    except Exception as e:
        logging.error(f"Unexpected error saving JSON to {json_filepath}: {e}", exc_info=True)