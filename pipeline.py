# -*- coding: utf-8 -*-
"""
Enhanced Main Pipeline Orchestration for FCGR Analysis.
Supports genome sampler input format and includes performance optimizations.
"""
import os
import time
import logging
import pandas as pd
import numpy as np
import json
import traceback
import io
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import gc  # For memory management

# Import framework components and check flags
from . import config as cfg
from . import utils
from . import data_acquisition as da
from . import fcgr as fcg
from . import feature_extraction as fe
from . import analysis as an
from . import machine_learning as ml
from . import reporting as rp
from .utils import IS_PYODIDE # Import the flag

# --- Conditional Joblib Import ---
PARALLEL_AVAILABLE = False
Parallel = None; delayed = None
if not IS_PYODIDE and utils.JOBLIB_AVAILABLE:
    try:
        from joblib import Parallel as jl_Parallel, delayed as jl_delayed
        Parallel = jl_Parallel; delayed = jl_delayed
        PARALLEL_AVAILABLE = True
        logging.debug("Joblib Parallel imported.")
    except ImportError:
        logging.warning("Joblib available but Parallel/delayed import failed.")

# --- Performance Monitoring ---
class PerformanceMonitor:
    """Monitor pipeline performance metrics."""
    def __init__(self):
        self.metrics = {
            'stage_times': {},
            'memory_usage': {},
            'data_sizes': {},
            'processing_rates': {}
        }
        self.stage_start_time = None
        self.current_stage = None
    
    def start_stage(self, stage_name: str):
        """Mark the start of a processing stage."""
        self.current_stage = stage_name
        self.stage_start_time = time.time()
        
        # Record memory usage at start
        try:
            import psutil
            process = psutil.Process()
            self.metrics['memory_usage'][f"{stage_name}_start"] = process.memory_info().rss / 1024 / 1024  # MB
        except:
            pass
    
    def end_stage(self, items_processed: Optional[int] = None):
        """Mark the end of a processing stage."""
        if self.current_stage and self.stage_start_time:
            elapsed = time.time() - self.stage_start_time
            self.metrics['stage_times'][self.current_stage] = elapsed
            
            # Calculate processing rate if applicable
            if items_processed and elapsed > 0:
                self.metrics['processing_rates'][self.current_stage] = items_processed / elapsed
            
            # Record memory usage at end
            try:
                import psutil
                process = psutil.Process()
                self.metrics['memory_usage'][f"{self.current_stage}_end"] = process.memory_info().rss / 1024 / 1024  # MB
            except:
                pass
            
            logging.info(f"Stage '{self.current_stage}' completed in {elapsed:.2f}s")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return self.metrics


# --- Enhanced Pipeline Function ---
def run_pipeline(output_dir: Optional[str],
                 cache_dir: Optional[str],
                 targets: List[Union[tuple, dict]],
                 config_dict: dict,
                 input_fasta: Optional[str] = None) -> dict:
    """
    Enhanced pipeline with genome sampler support and performance optimizations.
    
    Args:
        output_dir: Path for saving files (Native mode only).
        cache_dir: Path for caching (Native mode only).
        targets: List of sequence targets (tuples) or None if using input_fasta.
        config_dict: Dictionary containing configuration parameters.
        input_fasta: Optional path to genome sampler FASTA file.
    
    Returns:
        Dictionary containing the summary of results and timings.
    """
    pipeline_start_time = time.time()
    monitor = PerformanceMonitor()
    
    logging.info(f"--- Starting Enhanced FCGR Analysis Pipeline (Env: {'Pyodide' if IS_PYODIDE else 'Native'}) ---")

    # --- Determine Environment Settings ---
    is_web_mode = IS_PYODIDE
    n_jobs = 1 if is_web_mode or not PARALLEL_AVAILABLE else config_dict.get('N_JOBS', cfg.N_JOBS)
    plot_output_format = 'base64' if is_web_mode else 'file'
    use_cache = not is_web_mode and cache_dir and utils.JOBLIB_AVAILABLE
    try_pdf = not is_web_mode and config_dict.get('PANDOC_CHECK_ENABLED', False)
    plotting_enabled = config_dict.get('PLOTTING_ENABLED', cfg.PLOTTING_ENABLED)

    logging.info(f"Runtime Settings: n_jobs={n_jobs}, plot_format='{plot_output_format}', use_cache={use_cache}, try_pdf={try_pdf}, plotting={plotting_enabled}")

    # --- Setup Output Dirs (native only) ---
    figures_dir = None; data_dir = None
    log_filepath = None
    if not is_web_mode:
        if not output_dir: 
            output_dir = cfg.DEFAULT_OUTPUT_DIR
        try:
            # Resolve relative paths before creating directories
            output_dir = os.path.abspath(os.path.expanduser(output_dir))
            figures_dir = os.path.join(output_dir, cfg.FIGURES_SUBDIR)
            data_dir = os.path.join(output_dir, cfg.DATA_SUBDIR)
            os.makedirs(output_dir, exist_ok=True)
            if plotting_enabled:
                os.makedirs(figures_dir, exist_ok=True)
            os.makedirs(data_dir, exist_ok=True)
            
            # Setup file logging
            log_filepath = os.path.join(output_dir, f"fcgr_analysis_{time.strftime('%Y%m%d_%H%M%S')}.log")
            file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
            file_handler.setLevel(logging.DEBUG)
            logging.getLogger().addHandler(file_handler)
            logging.info(f"Log file created: {log_filepath}")
            
            logging.info(f"Native Output Dir: {output_dir}")
        except OSError as e:
            logging.error(f"Failed creating output dirs: {e}. File saving disabled.")
            output_dir = None; figures_dir = None; data_dir = None
    else:
        output_dir = "/virtual"; figures_dir = "/virtual/figures"; data_dir = "/virtual/data"

    # --- Initialize Caching (native only) ---
    da.memory_cache = None
    if use_cache and cache_dir:
         memory = utils.setup_joblib_cache(cache_dir)
         if memory: 
             da.memory_cache = memory

    # --- Initialize Results Summary ---
    plot_key_suffix = "_b64" if is_web_mode else ""
    results_summary = {
        'config': {k: v for k, v in config_dict.items() if isinstance(v, (int, float, str, bool, list, dict, type(None)))},
        'data_summary': {
            'targets_requested': len(targets) if targets else 0,
            'sequences_processed': 0,
            'fcgrs_generated': 0,
            'features_extracted_count': 0,
            'species_counts': None,
            'biotype_counts': None,
            'input_source': 'targets' if targets else 'genome_sampler_fasta'
        },
        'timings': {},
        'performance_metrics': {},
        f'fcgr_examples{plot_key_suffix}': [],
        'feature_analysis': {
            'stats_tests': [],
            f'correlation_plot{plot_key_suffix}': None,
            f'correlation_network{plot_key_suffix}': None,
            f'length_distribution_plot{plot_key_suffix}': None,
            f'pairwise_comparisons_heatmap{plot_key_suffix}': None,
            f'feature_heatmap_species{plot_key_suffix}': None,
            f'feature_heatmap_biotype{plot_key_suffix}': None,
            'length_adjusted_analyses': [],
            'comprehensive_stats': {},
            'example_dist_plots_b64': {} if is_web_mode else None,
            'dim_reduction_plots': {}
        },
        'ml_results': {'Species': None, 'Biotype': None},
        'error': None,
    }
    
    # Add native-only keys if applicable
    if not is_web_mode:
        results_summary['report_paths'] = {
            'json_summary': None,
            'markdown_report': None,
            'pdf_report': None,
            'log_file': None
        }
        results_summary['data_summary']['sequence_summary_path'] = None
        results_summary['data_summary']['features_path'] = None
    else:
        results_summary['report_md_content'] = None
        results_summary['logs'] = None

    # --- Pipeline Steps ---
    df = pd.DataFrame()
    feature_cols = []
    
    try:
        # --- 1. Data Acquisition ---
        monitor.start_stage("data_acquisition")
        logging.info("--- Step 1: Data Acquisition ---")
        
        # Handle genome sampler input
        if input_fasta and os.path.exists(input_fasta):
            logging.info(f"Loading sequences from genome sampler FASTA: {input_fasta}")
            sequences_data = da.parse_fasta_file(input_fasta)
            
            if sequences_data:
                # Convert to target format for processing
                targets = []
                for seq_data in sequences_data:
                    # Create target tuple in expected format
                    target = (
                        seq_data['species'],
                        seq_data['biotype'],
                        'genome_sampler',
                        seq_data  # Pass the whole dict as identifier
                    )
                    targets.append(target)
                
                logging.info(f"Loaded {len(targets)} sequences from genome sampler file")
                results_summary['data_summary']['targets_requested'] = len(targets)
                results_summary['data_summary']['input_source'] = f'genome_sampler: {os.path.basename(input_fasta)}'
        
        # Configure Entrez if needed
        if any(t[2] == 'accession' for t in targets if len(t) >= 3):
            try:
                req_session = utils.setup_requests_session(
                    retries=config_dict.get('REQUEST_RETRIES', cfg.REQUEST_RETRIES),
                    backoff_factor=config_dict.get('REQUEST_BACKOFF_FACTOR', cfg.REQUEST_BACKOFF_FACTOR),
                    status_forcelist=config_dict.get('REQUEST_STATUS_FORCELIST', cfg.REQUEST_STATUS_FORCELIST)
                )
                da.configure_entrez(
                    config_dict.get('ENTREZ_EMAIL', cfg.ENTREZ_EMAIL),
                    config_dict.get('ENTREZ_API_KEY', cfg.ENTREZ_API_KEY),
                    req_session
                )
            except ValueError as e:
                raise ValueError(f"Entrez config error: {e}")
            except Exception as se:
                logging.warning(f"Entrez session setup failed: {se}")
        
        # Process targets with parallel support
        if PARALLEL_AVAILABLE and n_jobs > 1 and len(targets) > 10:
            processed_data_list = da.process_targets_parallel(targets, max_workers=n_jobs, config_dict=config_dict)
        else:
            processed_data_list = [da.process_target(t, config_dict) for t in targets]
        
        valid_processed_data = [item for item in processed_data_list if item is not None]
        
        if not valid_processed_data:
            raise ValueError("No valid sequences obtained after processing targets.")
        
        df = pd.DataFrame(valid_processed_data)
        results_summary['data_summary']['sequences_processed'] = len(df)
        results_summary['data_summary']['species_counts'] = df['species'].value_counts().to_dict()
        results_summary['data_summary']['biotype_counts'] = df['biotype'].value_counts().to_dict()
        
        # Save summary CSV (native only)
        if not is_web_mode and data_dir and output_dir:
            summary_csv_path = os.path.join(data_dir, "processed_sequences_summary.csv")
            try:
                df[['id', 'original_id', 'species', 'biotype', 'length']].to_csv(summary_csv_path, index=False)
                results_summary['data_summary']['sequence_summary_path'] = os.path.relpath(summary_csv_path, output_dir)
                logging.info(f"Sequence summary saved.")
            except Exception as e:
                logging.warning(f"Save sequence summary CSV failed: {e}")
        
        monitor.end_stage(len(valid_processed_data))
        results_summary['timings']['data_acquisition'] = monitor.metrics['stage_times']['data_acquisition']
        
        # --- 2. Generate FCGRs ---
        monitor.start_stage("fcgr_generation")
        logging.info(f"--- Step 2: Generating FCGRs (k={cfg.FCGR_K}) ---")
        
        sequences = df['sequence'].tolist()
        
        # Use parallel processing for FCGR generation
        if PARALLEL_AVAILABLE and n_jobs > 1 and len(sequences) > 20:
            fcgr_results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(fcg.generate_fcgr)(s, cfg.FCGR_K) for s in sequences
            )
        else:
            fcgr_results = [fcg.generate_fcgr(s, cfg.FCGR_K) for s in sequences]
        
        df['fcgr'] = fcgr_results
        initial_count = len(df)
        
        # Filter valid FCGRs
        valid_fcgr_mask = df['fcgr'].apply(
            lambda x: isinstance(x, np.ndarray) and 
                     x.shape == (cfg.FCGR_DIM, cfg.FCGR_DIM) and 
                     np.any(np.abs(x) > cfg.EPSILON)
        )
        df = df[valid_fcgr_mask].reset_index(drop=True)
        
        results_summary['data_summary']['fcgrs_generated'] = len(df)
        filtered_count = initial_count - len(df)
        
        if df.empty:
            raise ValueError("No valid FCGR matrices generated after filtering.")
        
        monitor.end_stage(len(df))
        results_summary['timings']['fcgr_generation'] = monitor.metrics['stage_times']['fcgr_generation']
        logging.info(f"FCGR Generation complete ({len(df)} valid, {filtered_count} filtered).")
        
        # --- 3. Extract Features ---
        monitor.start_stage("feature_extraction")
        logging.info("--- Step 3: Extracting Features ---")
        
        fcgr_matrices = df['fcgr'].tolist()
        
        # Parallel feature extraction
        if PARALLEL_AVAILABLE and n_jobs > 1 and len(fcgr_matrices) > 20:
            feature_results_list = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(fe.extract_all_features)(m) for m in fcgr_matrices
            )
        else:
            feature_results_list = [fe.extract_all_features(m) for m in fcgr_matrices]
        
        features_df = pd.DataFrame(feature_results_list, index=df.index)
        df = pd.concat([df, features_df], axis=1)
        
        base_cols = ['id', 'original_id', 'species', 'biotype', 'sequence', 'length', 'fcgr']
        feature_cols = [col for col in df.columns if col not in base_cols]
        results_summary['data_summary']['features_extracted_count'] = len(feature_cols)
        
        logging.info(f"Extracted {len(feature_cols)} features.")
        
        if not is_web_mode and data_dir and output_dir:
            features_csv_path = os.path.join(data_dir, "extracted_features.csv")
            try:
                df.drop(columns=['sequence', 'fcgr'], errors='ignore').to_csv(features_csv_path, index=False)
                results_summary['data_summary']['features_path'] = os.path.relpath(features_csv_path, output_dir)
                logging.info(f"Features saved.")
                
                # Also save the complete feature matrix for re-analysis
                feature_matrix_path = os.path.join(data_dir, "feature_matrix.npz")
                feature_matrix = df[feature_cols].values
                metadata = {
                    'feature_names': feature_cols,
                    'species': df['species'].values,
                    'biotype': df['biotype'].values,
                    'length': df['length'].values,
                    'original_id': df['original_id'].values
                }
                np.savez_compressed(feature_matrix_path, 
                                  features=feature_matrix,
                                  **metadata)
                logging.info(f"Feature matrix saved to {feature_matrix_path}")
                
            except Exception as e:
                logging.warning(f"Save features CSV failed: {e}")
        
        monitor.end_stage(len(feature_cols))
        results_summary['timings']['feature_extraction'] = monitor.metrics['stage_times']['feature_extraction']

        
        # --- 4. Analysis & Visualization ---
        monitor.start_stage("analysis_visualization")
        logging.info(f"--- Step 4: Analysis & Visualization (Format: {plot_output_format}) ---")
        
        fcgr_plot_key = f'fcgr_examples{plot_key_suffix}'
        results_summary[fcgr_plot_key] = []
        
        if plotting_enabled and len(df) > 0 and an.PLOTTING_LIBS_AVAILABLE:
            # FCGR example plots
            num_examples = min(cfg.FCGR_PLOT_EXAMPLES, len(df))
            indices_to_plot = df.sample(n=num_examples, random_state=cfg.RANDOM_STATE).index
            
            for i, idx in enumerate(indices_to_plot):
                row = df.loc[idx]
                title = f"FCGR {i+1}: {row['species']} - {row['biotype']}"
                safe_base = utils.safe_filename(f"fcgr_example_{i+1}_{row['species']}_{row['biotype']}_{row['original_id']}")
                save_target = 'base64' if is_web_mode else os.path.join(figures_dir or ".", f"{safe_base}.png")
                
                plot_result = an.plot_fcgr(
                    row['fcgr'], title, save_target, 
                    output_format=plot_output_format,
                    show_grid=True, annotate_regions=True
                )
                
                if plot_result:
                    results_summary[fcgr_plot_key].append(plot_result)
            
            # Feature analysis
            if not df.empty and feature_cols:
                df_features_only = df[feature_cols].copy()
                
                # 1. Sequence length distribution plot
                length_plot = an.plot_sequence_length_distribution(
                    df, figures_dir, output_format=plot_output_format
                )
                if length_plot:
                    results_summary['feature_analysis'][f'length_distribution_plot{plot_key_suffix}'] = length_plot
                
                # 2. Correlation heatmap
                corr_target = 'base64' if is_web_mode else figures_dir
                corr_plot_result = an.plot_correlation_heatmap(
                    df_features_only, corr_target, 
                    output_format=plot_output_format,
                    method='pearson', cluster=True
                )
                if corr_plot_result:
                    results_summary['feature_analysis'][f'correlation_plot{plot_key_suffix}'] = corr_plot_result
                
                # 3. Feature correlation network
                corr_network = an.plot_feature_correlations_network(
                    df, feature_cols, figures_dir, 
                    output_format=plot_output_format,
                    correlation_threshold=0.5
                )
                if corr_network:
                    results_summary['feature_analysis'][f'correlation_network{plot_key_suffix}'] = corr_network
                
                # 4. Normalized feature heatmaps
                heatmap_species = an.plot_feature_heatmap_normalized(
                    df, feature_cols[:20], 'species', figures_dir,
                    output_format=plot_output_format
                )
                if heatmap_species:
                    results_summary['feature_analysis'][f'feature_heatmap_species{plot_key_suffix}'] = heatmap_species
                    
                heatmap_biotype = an.plot_feature_heatmap_normalized(
                    df, feature_cols[:20], 'biotype', figures_dir,
                    output_format=plot_output_format
                )
                if heatmap_biotype:
                    results_summary['feature_analysis'][f'feature_heatmap_biotype{plot_key_suffix}'] = heatmap_biotype
                
                # 5. Statistical tests
                stats_tests_results = []
                dist_plot_results_b64 = {}
                length_adjusted_results = []
                
                # Run comprehensive statistics
                comprehensive_stats = an.run_comprehensive_statistics(
                    df, feature_cols, ['species', 'biotype']
                )
                results_summary['feature_analysis']['comprehensive_stats'] = comprehensive_stats
                
                # Feature-wise analysis (limit to top 10 for performance)
                for feature in feature_cols[:10]:
                    # Standard analysis by species
                    stats_res_sp, plot_out_sp = an.run_feature_analysis(
                        df, feature, 'species', figures_dir, 
                        output_format=plot_output_format,
                        statistical_tests=['kruskal', 'anova', 'mannwhitney']
                    )
                    stats_tests_results.append(stats_res_sp)
                    
                    if is_web_mode and plot_out_sp:
                        dist_plot_results_b64[f"{feature}_species"] = plot_out_sp
                    elif not is_web_mode and plot_out_sp and 'plot_path' not in stats_res_sp:
                        stats_res_sp['plot_path'] = plot_out_sp
                    
                    # Standard analysis by biotype
                    stats_res_bt, plot_out_bt = an.run_feature_analysis(
                        df, feature, 'biotype', figures_dir,
                        output_format=plot_output_format,
                        statistical_tests=['kruskal', 'anova', 'mannwhitney']
                    )
                    stats_tests_results.append(stats_res_bt)
                    
                    if is_web_mode and plot_out_bt:
                        dist_plot_results_b64[f"{feature}_biotype"] = plot_out_bt
                    elif not is_web_mode and plot_out_bt and 'plot_path' not in stats_res_bt:
                        stats_res_bt['plot_path'] = plot_out_bt
                    
                    # Length-adjusted analysis
                    if feature not in ['length', 'center_mass_x', 'center_mass_y']:
                        adj_res, adj_plot = an.run_length_adjusted_analysis(
                            df, feature, figures_dir, output_format=plot_output_format
                        )
                        length_adjusted_results.append(adj_res)
                        if adj_plot:
                            if is_web_mode:
                                dist_plot_results_b64[f"{feature}_length_adjusted"] = adj_plot
                            else:
                                adj_res['plot_path'] = adj_plot
                
                results_summary['feature_analysis']['stats_tests'] = stats_tests_results
                results_summary['feature_analysis']['length_adjusted_analyses'] = length_adjusted_results
                
                if is_web_mode:
                    results_summary['feature_analysis']['example_dist_plots_b64'] = dist_plot_results_b64
                
                # 6. Pairwise comparisons heatmap
                pairwise_heatmap = an.plot_pairwise_comparisons_heatmap(
                    stats_tests_results, figures_dir, output_format=plot_output_format
                )
                if pairwise_heatmap:
                    results_summary['feature_analysis'][f'pairwise_comparisons_heatmap{plot_key_suffix}'] = pairwise_heatmap
                
                # Save all statistical results to CSV for later use
                if not is_web_mode and data_dir:
                    try:
                        # Save comprehensive stats
                        stats_df_rows = []
                        for test_result in stats_tests_results:
                            if 'tests' in test_result and test_result['tests']:
                                row = {
                                    'feature': test_result['feature'],
                                    'group_by': test_result['group_by'],
                                    'effect_size': test_result.get('effect_size', None)
                                }
                                # Add test results
                                for test_name, test_data in test_result['tests'].items():
                                    if isinstance(test_data, dict):
                                        for key, value in test_data.items():
                                            if not isinstance(value, (list, dict)):
                                                row[f"{test_name}_{key}"] = value
                                stats_df_rows.append(row)
                        
                        if stats_df_rows:
                            stats_df = pd.DataFrame(stats_df_rows)
                            stats_csv_path = os.path.join(data_dir, "statistical_tests_results.csv")
                            stats_df.to_csv(stats_csv_path, index=False)
                            logging.info(f"Statistical test results saved to {stats_csv_path}")
                            
                    except Exception as e:
                        logging.warning(f"Failed to save statistical results: {e}")
                
                # 7. Dimensionality Reduction
                if an.SKLEARN_AVAILABLE:
                    try:
                        variances = df_features_only.var(ddof=0)
                        features_to_reduce = variances[variances > cfg.EPSILON].index.tolist()
                    except Exception:
                        features_to_reduce = []
                    
                    if len(features_to_reduce) >= 2 and len(df) >= 10:
                        scaler = an.StandardScaler()
                        scaled_features = scaler.fit_transform(df_features_only[features_to_reduce])
                        
                        labels_species = df.loc[df_features_only[features_to_reduce].index, 'species'].values
                        labels_biotype = df.loc[df_features_only[features_to_reduce].index, 'biotype'].values
                        
                        dr_target = 'base64' if is_web_mode else figures_dir
                        
                        # Include UMAP if available
                        dr_methods = ['PCA', 't-SNE']
                        if an.UMAP is not None:
                            dr_methods.append('UMAP')
                        
                        # Species DR
                        sp_dr_res = an.run_dimensionality_reduction(
                            scaled_features, labels_species, 'Species', 
                            dr_target, methods=dr_methods,
                            output_format=plot_output_format
                        )
                        if sp_dr_res:
                            results_summary['feature_analysis']['dim_reduction_plots'][f'species{plot_key_suffix}'] = sp_dr_res
                        
                        # Biotype DR
                        bt_dr_res = an.run_dimensionality_reduction(
                            scaled_features, labels_biotype, 'Biotype', 
                            dr_target, methods=dr_methods,
                            output_format=plot_output_format
                        )
                        if bt_dr_res:
                            results_summary['feature_analysis']['dim_reduction_plots'][f'biotype{plot_key_suffix}'] = bt_dr_res
                    else:
                        logging.warning("Skipping DR: Not enough features/samples with variance.")
                else:
                    logging.warning("Skipping DR: Sklearn not available.")

                # 8. Feature Importance Analysis (using Random Forest)
                if ml.SKLEARN_AVAILABLE and len(df) >= 20 and not is_web_mode:
                    try:
                        from sklearn.ensemble import RandomForestClassifier
                        
                        # Prepare data
                        X_features = df[feature_cols].values
                        
                        # Train RF for species classification
                        rf_species = RandomForestClassifier(
                            n_estimators=100, 
                            random_state=cfg.RANDOM_STATE,
                            n_jobs=n_jobs
                        )
                        rf_species.fit(X_features, df['species'])
                        
                        # Plot feature importance
                        importance_plot = an.plot_feature_importance(
                            feature_cols, 
                            rf_species.feature_importances_,
                            title="Feature Importances - Species Classification (Random Forest)",
                            figures_dir=figures_dir,
                            output_format=plot_output_format,
                            top_n=min(20, len(feature_cols))
                        )
                        if importance_plot:
                            results_summary['feature_analysis'][f'rf_importance_species{plot_key_suffix}'] = importance_plot
                        
                        # Also for biotype if enough samples per class
                        biotype_counts = df['biotype'].value_counts()
                        if biotype_counts.min() >= 3:
                            rf_biotype = RandomForestClassifier(
                                n_estimators=100, 
                                random_state=cfg.RANDOM_STATE,
                                n_jobs=n_jobs
                            )
                            rf_biotype.fit(X_features, df['biotype'])
                            
                            importance_plot_bt = an.plot_feature_importance(
                                feature_cols, 
                                rf_biotype.feature_importances_,
                                title="Feature Importances - Biotype Classification (Random Forest)",
                                figures_dir=figures_dir,
                                output_format=plot_output_format,
                                top_n=min(20, len(feature_cols))
                        )
                        if importance_plot_bt:
                            results_summary['feature_analysis'][f'rf_importance_biotype{plot_key_suffix}'] = importance_plot_bt
                                
                    except Exception as e:
                        logging.warning(f"Feature importance analysis failed: {e}")
                else:
                    logging.warning(f"Feature importance analysis skipped: number of features is {len(df)} and/or scikit-learn is not available")

                # 9. Learning Curves (if sufficient data)
                if ml.SKLEARN_AVAILABLE and len(df) >= 50 and len(feature_cols) >= 5 and not is_web_mode:
                    try:
                        from sklearn.model_selection import learning_curve
                        from sklearn.svm import SVC
                        from sklearn.preprocessing import StandardScaler
                        
                        # Prepare and scale data
                        X_scaled = StandardScaler().fit_transform(df[feature_cols])
                        
                        # Calculate learning curves for species classification
                        train_sizes, train_scores, val_scores = learning_curve(
                            SVC(kernel='rbf', C=1.0, random_state=cfg.RANDOM_STATE),
                            X_scaled, df['species'],
                            cv=5,
                            n_jobs=n_jobs,
                            train_sizes=np.linspace(0.2, 1.0, 8),
                            scoring='accuracy'
                        )
                        
                        learning_plot = an.plot_learning_curves(
                            train_sizes, train_scores, val_scores,
                            title="Learning Curves - SVM Species Classification",
                            figures_dir=figures_dir,
                            output_format=plot_output_format
                        )
                        if learning_plot:
                            results_summary['feature_analysis'][f'learning_curves_species{plot_key_suffix}'] = learning_plot
                            
                    except Exception as e:
                        logging.warning(f"Learning curves analysis failed: {e}")
                else:
                    logging.warning(f"Learning curve skipped: number of features is {len(df)} and/or scikit-learn is not available")


            else:
                logging.warning("Skipping analysis plots: No features or data.")
        else:
            logging.info("Skipping plotting: Plotting disabled or no data.")
        
        monitor.end_stage()
        results_summary['timings']['analysis_plotting'] = monitor.metrics['stage_times']['analysis_visualization']
        
        # --- 5. Machine Learning (Conditional) ---
        monitor.start_stage("machine_learning")
        
        if not is_web_mode:
            logging.info("--- Step 5: Machine Learning Classification (Native) ---")
            
            if ml.TENSORFLOW_AVAILABLE and ml.SKLEARN_AVAILABLE and len(df) >= 20:
                try:
                    X_cnn_data = np.stack(df['fcgr'].values, axis=0).astype(np.float32)
                    y_species_labels = df['species'].values
                    y_biotype_labels = df['biotype'].values
                    
                    # Species classification
                    sp_encoder, sp_results = ml.run_classification(
                        X_cnn_data, y_species_labels, "Species", 
                        figures_dir, output_format=plot_output_format
                    )
                    results_summary['ml_results']['Species'] = sp_results
                    
                    # Biotype classification
                    bt_encoder, bt_results = ml.run_classification(
                        X_cnn_data, y_biotype_labels, "Biotype", 
                        figures_dir, label_encoder=None, 
                        output_format=plot_output_format
                    )
                    results_summary['ml_results']['Biotype'] = bt_results
                    
                    # Add feature importance analysis if possible
                    if hasattr(sp_encoder, 'feature_importances_'):
                        importance_plot_sp = an.plot_feature_importance(
                            feature_cols, sp_encoder.feature_importances_,
                            title="Feature Importances - Species Classification",
                            figures_dir=figures_dir,
                            output_format=plot_output_format
                        )
                        if importance_plot:
                            results_summary['ml_results']['feature_importance_plot_sp'] = importance_plot_sp

                    if hasattr(bt_encoder, 'feature_importances_'):
                        importance_plot_bt = an.plot_feature_importance(
                            feature_cols, bt_encoder.feature_importances_,
                            title="Feature Importances - Biotype Classification",
                            figures_dir=figures_dir,
                            output_format=plot_output_format
                        )
                        if importance_plot:
                            results_summary['ml_results']['feature_importance_plot_bt'] = importance_plot_bt

                except Exception as e:
                    logging.error(f"ML phase failed: {e}", exc_info=True)
            else:
                logging.warning("ML Skipped (Native): Dependencies missing or insufficient data.")
        else:
            logging.info("--- Step 5: Machine Learning Classification---")
        
        monitor.end_stage()
        results_summary['timings']['machine_learning'] = monitor.metrics['stage_times']['machine_learning']
 # --- 6. Finalize and Generate Reports ---
        monitor.start_stage("reporting")
        logging.info(f"--- Step 6: Finalizing and Reporting ---")
        
        # Add performance metrics
        results_summary['performance_metrics'] = monitor.get_summary()
        results_summary['timings']['total_pipeline'] = time.time() - pipeline_start_time
        
        # Add log file path if native
        if not is_web_mode and log_filepath and os.path.exists(log_filepath):
            results_summary['report_paths']['log_file'] = os.path.relpath(log_filepath, output_dir)
        
        # Generate Markdown Report
        md_report_path = os.path.join(output_dir or ".", "fcgr_analysis_report.md")
        md_content = rp.generate_markdown_report(
            results_summary, 
            md_report_path if not is_web_mode else None
        )
        
        if is_web_mode:
            results_summary['report_md_content'] = md_content
        elif md_content and output_dir:
            results_summary['report_paths']['markdown_report'] = os.path.basename(md_report_path)
        
        # Save JSON summary (native only)
        if not is_web_mode and data_dir and output_dir:
            json_filename = "fcgr_analysis_summary.json"
            json_filepath = os.path.join(data_dir, json_filename)
            rp.save_results_summary(results_summary, json_filepath)
            
            if os.path.exists(json_filepath):
                results_summary['report_paths']['json_summary'] = os.path.join(
                    os.path.basename(data_dir), json_filename
                )
        
        # Generate PDF Report (native only)
        if try_pdf and not is_web_mode and output_dir and results_summary.get('report_paths', {}).get('markdown_report'):
            pdf_filename = "fcgr_analysis_report.pdf"
            pdf_filepath = os.path.join(output_dir, pdf_filename)
            md_file_to_convert = os.path.join(output_dir, results_summary['report_paths']['markdown_report'])
            
            if os.path.exists(md_file_to_convert):
                pdf_success = rp.generate_pdf_report(md_file_to_convert, pdf_filepath)
                if pdf_success and os.path.exists(pdf_filepath):
                    results_summary['report_paths']['pdf_report'] = os.path.basename(pdf_filepath)
            else:
                logging.warning("Skipping PDF: Source markdown report file not found.")
        
        monitor.end_stage()
        results_summary['timings']['reporting'] = monitor.metrics['stage_times']['reporting']
        
        # Cleanup memory
        gc.collect()
        
    except Exception as e:
        logging.critical(f"Pipeline execution failed: {e}", exc_info=True)
        results_summary['error'] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    
    finally:
        # Final Summary Printout
        print("\n" + "="*30 + " Pipeline Execution Summary " + "="*30)
        print(f"Mode: {'Pyodide (Web)' if IS_PYODIDE else 'Native'}")
        total_time = results_summary['timings'].get('total_pipeline', time.time() - pipeline_start_time)
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Sequences Processed / FCGRs Generated: {results_summary['data_summary'].get('sequences_processed', 0)} / {results_summary['data_summary'].get('fcgrs_generated', 0)}")
        
        if results_summary.get('error'):
            print(f"*** ERROR: {results_summary['error'].splitlines()[0]} ***")
        
        if not is_web_mode:
            sp_acc = rp.get_nested(results_summary, ['ml_results', 'Species', 'accuracy'])
            bt_acc = rp.get_nested(results_summary, ['ml_results', 'Biotype', 'accuracy'])
            print(f"Species CNN Accuracy: {sp_acc:.4f}" if sp_acc is not None else "Species CNN: N/A")
            print(f"Biotype CNN Accuracy: {bt_acc:.4f}" if bt_acc is not None else "Biotype CNN: N/A")
            
            if output_dir:
                print(f"\nResults saved in: {os.path.abspath(output_dir)}")
                if results_summary.get('report_paths', {}).get('json_summary'):
                    print(f"  Summary JSON: {results_summary['report_paths']['json_summary']}")
                if results_summary.get('report_paths', {}).get('markdown_report'):
                    print(f"  Markdown Report: {results_summary['report_paths']['markdown_report']}")
                if results_summary.get('report_paths', {}).get('pdf_report'):
                    print(f"  PDF Report: {results_summary['report_paths']['pdf_report']}")
                if results_summary.get('report_paths', {}).get('log_file'):
                    print(f"  Log File: {results_summary['report_paths']['log_file']}")
        else:
            print("(ML Classification skipped in web mode)")
            print("(Results dictionary returned to JavaScript caller)")
        
        print("="*88)
        logging.info(f"--- FCGR Analysis Pipeline Finished ---")
    
    return results_summary
 
 
# --- Web Entry Point Wrapper ---
def run_web_pipeline(targets_string: str, config_overrides: dict):
   """Web entry point: Parses inputs, runs pipeline, captures logs, returns JSON."""
   log_stream = io.StringIO()
   log_format = '%(asctime)s [%(levelname)s] %(message)s'
   date_format = '%H:%M:%S'
   web_log_handler = logging.StreamHandler(log_stream)
   web_log_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
   root_logger = logging.getLogger()
   
   # Remove previous web handlers to avoid duplication
   for handler in root_logger.handlers[:]:
       if isinstance(handler, logging.StreamHandler) and handler.formatter and handler.formatter._fmt == log_format:
           root_logger.removeHandler(handler)
           handler.close()
   
   root_logger.addHandler(web_log_handler)
   root_logger.setLevel(logging.INFO)
   
   results_dict = {}
   json_output = "{}"
   
   try:
       logging.info("--- run_web_pipeline: Parsing web inputs ---")
       targets = []
       target_lines = [line.strip() for line in targets_string.splitlines() if line.strip() and not line.strip().startswith('#')]
       
       for i, line in enumerate(target_lines):
           try:
               parts = [p.strip() for p in line.split(',')]
               if len(parts) < 4:
                   parts = [p.strip() for p in line.split('\t')]
               if 4 <= len(parts) <= 5:
                   targets.append(tuple(parts))
               else:
                   logging.warning(f"Web: Skip target line {i+1}: {line}")
           except Exception as parse_e:
               logging.warning(f"Web: Error parsing line {i+1}: {parse_e}")
       
       if not targets:
           raise ValueError("No valid targets found in web input.")
       
       logging.info(f"Parsed {len(targets)} targets from web input.")
       
       # Prepare config
       effective_config = {k: v for k, v in vars(cfg).items() if not k.startswith('__')}
       if isinstance(config_overrides, dict):
           effective_config.update(config_overrides)
       
       # Force web settings
       effective_config['N_JOBS'] = 1
       effective_config['PANDOC_CHECK_ENABLED'] = False
       effective_config['PLOTTING_ENABLED'] = True
       effective_config['DEFAULT_CACHE_DIR'] = None
       effective_config['TF_CPP_MIN_LOG_LEVEL'] = '3'
       effective_config['targets_source'] = 'Web Input'
       
       # Call the main pipeline function
       results_dict = run_pipeline(
           output_dir=None,
           cache_dir=None,
           targets=targets,
           config_dict=effective_config
       )
       
   except Exception as e:
       logging.error(f"run_web_pipeline failed: {e}", exc_info=True)
       error_info = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
       results_dict = {'error': error_info}
   
   finally:
       # Capture logs
       results_dict['logs'] = log_stream.getvalue()
       try:
           root_logger.removeHandler(web_log_handler)
           web_log_handler.close()
       except Exception:
           pass
   
   # Convert to JSON
   try:
       json_output = json.dumps(results_dict, default=utils.convert_numpy_for_json, allow_nan=False)
   except Exception as json_e:
       logging.error(f"Failed to serialize results to JSON: {json_e}")
       json_output = json.dumps({
           'error': f"JSON Serialization Failed: {type(json_e).__name__}: {json_e}",
           'logs': results_dict.get('logs', '')
       })
   
   return json_output