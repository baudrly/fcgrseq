#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Command-Line Interface (CLI) for the FCGR Analyzer pipeline.
Supports genome sampler FASTA input and performance optimizations.
"""
import os
import sys
import argparse
import logging
import time
import json
import pkg_resources
from pathlib import Path
import multiprocessing

# Import pipeline components and configuration defaults
from . import pipeline
from . import config as default_config
from . import utils

def load_targets_from_file(filepath: str) -> list:
    """Loads sequence targets from a CSV or TSV file."""
    targets = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Detect delimiter (comma or tab)
                if '\t' in line:
                    parts = line.split('\t')
                else:
                    parts = line.split(',')

                if len(parts) < 4 or len(parts) > 5:
                    logging.warning(f"Skipping invalid line {i+1} in targets file '{filepath}': Expected 4 or 5 columns, got {len(parts)}. Line: '{line}'")
                    continue
                # Strip whitespace from each part
                parts = [p.strip() for p in parts]
                targets.append(tuple(parts))
        if not targets:
            logging.warning(f"No valid targets found in file: {filepath}")
            return None
        logging.info(f"Loaded {len(targets)} targets from file: {filepath}")
        return targets
    except FileNotFoundError:
        logging.error(f"Targets file not found: {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error reading targets file '{filepath}': {e}", exc_info=True)
        return None

def load_config_from_file(filepath: str) -> dict:
    """Loads configuration overrides from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
        logging.info(f"Loaded configuration overrides from: {filepath}")
        return user_config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {filepath}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON configuration file '{filepath}': {e}")
        return {}
    except Exception as e:
         logging.error(f"Error reading configuration file '{filepath}': {e}", exc_info=True)
         return {}

def run_tests():
    """Discovers and runs unit tests using pytest."""
    try:
        import pytest
    except ImportError:
        print("ERROR: pytest is required to run tests. Install with 'pip install pytest'", file=sys.stderr)
        return 1

    # Construct the path to the tests directory
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        test_dir = os.path.join(base_dir, 'tests')
        if not os.path.isdir(test_dir):
             try:
                  test_dir = pkg_resources.resource_filename('fcgr_analyzer', 'tests')
             except Exception:
                  print(f"ERROR: Cannot find the 'tests' directory relative to {__file__} or via package resources.", file=sys.stderr)
                  return 1
        print(f"Running tests from: {test_dir}")
        exit_code = pytest.main(['-v', test_dir])
        return exit_code

    except Exception as e:
        print(f"An error occurred while trying to run tests: {e}", file=sys.stderr)
        return 1


def validate_genome_sampler_fasta(filepath: str) -> bool:
    """Validate that the file is a valid genome sampler FASTA."""
    try:
        with open(filepath, 'r') as f:
            line = f.readline().strip()
            # Check if first line is a valid FASTA header in genome sampler format
            if line.startswith('>') and '|' in line:
                parts = line[1:].split('|')
                if len(parts) >= 3:
                    return True
        return False
    except Exception:
        return False


def main():
    """Main function to parse arguments and run the pipeline or tests."""

    parser = argparse.ArgumentParser(
        description="Enhanced Genome/Biotype Classification using Frequency Chaos Game Representation (FCGR).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Input/Output Arguments ---
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "-t", "--targets-file", type=str, default=None,
        help="Path to a CSV or TSV file containing sequence targets. "
             "Columns: species, biotype, identifier_type, identifier, [label_override]."
    )
    input_group.add_argument(
        "-i", "--input-fasta", type=str, default=None,
        help="Path to a genome sampler FASTA file (format: >species|biotype|description)."
    )
    
    parser.add_argument(
        "-o", "--output-dir", type=str, default=default_config.DEFAULT_OUTPUT_DIR,
        help="Directory to save all results, figures, and reports."
    )
    parser.add_argument(
        "-c", "--cache-dir", type=str, default=default_config.DEFAULT_CACHE_DIR,
        help="Directory for caching downloaded data and intermediate results (joblib)."
    )
    parser.add_argument(
        "--config-file", type=str, default=None,
        help="Path to a JSON file with configuration overrides (e.g., ENTREZ_EMAIL, FCGR_K)."
    )

    # --- Execution Control Arguments ---
    parser.add_argument(
        "--run-tests", action="store_true",
        help="Run unit tests using pytest instead of the main pipeline."
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level."
    )
    parser.add_argument(
        "-j", "--n-jobs", type=int, 
        default=max(1, multiprocessing.cpu_count() // 2),
        help="Number of parallel jobs to use for computation."
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Disable all plot generation."
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable filesystem caching (joblib)."
    )
    parser.add_argument(
        "--no-pdf", action="store_true",
        help="Disable PDF report generation attempt (even if pandoc is found)."
    )
    
    # --- Advanced Options ---
    advanced_group = parser.add_argument_group('Advanced Options')
    advanced_group.add_argument(
        "--fcgr-k", type=int, default=None,
        help="Override k-mer size for FCGR (default from config)."
    )
    advanced_group.add_argument(
        "--min-seq-len", type=int, default=None,
        help="Override minimum sequence length (default from config)."
    )
    advanced_group.add_argument(
        "--max-seq-len", type=int, default=None,
        help="Override maximum sequence length (default from config)."
    )
    advanced_group.add_argument(
        "--batch-size", type=int, default=1000,
        help="Batch size for processing large FASTA files."
    )

    # --- Parse Arguments ---
    args = parser.parse_args()

    # --- Setup Logging ---
    log_level_numeric = getattr(logging, args.log_level.upper(), logging.INFO)
    utils.setup_logging(log_level=log_level_numeric)

    # --- Run Tests if Requested ---
    if args.run_tests:
        print("Executing test suite...")
        test_exit_code = run_tests()
        print(f"Test suite finished with exit code: {test_exit_code}")
        sys.exit(test_exit_code)

    # --- Validate Input ---
    if not args.targets_file and not args.input_fasta:
        # Use default targets if no input specified
        targets = default_config.DEFAULT_SEQUENCE_TARGETS
        logging.info("No input file provided, using default targets list.")
        input_fasta = None
    elif args.input_fasta:
        # Validate genome sampler FASTA
        if not os.path.exists(args.input_fasta):
            print(f"ERROR: Input FASTA file not found: {args.input_fasta}", file=sys.stderr)
            sys.exit(1)
        if not validate_genome_sampler_fasta(args.input_fasta):
            print(f"ERROR: Input file does not appear to be a valid genome sampler FASTA: {args.input_fasta}", file=sys.stderr)
            print("Expected format: >species|biotype|description", file=sys.stderr)
            sys.exit(1)
        input_fasta = args.input_fasta
        targets = None  # Will be loaded from FASTA
        logging.info(f"Using genome sampler FASTA input: {input_fasta}")
    else:
        # Load targets from file
        targets = load_targets_from_file(args.targets_file)
        if targets is None:
            print(f"ERROR: Failed to load targets from specified file: {args.targets_file}", file=sys.stderr)
            sys.exit(1)
        input_fasta = None

    # --- Prepare Configuration ---
    # Start with defaults
    effective_config = {k: v for k, v in vars(default_config).items() if not k.startswith('__')}

    # Load overrides from JSON file if provided
    if args.config_file:
        user_config_overrides = load_config_from_file(args.config_file)
        # Validate and merge overrides
        valid_overrides = {}
        for key, value in user_config_overrides.items():
             if key in effective_config:
                 valid_overrides[key] = value
                 logging.debug(f"Applying config override: {key} = {value}")
             else:
                 logging.warning(f"Ignoring unknown key '{key}' found in config file '{args.config_file}'.")
        effective_config.update(valid_overrides)

    # Apply command-line overrides (which take precedence over file overrides)
    effective_config['N_JOBS'] = args.n_jobs
    if args.no_plots:
        effective_config['PLOTTING_ENABLED'] = False
    if args.no_pdf:
        effective_config['PANDOC_CHECK_ENABLED'] = False
    
    # Apply advanced options if provided
    if args.fcgr_k is not None:
        effective_config['FCGR_K'] = args.fcgr_k
        effective_config['FCGR_DIM'] = 1 << args.fcgr_k  # Update dimension
        logging.info(f"Overriding FCGR k-mer size to: {args.fcgr_k}")
    if args.min_seq_len is not None:
        effective_config['MIN_SEQ_LEN'] = args.min_seq_len
        logging.info(f"Overriding minimum sequence length to: {args.min_seq_len}")
    if args.max_seq_len is not None:
        effective_config['MAX_SEQ_LEN'] = args.max_seq_len
        logging.info(f"Overriding maximum sequence length to: {args.max_seq_len}")
    
    # Handle cache directory override
    cache_dir = args.cache_dir if not args.no_cache else None
    effective_config['DEFAULT_CACHE_DIR'] = cache_dir

    # Set output directory and derived paths
    output_dir = args.output_dir
    figures_dir = os.path.join(output_dir, default_config.FIGURES_SUBDIR)
    data_dir = os.path.join(output_dir, default_config.DATA_SUBDIR)
    effective_config['output_dir'] = output_dir
    effective_config['figures_dir'] = figures_dir
    effective_config['data_dir'] = data_dir

    # Set TensorFlow log level environment variable
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = effective_config.get('TF_CPP_MIN_LOG_LEVEL', '2')

    # Setup GPU and Mixed Precision based on config
    gpu_available, mixed_precision_enabled = utils.setup_gpu_and_mixed_precision(
        effective_config.get('ENABLE_MIXED_PRECISION_IF_SUPPORTED', True)
    )
    effective_config['gpu_available'] = gpu_available
    effective_config['mixed_precision'] = mixed_precision_enabled

    # Record input source
    if input_fasta:
        effective_config['targets_source'] = os.path.abspath(input_fasta)
    elif args.targets_file:
        effective_config['targets_source'] = os.path.abspath(args.targets_file)
    else:
        effective_config['targets_source'] = "Default List"

    # --- Final Configuration Validation ---
    if not input_fasta:  # Only check Entrez email if we might need to fetch sequences
        needs_entrez = False
        if targets:
            needs_entrez = any(t[2] == 'accession' for t in targets if len(t) >= 3)
        
        if needs_entrez:
            if not effective_config.get('ENTREZ_EMAIL') or effective_config.get('ENTREZ_EMAIL') == "your.email@example.com":
                print("CRITICAL ERROR: ENTREZ_EMAIL is not configured.", file=sys.stderr)
                print("Please set it in the config file (--config-file), ", file=sys.stderr)
                print("or modify it directly in fcgr_analyzer/config.py if running from source.", file=sys.stderr)
                sys.exit(1)

    # --- Update Global Config State ---
    for key, value in effective_config.items():
         if hasattr(default_config, key):
             setattr(default_config, key, value)

    # --- Print Configuration Summary ---
    print("\n" + "="*50)
    print("FCGR Analyzer - Configuration Summary")
    print("="*50)
    print(f"Input Source: {effective_config['targets_source']}")
    print(f"Output Directory: {output_dir}")
    print(f"Cache Directory: {cache_dir or 'Disabled'}")
    print(f"FCGR k-mer size: {effective_config['FCGR_K']}")
    print(f"Sequence length range: {effective_config['MIN_SEQ_LEN']}-{effective_config['MAX_SEQ_LEN']} bp")
    print(f"Parallel jobs: {effective_config['N_JOBS']}")
    print(f"GPU available: {gpu_available}")
    if gpu_available:
        print(f"Mixed precision: {mixed_precision_enabled}")
    print(f"Plotting enabled: {effective_config['PLOTTING_ENABLED']}")
    print("="*50 + "\n")

    # --- Run the Main Pipeline ---
    start_time = time.time()
    try:
        results = pipeline.run_pipeline(
            output_dir=output_dir,
            cache_dir=cache_dir,
            targets=targets,
            config_dict=effective_config,
            input_fasta=input_fasta  # Pass the FASTA file if provided
        )

        # Print performance metrics
        if 'performance_metrics' in results:
            print("\n" + "="*50)
            print("Performance Metrics")
            print("="*50)
            
            perf = results['performance_metrics']
            if 'stage_times' in perf:
                print("\nStage Execution Times:")
                for stage, duration in perf['stage_times'].items():
                    print(f"  {stage}: {duration:.2f}s")
            
            if 'processing_rates' in perf:
                print("\nProcessing Rates:")
                for stage, rate in perf['processing_rates'].items():
                    print(f"  {stage}: {rate:.1f} items/s")
            
            if 'memory_usage' in perf:
                print("\nMemory Usage (MB):")
                for checkpoint, usage in perf['memory_usage'].items():
                    print(f"  {checkpoint}: {usage:.1f} MB")
            
            print("="*50 + "\n")

    except Exception as e:
        logging.critical(f"Pipeline execution failed with an unhandled exception: {type(e).__name__}: {e}", exc_info=True)
        print(f"\nERROR: Pipeline failed unexpectedly. Check log for details.", file=sys.stderr)
        sys.exit(1)
    finally:
        end_time = time.time()
        logging.info(f"Total CLI execution time: {end_time - start_time:.2f} seconds.")

    # --- Exit ---
    sys.exit(0)


if __name__ == "__main__":
    main()