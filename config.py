# -*- coding: utf-8 -*-
"""
Configuration Settings for FCGR Analyzer.
"""
import os
import multiprocessing

# --- Critical Settings ---
# _MUST_ be changed by the user
ENTREZ_EMAIL = "your.email@uni.com"
# Optional: Add API key for higher rate limits (10 req/sec)
ENTREZ_API_KEY = None

# --- Directory Setup ---
# Default output location, can be overridden by CLI
DEFAULT_OUTPUT_DIR = "fcgr_analysis_results"
# Subdirectories (relative to output_dir)
FIGURES_SUBDIR = "figures"
DATA_SUBDIR = "data"
# Default cache location
DEFAULT_CACHE_DIR = "./genomic_cache"

# --- Sequence Processing Parameters ---
FCGR_K = 6                      # k-mer size for FCGR
MIN_SEQ_LEN = 200               # Minimum sequence length after cleaning
MAX_SEQ_LEN = 5000              # Maximum sequence length (longer sequences are trimmed)
# Default sequence targets if no file is provided via CLI
# Format: (species, biotype, identifier_type, identifier, label_override [optional])
DEFAULT_SEQUENCE_TARGETS = [
    ('Human', 'Protein-coding', 'accession', 'NM_007294.4'), # BRCA1
    ('Human', 'rRNA', 'accession', 'NR_145820.1'),          # 18S rRNA
    ('Mouse', 'Protein-coding', 'accession', 'NM_007393.5'), # Actb
    ('Bacteria', 'rRNA', 'accession', 'NC_000913.3', '16S-rRNA'), # E. coli Genome
    ('Synth', 'Control', 'local_sequence', 'ATGC' * (MIN_SEQ_LEN // 4) + 'AAAA'),
    ('Synth', 'PolyA', 'local_sequence', 'A' * MAX_SEQ_LEN, 'Synthetic-PolyA'),
]

# --- NCBI Fetching ---
NCBI_FETCH_DELAY = 0.4          # Delay between NCBI fetches (seconds)
NCBI_DB = 'nuccore'             # NCBI database for sequences
NCBI_RETTYPE_FASTA = 'fasta'
NCBI_RETTYPE_GB = 'gb'
NCBI_RETMODE = 'text'
REQUEST_RETRIES = 5
REQUEST_BACKOFF_FACTOR = 0.5
REQUEST_STATUS_FORCELIST = (500, 502, 503, 504)

# --- Machine Learning Parameters ---
TEST_SIZE = 0.25                # Proportion of data for testing
RANDOM_STATE = 42               # Seed for reproducibility
CNN_EPOCHS = 20                 # Max epochs for CNN training
CNN_BATCH_SIZE = 16             # Batch size for CNN training
CNN_VALIDATION_SPLIT = 0.2      # Validation split from training data for EarlyStopping
CNN_EARLY_STOPPING_PATIENCE = 10 # Patience for EarlyStopping

# --- Performance & Resources ---
# Use half the cores by default, ensure at least 1
N_JOBS = max(1, multiprocessing.cpu_count() // 2)
# TensorFlow logging level (0=all, 1=info, 2=warnings, 3=errors)
TF_CPP_MIN_LOG_LEVEL = '2'
# Enable mixed precision on compatible GPUs (CC >= 7.0) automatically?
ENABLE_MIXED_PRECISION_IF_SUPPORTED = True

# --- Plotting ---
# Enable/disable plotting globally (can be useful in non-GUI environments)
PLOTTING_ENABLED = True
# Plotting style settings (if PLOTTING_ENABLED is True)
PLOT_STYLE = "whitegrid"
PLOT_PALETTE = "deep"
PLOT_FIG_WIDTH = 10
PLOT_FIG_HEIGHT = 7
PLOT_DPI = 100
PLOT_SAVEFIG_DPI = 300
FCGR_PLOT_EXAMPLES = 10          # Number of example FCGRs to plot

# --- Reporting ---
PANDOC_CHECK_ENABLED = True     # Check for pandoc and attempt PDF generation

# --- Internal Constants ---
# Do not modify these unless you understand the implications
FCGR_DIM = 1 << FCGR_K          # Dimension of the FCGR matrix (2^k)
VALID_BASES_BYTES = b"ATGC"     # Valid DNA bases for cleaning
EPSILON = 1e-9                  # Small value to avoid division by zero

# --- Derived Paths (calculated based on output_dir in pipeline/cli) ---
# These will be set dynamically based on the final output directory
FIGURES_DIR = None
DATA_DIR = None