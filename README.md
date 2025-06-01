# fcgrseq

## Overview
fcgrseq is a comprehensive bioinformatics tool for analyzing genomic sequences using Frequency Chaos Game Representation (FCGR). This pipeline performs sequence processing, feature extraction, statistical analysis, and machine learning classification to study sequence characteristics across different species and biotypes by way of their image fingerprint. A basic CNN is also run on the images themselves for species/biotype classification.

A live demo of what FCGR images look like and how they are fingerprinted is available [here](https://baudrly.github.io/fcgrseq).

## Key Features

- **FCGR Generation**: Efficient computation of FCGR matrices from DNA sequences
- **Feature Extraction**: 
  - Statistical features (mean, variance, skewness, etc.)
  - Haralick texture features
  - Hu moments
  - Fractal dimension
- **Advanced Analysis**:
  - Statistical comparisons between groups
  - Dimensionality reduction (PCA, t-SNE, UMAP)
  - Correlation analysis
- **Machine Learning**:
  - CNN-based classification
  - Model evaluation and visualization
- **Visualization**:
  - Examples of FCGR heatmaps
  - Feature distributions
  - Various metrics
  - Learning curves

## Installation

### Prerequisites
- Python 3.8+
- Required packages (see `requirements.txt`):

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd fcgrseq
    ```

2.  **Install dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    Requirements include

    ```
    numpy
    scipy
    pandas
    matplotlib
    seaborn
    scikit-learn
    scikit-image
    biopython
    requests
    tensorflow (optional for ML)
    umap-learn (optional for UMAP)
    ```

    *Note: `requirements.txt` lists necessary packages including `tensorflow`. You might want `tensorflow-cpu` if you don't have a compatible GPU.*

4.  **(Optional) Install Pandoc and LaTeX:** For PDF report generation, you need to install Pandoc ([https://pandoc.org/installing.html](https://pandoc.org/installing.html)) and a LaTeX distribution (e.g., TeX Live, MiKTeX). Ensure `pandoc` is in your system's PATH.

## Configuration

Key configuration options are in `config.py`. Modify config.py or provide a JSON config file to adjust:

* Sequence processing parameters

* FCGR k-mer size

* Machine learning settings

* Plotting preferences

* NCBI Entrez credentials (if fetching data from online databases)

## Output

The pipeline generates:

* Processed sequence data

* FCGR matrices

* Feature tables

* Statistical analysis results

* Visualization plots

* Machine learning reports


## Usage

```bash
python -m fcgr_analyzer.cli [-h] [-t TARGETS_FILE] [-i INPUT_FASTA] [-o OUTPUT_DIR] 
                           [--config-file CONFIG_FILE] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
```

### Main arguments

```
-t/--targets-file: CSV/TSV file with sequence targets

-i/--input-fasta: Genome sampler FASTA file

-o/--output-dir: Output directory for results

--config-file: JSON configuration file

--log-level: Set logging verbosity

```

### Example

```bash
python -m fcgr_analyzer.cli -i input.fasta -o results --log-level INFO
```
