# -*- coding: utf-8 -*-
"""
Enhanced Analysis and Visualization of FCGR Data and Features.
Includes improved FCGR plotting, advanced statistical analysis, and publication-quality visualizations.
"""
import os
import logging
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency, f_oneway, normaltest
from scipy.cluster.hierarchy import dendrogram, linkage
from statsmodels.stats.multitest import multipletests
import io
import base64
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

# Import framework components and check flags
from .config import PLOT_STYLE, PLOT_PALETTE, PLOT_FIG_WIDTH, PLOT_FIG_HEIGHT, \
                    PLOT_DPI, PLOT_SAVEFIG_DPI, EPSILON, RANDOM_STATE, PLOTTING_ENABLED
from .utils import IS_PYODIDE, safe_filename # Import flags/helpers

# --- Conditional Dependency Imports and Availability Flags ---

PLOTTING_LIBS_AVAILABLE = False
plt = None
sns = None
if PLOTTING_ENABLED:
    try:
        import matplotlib
        import sys
        is_headless = not os.environ.get('DISPLAY', None) and not ('ipykernel' in sys.modules)
        if IS_PYODIDE or is_headless:
             try: matplotlib.use('Agg')
             except Exception: logging.warning("Could not set Matplotlib backend to Agg.")
        import matplotlib.pyplot as plt_
        plt = plt_
        import seaborn as sns_
        sns = sns_
        # Enhanced styling for publication-quality plots
        sns.set_theme(style=PLOT_STYLE, palette=PLOT_PALETTE)
        plt.rcParams['figure.figsize'] = (PLOT_FIG_WIDTH, PLOT_FIG_HEIGHT)
        plt.rcParams['figure.dpi'] = PLOT_DPI
        plt.rcParams['savefig.dpi'] = PLOT_SAVEFIG_DPI
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 16
        # Set high-quality font
        try:
            plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        except:
            pass
        PLOTTING_LIBS_AVAILABLE = True
        logging.info("Plotting enabled and libraries (Matplotlib/Seaborn) loaded with enhanced settings.")
    except ImportError:
        logging.warning("Matplotlib/Seaborn not found, disabling plotting even though PLOTTING_ENABLED=True.")
    except Exception as e:
        logging.warning(f"Error during Matplotlib/Seaborn setup: {e}. Plotting disabled.", exc_info=False)

SKLEARN_AVAILABLE = False
StandardScaler = None; PCA = None; TSNE = None; UMAP = None
try:
    from sklearn.preprocessing import StandardScaler as sk_StandardScaler
    StandardScaler = sk_StandardScaler
    from sklearn.decomposition import PCA as sk_PCA
    PCA = sk_PCA
    from sklearn.manifold import TSNE as sk_TSNE
    TSNE = sk_TSNE
    try:
        from umap import UMAP as umap_UMAP
        UMAP = umap_UMAP
        logging.debug("UMAP available for dimensionality reduction.")
    except ImportError:
        logging.debug("UMAP not found. Install with 'pip install umap-learn' for additional DR options.")
    SKLEARN_AVAILABLE = True
    logging.debug("scikit-learn found.")
except ImportError:
     logging.debug("scikit-learn not found. Dimensionality reduction and scaling disabled.")

# Import N_JOBS dynamically based on environment
N_JOBS = 1 # Default for Pyodide or if config fails
if not IS_PYODIDE:
    try: from .config import N_JOBS
    except ImportError: logging.warning("Could not import N_JOBS from config, defaulting to 1.")


# --- Enhanced Color Palettes ---
ENHANCED_PALETTES = {
    'categorical': ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', 
                    '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#d35400'],
    'diverging': 'RdBu_r',
    'sequential': 'viridis',
    'heatmap': 'coolwarm'
}

# --- Plot Saving Helpers ---

def _save_plot_to_file(fig, save_path: str, dpi=300) -> bool:
    """Saves a matplotlib figure to a file."""
    if not fig or not plt or not save_path: return False
    if IS_PYODIDE:
         logging.warning("File saving skipped in Pyodide environment.")
         plt.close(fig)
         return False
    try:
        output_dir = os.path.dirname(save_path)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=dpi, facecolor='white', edgecolor='none')
        logging.debug(f"Saved plot file: {save_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to save plot to file {save_path}: {e}", exc_info=False)
        return False
    finally:
         plt.close(fig) # Close figure after saving attempt

def _save_plot_to_base64(fig, dpi=100) -> Optional[str]:
    """Saves a matplotlib figure to a base64 encoded PNG string."""
    if not fig or not plt: return None
    current_backend = plt.get_backend()
    backend_changed = False
    # Switch only if necessary and possible
    if current_backend.lower() not in ('agg', 'macosx'): # Avoid switching if already Agg or potentially problematic Mac backend
        try: plt.switch_backend('Agg'); backend_changed = True
        except Exception as be: logging.warning(f"Could not switch backend to Agg: {be}")

    img_base64 = None
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi, facecolor='white', edgecolor='none')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
    except Exception as e:
        logging.error(f"Failed to save plot to base64: {e}", exc_info=False)
    finally:
        if backend_changed:
            try: plt.switch_backend(current_backend)
            except Exception: pass # Ignore error switching back
        plt.close(fig) # Always close the figure

    return img_base64


# --- Enhanced Plotting Functions ---

def plot_fcgr(matrix: np.ndarray, title: str, output_target: str,
              output_format: str = 'file', base_dpi: int = 100,
              show_grid: bool = True, annotate_regions: bool = True) -> Optional[str]:
    """Enhanced FCGR plot with annotations and better aesthetics."""
    if not PLOTTING_ENABLED or not PLOTTING_LIBS_AVAILABLE or not plt: return None
    if matrix is None or matrix.size == 0: return None

    fig = None
    try:
        fig, ax = plt.subplots(figsize=(10, 10))  # FIXED: Increased from (7, 7)
        
        # Use log scale with better color mapping
        plot_data = np.log1p(matrix * 1000)
        
        # Create custom colormap
        from matplotlib.colors import LinearSegmentedColormap
        colors = ['#ffffff', '#e8f4f8', '#a8d5e5', '#5b9bb5', '#2c5f7c', '#163a4a']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('fcgr_cmap', colors, N=n_bins)
        
        # Plot with enhanced styling
        img = ax.imshow(plot_data, cmap=cmap, origin='lower', interpolation='bilinear', aspect='equal')
        
        # Add title with better formatting
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)  # FIXED: Larger font
        ax.set_xlabel("k-mer suffix (G/T axis)", fontsize=14)
        ax.set_ylabel("k-mer prefix (C/A axis)", fontsize=14)
        
        # Add grid if requested
        if show_grid and matrix.shape[0] <= 64:
            for i in range(1, matrix.shape[0]):
                ax.axhline(y=i-0.5, color='gray', linewidth=0.5, alpha=0.3)
                ax.axvline(x=i-0.5, color='gray', linewidth=0.5, alpha=0.3)
        
        # Annotate quadrants if requested
        if annotate_regions and matrix.shape[0] >= 4:
            mid = matrix.shape[0] // 2
            regions = {
                'AA-rich': (mid//2, mid//2),
                'AT-rich': (3*mid//2, mid//2),
                'CC-rich': (mid//2, 3*mid//2),
                'GC-rich': (3*mid//2, 3*mid//2)
            }
            for label, (x, y) in regions.items():
                if 0 <= x < matrix.shape[1] and 0 <= y < matrix.shape[0]:
                    ax.text(x, y, label, ha='center', va='center', 
                           fontsize=12, fontweight='bold', color='white',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add colorbar with better styling
        cbar = plt.colorbar(img, ax=ax, label='Log(1 + 1000 × Frequency)', shrink=0.8)
        cbar.ax.yaxis.label.set_fontsize(12)
        
        plt.tight_layout()

        if output_format == 'base64':
            return _save_plot_to_base64(fig, dpi=150)  # FIXED: Higher DPI
        elif output_format == 'file':
            return output_target if _save_plot_to_file(fig, output_target, dpi=300) else None  # FIXED: Higher DPI
        else:
            logging.warning(f"Unsupported plot output format: {output_format}")
            plt.close(fig)
            return None
    except Exception as plot_e:
        logging.error(f"Error during FCGR plot generation for '{title}': {plot_e}", exc_info=False)
        if fig: plt.close(fig)
        return None


def run_feature_analysis(df: pd.DataFrame, feature_name: str, group_by_col: str,
                         figures_dir: Optional[str] = None,
                         output_format: str = 'file',
                         statistical_tests: List[str] = ['kruskal', 'mannwhitney']) -> Tuple[dict, Optional[str]]:
    """Enhanced feature analysis with multiple statistical tests and better visualization."""
    analysis_results = {
        'feature': feature_name, 
        'group_by': group_by_col, 
        'tests': {},
        'descriptive_stats': {},
        'effect_size': None
    }
    plot_output = None

# --- Input validation ---
    if feature_name not in df.columns or group_by_col not in df.columns: 
        return analysis_results, plot_output
    if df[feature_name].isnull().all(): 
        return analysis_results, plot_output
    unique_groups = df[group_by_col].nunique()
    if unique_groups < 2: 
        return analysis_results, plot_output
    
    # Check variance
    try: 
        feature_variance = df[feature_name].dropna().var()
    except Exception: 
        feature_variance = 0
    if feature_variance < EPSILON: 
        return analysis_results, plot_output
 
    logging.info(f"Analyzing feature '{feature_name}' by '{group_by_col}' (Format: {output_format})...")
 
    # --- Compute descriptive statistics ---
    grouped = df.groupby(group_by_col)[feature_name]
    desc_stats = grouped.describe().to_dict()
    analysis_results['descriptive_stats'] = desc_stats
 
    # --- Statistical Tests ---
    try:
        groups_data = [group[feature_name].dropna().values for name, group in df.groupby(group_by_col)]
        valid_groups_data = [g for g in groups_data if len(g) >= 3]
        
        if len(valid_groups_data) >= 2:
            # Kruskal-Wallis test
            if 'kruskal' in statistical_tests and len(valid_groups_data) > 2:
                h_stat, p_value = stats.kruskal(*valid_groups_data)
                analysis_results['tests']['kruskal_wallis'] = {
                    'statistic': float(h_stat) if np.isfinite(h_stat) else None,
                    'p_value': float(p_value) if np.isfinite(p_value) else None,
                    'significant': bool(p_value < 0.05) if np.isfinite(p_value) else None
                }
            
            # ANOVA (if data is approximately normal)
            if 'anova' in statistical_tests and len(valid_groups_data) > 2:
                # Check normality for each group
                normality_ok = True
                for g in valid_groups_data:
                    if len(g) >= 8:  # Need sufficient samples for normality test
                        _, norm_p = normaltest(g)
                        if norm_p < 0.05:
                            normality_ok = False
                            break
                
                if normality_ok:
                    f_stat, p_value = f_oneway(*valid_groups_data)
                    analysis_results['tests']['anova'] = {
                        'statistic': float(f_stat) if np.isfinite(f_stat) else None,
                        'p_value': float(p_value) if np.isfinite(p_value) else None,
                        'significant': bool(p_value < 0.05) if np.isfinite(p_value) else None
                    }
            
            # Pairwise comparisons if significant
            if len(valid_groups_data) == 2 or (
                analysis_results['tests'].get('kruskal_wallis', {}).get('significant', False)
            ):
                pairwise_results = []
                group_names = sorted(df[group_by_col].dropna().unique())
                
                for i in range(len(group_names)):
                    for j in range(i + 1, len(group_names)):
                        g1_data = df[df[group_by_col] == group_names[i]][feature_name].dropna()
                        g2_data = df[df[group_by_col] == group_names[j]][feature_name].dropna()
                        
                        if len(g1_data) >= 3 and len(g2_data) >= 3:
                            u_stat, p_val = mannwhitneyu(g1_data, g2_data, alternative='two-sided')
                            pairwise_results.append({
                                'group1': group_names[i],
                                'group2': group_names[j],
                                'statistic': float(u_stat),
                                'p_value': float(p_val),
                                'effect_size': float((u_stat / (len(g1_data) * len(g2_data))) * 2 - 1)  # Rank-biserial correlation
                            })
                
                # Apply multiple testing correction
                if pairwise_results:
                    p_values = [r['p_value'] for r in pairwise_results]
                    corrected = multipletests(p_values, method='bonferroni')
                    for i, result in enumerate(pairwise_results):
                        result['p_value_corrected'] = corrected[1][i]
                        result['significant_corrected'] = corrected[0][i]
                    
                    analysis_results['tests']['pairwise_comparisons'] = pairwise_results
            
            # Calculate effect size (eta-squared for Kruskal-Wallis)
            if 'kruskal_wallis' in analysis_results['tests']:
                h = analysis_results['tests']['kruskal_wallis']['statistic']
                n = len(df[feature_name].dropna())
                k = len(valid_groups_data)
                if h and n > k:
                    eta_squared = (h - k + 1) / (n - k)
                    analysis_results['effect_size'] = float(eta_squared) if eta_squared >= 0 else 0.0
                    
    except Exception as e:
        logging.warning(f"Statistical tests failed for '{feature_name}': {e}")
 
    # --- Enhanced Plotting ---
    fig = None; save_path = None; plot_filename = None
    if PLOTTING_ENABLED and PLOTTING_LIBS_AVAILABLE and plt and sns:
        if output_format == 'file' and figures_dir:
            plot_filename_base = safe_filename(f"dist_{feature_name}_by_{group_by_col}")
            plot_filename = f"{plot_filename_base}.png"
            save_path = os.path.join(figures_dir, plot_filename)
 
        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Get sorted groups
            order = sorted(df[group_by_col].dropna().unique())
            n_groups = len(order)
            
            # Create appropriate palette
            if n_groups <= 10:
                palette = ENHANCED_PALETTES['categorical'][:n_groups]
            else:
                palette = sns.color_palette('husl', n_colors=n_groups)
            
            # Subplot 1: Enhanced box/violin plot
            avg_group_size = df.groupby(group_by_col).size().mean()
            
            if avg_group_size < 30:
                # Use violin plot for smaller samples
                sns.violinplot(data=df, x=group_by_col, y=feature_name, order=order, inner='box', palette=palette, ax=ax1)
            else:
                # Use box plot with points for larger samples
                sns.boxplot(data=df, x=group_by_col, y=feature_name, order=order, palette=palette, showfliers=False, ax=ax1)
                
                # Add strip plot for individual points if not too many
                if len(df) < 1000:
                    sns.stripplot(data=df, x=group_by_col, y=feature_name, order=order, color=".3", alpha=0.5, size=3, jitter=True, ax=ax1)
            
            ax1.set_title(f'Distribution of {feature_name} by {group_by_col}', fontsize=12, fontweight='bold')
            ax1.set_xlabel(group_by_col.replace('_', ' ').capitalize(), fontsize=11)
            ax1.set_ylabel(feature_name.replace('_', ' ').capitalize(), fontsize=11)
            
            # Rotate x-axis labels if needed
            if n_groups > 10 or any(len(str(g)) > 10 for g in order):
                ax1.tick_params(axis='x', rotation=45, labelsize=8)
            else:
                ax1.tick_params(axis='x', rotation=30, labelsize=9)
            
            # Add significance annotations if available
            if 'kruskal_wallis' in analysis_results['tests']:
                p_val = analysis_results['tests']['kruskal_wallis']['p_value']
                if p_val is not None:
                    sig_text = f"Kruskal-Wallis p = {p_val:.3e}" if p_val < 0.001 else f"Kruskal-Wallis p = {p_val:.3f}"
                    if p_val < 0.05:
                        sig_text += " *"
                    if p_val < 0.01:
                        sig_text += "*"
                    if p_val < 0.001:
                        sig_text += "*"
                    ax1.text(0.02, 0.98, sig_text, transform=ax1.transAxes, 
                            verticalalignment='top', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Subplot 2: Statistical summary plot
            # Create bar plot with error bars
            summary_data = df.groupby(group_by_col)[feature_name].agg(['mean', 'std', 'count']).reset_index()
            summary_data['sem'] = summary_data['std'] / np.sqrt(summary_data['count'])
            
            bars = ax2.bar(range(len(order)), summary_data.set_index(group_by_col).loc[order, 'mean'],
                           yerr=summary_data.set_index(group_by_col).loc[order, 'sem'],
                           capsize=5, color=palette,
                           edgecolor='black', linewidth=1)
            
            ax2.set_xticks(range(len(order)))
            ax2.set_xticklabels(order, rotation=45 if n_groups > 10 else 30, ha='right')
            ax2.set_title(f'Mean ± SEM of {feature_name}', fontsize=12, fontweight='bold')
            ax2.set_xlabel(group_by_col.replace('_', ' ').capitalize(), fontsize=11)
            ax2.set_ylabel(f'Mean {feature_name.replace("_", " ").capitalize()}', fontsize=11)
            
            # Add sample sizes on bars
            for i, (idx, row) in enumerate(summary_data.set_index(group_by_col).loc[order].iterrows()):
                ax2.text(i, row['mean'] + row['sem'] + ax2.get_ylim()[1]*0.01, 
                        f"n={int(row['count'])}", ha='center', va='bottom', fontsize=9)
            
            # Add grid
            ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
            ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
            
            # Overall title
            fig.suptitle(f'Feature Analysis: {feature_name} by {group_by_col}', 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
 
            if output_format == 'base64': 
                plot_output = _save_plot_to_base64(fig, dpi=90)
            elif output_format == 'file' and save_path:
                success = _save_plot_to_file(fig, save_path, dpi=PLOT_SAVEFIG_DPI)
                if success and figures_dir: 
                    plot_output = os.path.join(os.path.basename(figures_dir), plot_filename)
            else: 
                plt.close(fig) # Close if not saved
            fig = None # Mark as handled
 
        except Exception as e:
            logging.error(f"Failed plot distribution for '{feature_name}': {e}", exc_info=False)
            if fig: plt.close(fig)
            plot_output = None
 
    # Add plot path to results only if native file format
    if output_format == 'file' and plot_output:
        analysis_results['plot_path'] = plot_output
 
    return analysis_results, plot_output


def plot_sequence_length_distribution(df: pd.DataFrame, figures_dir: Optional[str] = None,
                                    output_format: str = 'file') -> Optional[str]:
    """Plot sequence length distribution by species and biotype."""
    plot_output = None
    if not PLOTTING_ENABLED or not PLOTTING_LIBS_AVAILABLE or not plt or not sns:
        return plot_output
    
    fig = None
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Overall length distribution
        ax1.hist(df['length'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Sequence Length (bp)', fontsize=11)
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title('Overall Sequence Length Distribution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Add statistics
        mean_len = df['length'].mean()
        median_len = df['length'].median()
        ax1.axvline(mean_len, color='red', linestyle='--', label=f'Mean: {mean_len:.0f}')
        ax1.axvline(median_len, color='green', linestyle='--', label=f'Median: {median_len:.0f}')
        ax1.legend()
        
        # 2. Length by species
        species_order = df.groupby('species')['length'].median().sort_values().index
        n_species = len(species_order)
        species_palette = sns.color_palette(ENHANCED_PALETTES['categorical'], n_colors=n_species) if n_species <= 10 else sns.color_palette('husl', n_colors=n_species)
        sns.boxplot(data=df, y='species', x='length', order=species_order, 
                   palette=species_palette, ax=ax2)
        ax2.set_xlabel('Sequence Length (bp)', fontsize=11)
        ax2.set_ylabel('Species', fontsize=11)
        ax2.set_title('Sequence Length by Species', fontsize=12, fontweight='bold')
        ax2.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        # 3. Length by biotype
        biotype_order = df.groupby('biotype')['length'].median().sort_values().index
        n_biotypes = len(biotype_order)
        biotype_palette = sns.color_palette(ENHANCED_PALETTES['categorical'], n_colors=n_biotypes) if n_biotypes <= 10 else sns.color_palette('husl', n_colors=n_biotypes)
        sns.violinplot(data=df, x='biotype', y='length', order=biotype_order,
                      palette=biotype_palette, inner='box', ax=ax3)
        ax3.set_xlabel('Biotype', fontsize=11)
        ax3.set_ylabel('Sequence Length (bp)', fontsize=11)
        ax3.set_title('Sequence Length by Biotype', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        # 4. 2D hexbin: species vs biotype colored by mean length
        pivot_table = df.pivot_table(values='length', index='species', columns='biotype', aggfunc='mean')
        im = ax4.imshow(pivot_table.values, cmap='viridis', aspect='auto')
        ax4.set_xticks(range(len(pivot_table.columns)))
        ax4.set_yticks(range(len(pivot_table.index)))
        ax4.set_xticklabels(pivot_table.columns, rotation=45, ha='right')
        ax4.set_yticklabels(pivot_table.index)
        ax4.set_xlabel('Biotype', fontsize=11)
        ax4.set_ylabel('Species', fontsize=11)
        ax4.set_title('Mean Sequence Length Heatmap', fontsize=12, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Mean Length (bp)', fontsize=10)
        
        # Add text annotations
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                val = pivot_table.values[i, j]
                if not np.isnan(val):
                    text_color = 'white' if val < pivot_table.values.mean() else 'black'
                    ax4.text(j, i, f'{val:.0f}', ha='center', va='center', 
                           color=text_color, fontsize=8)
        
        plt.suptitle('Sequence Length Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if output_format == 'base64':
            plot_output = _save_plot_to_base64(fig, dpi=90)
        elif output_format == 'file' and figures_dir:
            save_path = os.path.join(figures_dir, 'sequence_length_analysis.png')
            success = _save_plot_to_file(fig, save_path, dpi=PLOT_SAVEFIG_DPI)
            if success:
                plot_output = os.path.join(os.path.basename(figures_dir), 'sequence_length_analysis.png')
        else:
            plt.close(fig)
            
    except Exception as e:
        logging.error(f"Failed to plot sequence length analysis: {e}", exc_info=False)
        if fig:
            plt.close(fig)
            
    return plot_output


def run_length_adjusted_analysis(df: pd.DataFrame, feature_name: str, 
                               figures_dir: Optional[str] = None,
                               output_format: str = 'file') -> Tuple[dict, Optional[str]]:
    """Run statistical analysis with sequence length adjustment."""
    results = {
        'feature': feature_name,
        'length_correlation': None,
        'residuals_stats': {},
        'length_adjusted_comparisons': {}
    }
    plot_output = None
    
    if feature_name not in df.columns or 'length' not in df.columns:
        return results, plot_output
    
    # Calculate correlation with length
    try:
        data = df[[feature_name, 'length', 'species', 'biotype']].dropna()
        if len(data) < 10:
            return results, plot_output
            
        # Correlation analysis
        corr_r, corr_p = stats.pearsonr(data[feature_name], data['length'])
        results['length_correlation'] = {
            'pearson_r': float(corr_r),
            'p_value': float(corr_p),
            'significant': corr_p < 0.05
        }
        
        # Calculate residuals after length adjustment
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        X = data[['length']].values
        y = data[feature_name].values
        lr.fit(X, y)
        y_pred = lr.predict(X)
        residuals = y - y_pred
        
        # Add residuals to data
        data['residuals'] = residuals
        
        # Run comparisons on residuals
        for group_col in ['species', 'biotype']:
            groups = []
            labels = []
            for name, group in data.groupby(group_col):
                if len(group) >= 3:
                    groups.append(group['residuals'].values)
                    labels.append(name)
            
            if len(groups) >= 2:
                # Kruskal-Wallis on residuals
                h_stat, p_val = kruskal(*groups)
                results['length_adjusted_comparisons'][group_col] = {
                    'kruskal_h': float(h_stat),
                    'p_value': float(p_val),
                    'n_groups': len(groups),
                    'groups': labels
                }
        
        # Plotting
        if PLOTTING_ENABLED and PLOTTING_LIBS_AVAILABLE and plt:
            fig = None
            try:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                
                # 1. Feature vs Length scatter
                scatter = ax1.scatter(data['length'], data[feature_name], 
                                    c=data['species'].astype('category').cat.codes,
                                    alpha=0.6, cmap='tab10')
                ax1.plot(X, y_pred, 'r--', linewidth=2, label=f'Linear fit (r={corr_r:.3f})')
                ax1.set_xlabel('Sequence Length (bp)', fontsize=11)
                ax1.set_ylabel(feature_name, fontsize=11)
                ax1.set_title(f'{feature_name} vs Sequence Length', fontsize=12, fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 2. Residuals distribution
                ax2.hist(residuals, bins=30, color='green', alpha=0.7, edgecolor='black')
                ax2.set_xlabel('Residuals (Length-adjusted)', fontsize=11)
                ax2.set_ylabel('Count', fontsize=11)
                ax2.set_title('Distribution of Length-adjusted Values', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                # 3. Residuals by species
                species_order = data.groupby('species')['residuals'].median().sort_values().index
                sns.boxplot(data=data, x='species', y='residuals', order=species_order,
                          palette=ENHANCED_PALETTES['categorical'], ax=ax3)
                ax3.set_xlabel('Species', fontsize=11)
                ax3.set_ylabel('Length-adjusted ' + feature_name, fontsize=11)
                ax3.set_title('Length-adjusted Values by Species', fontsize=12, fontweight='bold')
                ax3.tick_params(axis='x', rotation=45)
                ax3.grid(True, axis='y', alpha=0.3)
                
                # Add p-value if available
                if 'species' in results['length_adjusted_comparisons']:
                    p_val = results['length_adjusted_comparisons']['species']['p_value']
                    ax3.text(0.02, 0.98, f'Kruskal-Wallis p = {p_val:.3e}' if p_val < 0.001 else f'p = {p_val:.3f}',
                           transform=ax3.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # 4. Residuals by biotype
                biotype_order = data.groupby('biotype')['residuals'].median().sort_values().index
                sns.boxplot(data=data, x='biotype', y='residuals', order=biotype_order,
                          palette=ENHANCED_PALETTES['categorical'], ax=ax4)
                ax4.set_xlabel('Biotype', fontsize=11)
                ax4.set_ylabel('Length-adjusted ' + feature_name, fontsize=11)
                ax4.set_title('Length-adjusted Values by Biotype', fontsize=12, fontweight='bold')
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, axis='y', alpha=0.3)
                
                plt.suptitle(f'Length-adjusted Analysis: {feature_name}', fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                if output_format == 'base64':
                    plot_output = _save_plot_to_base64(fig, dpi=90)
                elif output_format == 'file' and figures_dir:
                    save_path = os.path.join(figures_dir, f'length_adjusted_{safe_filename(feature_name)}.png')
                    success = _save_plot_to_file(fig, save_path, dpi=PLOT_SAVEFIG_DPI)
                    if success:
                        plot_output = os.path.join(os.path.basename(figures_dir), 
                                                 f'length_adjusted_{safe_filename(feature_name)}.png')
                else:
                    plt.close(fig)
                    
            except Exception as e:
                logging.error(f"Failed to plot length-adjusted analysis: {e}", exc_info=False)
                if fig:
                    plt.close(fig)
                    
    except Exception as e:
        logging.error(f"Length-adjusted analysis failed for {feature_name}: {e}", exc_info=False)
    
    return results, plot_output


def plot_pairwise_comparisons_heatmap(stats_results: List[dict], 
                                    figures_dir: Optional[str] = None,
                                    output_format: str = 'file') -> Optional[str]:
    """Create comprehensive heatmaps for all pairwise comparisons."""
    plot_output = None
    if not PLOTTING_ENABLED or not PLOTTING_LIBS_AVAILABLE or not plt or not sns:
        return plot_output
    
    # Collect all pairwise comparisons
    comparisons_by_feature = {}
    
    for result in stats_results:
        if 'tests' in result and 'pairwise_comparisons' in result['tests']:
            feature = result['feature']
            group_by = result['group_by']
            key = f"{feature}_{group_by}"
            comparisons_by_feature[key] = {
                'feature': feature,
                'group_by': group_by,
                'comparisons': result['tests']['pairwise_comparisons']
            }
    
    if not comparisons_by_feature:
        return plot_output
    
    fig = None
    try:
        # Create subplots for each feature/grouping combination
        n_plots = len(comparisons_by_feature)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(7 * n_cols, 6 * n_rows))
        
        for idx, (key, data) in enumerate(comparisons_by_feature.items()):
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            
            # Extract groups and create matrix
            comparisons = data['comparisons']
            if not comparisons:
                continue
                
            # Get unique groups
            groups = set()
            for comp in comparisons:
                groups.add(comp['group1'])
                groups.add(comp['group2'])
            groups = sorted(groups)
            
            # Create p-value matrix
            n_groups = len(groups)
            p_matrix = np.ones((n_groups, n_groups))
            effect_matrix = np.zeros((n_groups, n_groups))
            
            group_to_idx = {g: i for i, g in enumerate(groups)}
            
            for comp in comparisons:
                i = group_to_idx[comp['group1']]
                j = group_to_idx[comp['group2']]
                p_val = comp['p_value_corrected']
                effect = comp['effect_size']
                
                p_matrix[i, j] = p_val
                p_matrix[j, i] = p_val
                effect_matrix[i, j] = effect
                effect_matrix[j, i] = -effect  # Negative for opposite direction
            
            # Plot heatmap with significance stars
            # Use -log10(p) for better visualization
            log_p_matrix = -np.log10(np.maximum(p_matrix, 1e-10))
            np.fill_diagonal(log_p_matrix, 0)
            
            sns.heatmap(log_p_matrix, 
                       xticklabels=groups,
                       yticklabels=groups,
                       cmap='Reds',
                       center=0,
                       square=True,
                       annot=False,
                       cbar_kws={'label': '-log10(p-value)'},
                       ax=ax)
            
            # Add significance stars
            for i in range(n_groups):
                for j in range(i+1, n_groups):
                    p_val = p_matrix[i, j]
                    if p_val < 0.001:
                        marker = '***'
                    elif p_val < 0.01:
                        marker = '**'
                    elif p_val < 0.05:
                        marker = '*'
                    else:
                        marker = ''
                    
                    if marker:
                        ax.text(j + 0.5, i + 0.5, marker, 
                               ha='center', va='center', 
                               color='white' if log_p_matrix[i, j] > 2 else 'black',
                               fontsize=12, fontweight='bold')
            
            ax.set_title(f'{data["feature"]} by {data["group_by"]}', 
                        fontsize=12, fontweight='bold')
            ax.tick_params(axis='both', rotation=45)
        
        plt.suptitle('Pairwise Comparisons (Bonferroni-corrected p-values)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if output_format == 'base64':
            plot_output = _save_plot_to_base64(fig, dpi=90)
        elif output_format == 'file' and figures_dir:
            save_path = os.path.join(figures_dir, 'pairwise_comparisons_heatmap.png')
            success = _save_plot_to_file(fig, save_path, dpi=PLOT_SAVEFIG_DPI)
            if success:
                plot_output = os.path.join(os.path.basename(figures_dir), 'pairwise_comparisons_heatmap.png')
        else:
            plt.close(fig)
            
    except Exception as e:
        logging.error(f"Failed to plot pairwise comparisons heatmap: {e}", exc_info=False)
        if fig:
            plt.close(fig)
            
    return plot_output


def calculate_feature_entropy(matrix: np.ndarray) -> float:
    """Calculate Shannon entropy of FCGR matrix."""
    if matrix is None or matrix.size == 0:
        return 0.0
    
    # Flatten and normalize
    flat = matrix.flatten()
    flat_positive = np.maximum(flat, 0)
    total = np.sum(flat_positive)
    
    if total <= 0:
        return 0.0
    
    # Calculate probabilities
    probs = flat_positive / total
    
    # Remove zeros for log calculation
    probs_nonzero = probs[probs > 0]
    
    # Shannon entropy
    entropy = -np.sum(probs_nonzero * np.log2(probs_nonzero))
    
    return float(entropy)


def plot_feature_correlations_network(df: pd.DataFrame, features: List[str],
                                    figures_dir: Optional[str] = None,
                                    output_format: str = 'file',
                                    correlation_threshold: float = 0.5) -> Optional[str]:
    """Create network visualization of feature correlations."""
    plot_output = None
    if not PLOTTING_ENABLED or not PLOTTING_LIBS_AVAILABLE or not plt:
        return plot_output
    
    try:
        import networkx as nx
        NETWORKX_AVAILABLE = True
    except ImportError:
        logging.warning("NetworkX not available for correlation network plot")
        NETWORKX_AVAILABLE = False
        return plot_output
    
    fig = None
    try:
        # Calculate correlations
        valid_features = [f for f in features if f in df.columns]
        if len(valid_features) < 3:
            return plot_output
            
        corr_matrix = df[valid_features].corr()
        
        # Create network
        G = nx.Graph()
        
        # Add nodes
        for feature in valid_features:
            G.add_node(feature)
        
        # Add edges for significant correlations
        for i, feat1 in enumerate(valid_features):
            for j, feat2 in enumerate(valid_features[i+1:], i+1):
                corr = corr_matrix.loc[feat1, feat2]
                if abs(corr) > correlation_threshold:
                    G.add_edge(feat1, feat2, weight=abs(corr), correlation=corr)
        
        if G.number_of_edges() == 0:
            logging.warning("No correlations above threshold for network plot")
            return plot_output
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        node_sizes = [800 for _ in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                             node_color='lightblue',
                             edgecolors='darkblue',
                             linewidths=2,
                             ax=ax)
        
        # Draw edges with varying width and color
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        correlations = [G[u][v]['correlation'] for u, v in edges]
        
        # Positive correlations in blue, negative in red
        edge_colors = ['blue' if c > 0 else 'red' for c in correlations]
        edge_widths = [w * 5 for w in weights]  # Scale width
        
        nx.draw_networkx_edges(G, pos, 
                             edge_color=edge_colors,
                             width=edge_widths,
                             alpha=0.6,
                             ax=ax)
        
        # Draw labels
        labels = {node: node.replace('_', '\n') for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, 
                              font_size=10, 
                              font_weight='bold',
                              ax=ax)
        
        # Add edge labels for strong correlations
        edge_labels = {}
        for u, v in edges:
            corr = G[u][v]['correlation']
            if abs(corr) > 0.7:
                edge_labels[(u, v)] = f"{corr:.2f}"
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels, 
                                   font_size=8,
                                   ax=ax)
        
        ax.set_title('Feature Correlation Network\n(Blue: positive, Red: negative)',
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=3, label='Positive correlation'),
            Line2D([0], [0], color='red', linewidth=3, label='Negative correlation'),
            Line2D([0], [0], color='gray', linewidth=1, label=f'Threshold: |r| > {correlation_threshold}')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        plt.tight_layout()
        
        if output_format == 'base64':
            plot_output = _save_plot_to_base64(fig, dpi=90)
        elif output_format == 'file' and figures_dir:
            save_path = os.path.join(figures_dir, 'feature_correlation_network.png')
            success = _save_plot_to_file(fig, save_path, dpi=PLOT_SAVEFIG_DPI)
            if success:
                plot_output = os.path.join(os.path.basename(figures_dir), 'feature_correlation_network.png')
        else:
            plt.close(fig)
            
    except Exception as e:
        logging.error(f"Failed to plot correlation network: {e}", exc_info=False)
        if fig:
            plt.close(fig)
            
    return plot_output


def run_dimensionality_reduction(features: np.ndarray, labels: np.ndarray, label_name: str,
                                 figures_dir: Optional[str] = None,
                                 methods: List[str] = ['PCA', 't-SNE', 'UMAP'],
                                 output_format: str = 'file',
                                 perplexity: int = 30) -> Optional[str]:
    """Enhanced DR with UMAP support and better visualizations."""
    plot_output = None
    if not SKLEARN_AVAILABLE or not StandardScaler or not PCA or not TSNE: 
        return plot_output
    if features is None or labels is None or features.size == 0 or labels.size == 0:
        return plot_output
    n_samples, n_features = features.shape
    try: 
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
    except Exception: 
        return plot_output
    if n_samples <= n_features or n_samples < 3 or n_classes < 2:
        logging.warning(f"Skipping DR for '{label_name}': Insufficient samples/features/classes.")
        return plot_output
    if not PLOTTING_ENABLED or not PLOTTING_LIBS_AVAILABLE or not plt or not sns: 
        return plot_output

    # --- Run Reducers ---
    results = {}
    valid_methods = []
    tsne_n_jobs = 1 if IS_PYODIDE else N_JOBS

    # Scale features first
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    for method in methods:
        method_name = method.upper()
        logging.info(f"Calculating {method_name} for '{label_name}' (Format: {output_format})...")
        
        # Skip compute-intensive methods for large datasets in Pyodide
        if method_name == 'T-SNE' and n_samples > (1000 if IS_PYODIDE else 5000): 
            continue
        if method_name == 'UMAP' and n_samples > (1500 if IS_PYODIDE else 10000): 
            continue
            
        try:
            reduced_data = None
            xlabel = ''
            ylabel = ''
            
            if method_name == 'PCA':
                n_comp = min(2, n_samples - 1, n_features)
                if n_comp < 2: 
                    continue
                reducer = PCA(n_components=n_comp, random_state=RANDOM_STATE)
                reduced_data = reducer.fit_transform(features_scaled)
                var_ratio = reducer.explained_variance_ratio_
                xlabel = f'PC1 ({var_ratio[0]*100:.1f}%)'
                ylabel = f'PC2 ({var_ratio[1]*100:.1f}%)'
                
            elif method_name == 'T-SNE':
                # Adaptive perplexity
                perplexity = min(perplexity, max(5.0, float(n_samples - 1) / 3.0))
                if n_samples <= perplexity + 1: 
                    continue
                    
                # Use PCA for initial dimensionality reduction if needed
                if n_features > 50:
                    pca_init = PCA(n_components=50, random_state=RANDOM_STATE)
                    features_pca = pca_init.fit_transform(features_scaled)
                else:
                    features_pca = features_scaled
                    
                reducer = TSNE(n_components=2, perplexity=perplexity, 
                              learning_rate='auto', init='pca',
                              random_state=RANDOM_STATE, n_jobs=tsne_n_jobs, 
                              n_iter=1000, method='barnes_hut')
                reduced_data = reducer.fit_transform(features_pca)
                xlabel = 't-SNE Dimension 1'
                ylabel = 't-SNE Dimension 2'
                
            elif method_name == 'UMAP' and UMAP is not None:
                # UMAP parameters
                n_neighbors = min(15, n_samples - 1)
                min_dist = 0.1
                
                reducer = UMAP(n_components=2, n_neighbors=n_neighbors,
                              min_dist=min_dist, random_state=RANDOM_STATE,
                              metric='euclidean')
                reduced_data = reducer.fit_transform(features_scaled)
                xlabel = 'UMAP Dimension 1'
                ylabel = 'UMAP Dimension 2'
            else: 
                continue
                
            if reduced_data is not None and reduced_data.shape == (n_samples, 2):
                results[method_name] = {
                    'reduced': reduced_data, 
                    'xlabel': xlabel, 
                    'ylabel': ylabel
                }
                valid_methods.append(method_name)
                logging.info(f"{method_name} calculation successful for '{label_name}'.")
            else: 
                logging.warning(f"{method_name} failed/unexpected shape for '{label_name}'.")
                
        except Exception as e: 
            logging.error(f"DR calc error {method_name} for '{label_name}': {e}", exc_info=False)

    if not valid_methods: 
        return plot_output

    # --- Enhanced Plotting ---
    fig = None
    save_path = None
    plot_filename = None
    
    try:
        # Create separate figures for each method to ensure readability
        for method in valid_methods:
            fig = plt.figure(figsize=(10, 8))
            ax = plt.gca()
            
            res = results[method]
            
            # Create DataFrame for plotting
            reduced_df = pd.DataFrame(res['reduced'], columns=['dim1', 'dim2'])
            reduced_df['label'] = labels
            
            # Prepare color scheme
            unique_labels_sorted = sorted(unique_labels)
            n_colors = len(unique_labels_sorted)
            
            # Use a better color palette for many classes
            if n_colors <= 10:
                palette = sns.color_palette(ENHANCED_PALETTES['categorical'], n_colors=n_colors)
            else:
                palette = sns.color_palette('husl', n_colors=n_colors)
                
            color_map = dict(zip(unique_labels_sorted, palette))
            
            # Plot with enhanced style
            for label in unique_labels_sorted:
                mask = reduced_df['label'] == label
                ax.scatter(reduced_df.loc[mask, 'dim1'], 
                          reduced_df.loc[mask, 'dim2'],
                          c=[color_map[label]], 
                          label=label,
                          s=30,  # Smaller marker size
                          alpha=0.7,
                          edgecolors='white',
                          linewidth=0.5)
            
            ax.set_title(f'{method} - {label_name}', fontsize=14, fontweight='bold')
            ax.set_xlabel(res['xlabel'], fontsize=12)
            ax.set_ylabel(res['ylabel'], fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            # Add subtle grid
            ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
            ax.set_axisbelow(True)
            
            # Set aspect ratio
            ax.set_aspect('equal', adjustable='box')
            
            # Add legend with proper formatting
            if n_colors <= 20:  # Only show legend for reasonable number of classes
                legend = ax.legend(title=label_name.capitalize(),
                                  bbox_to_anchor=(1.01, 0.5), 
                                  loc='center left',
                                  borderaxespad=0., 
                                  fontsize=9,
                                  title_fontsize=10,
                                  frameon=True,
                                  fancybox=True,
                                  shadow=True)
                # Adjust marker size in legend
                for handle in legend.legendHandles:
                    handle.set_sizes([50])
            
            plt.tight_layout()
            
            # Save individual plot
            if output_format == 'base64':
                if plot_output is None:
                    plot_output = {}
                plot_output[method] = _save_plot_to_base64(fig, dpi=90)
                plt.close(fig)
            elif output_format == 'file' and figures_dir:
                plot_filename = safe_filename(f"dim_reduction_{label_name}_{method}.png")
                save_path = os.path.join(figures_dir, plot_filename)
                success = _save_plot_to_file(fig, save_path, dpi=PLOT_SAVEFIG_DPI)
                if success:
                    if plot_output is None:
                        plot_output = {}
                    plot_output[method] = os.path.join(os.path.basename(figures_dir), plot_filename)
                plt.close(fig)
            else:
                plt.close(fig)

    except Exception as e:
        logging.error(f"Failed to plot DR for '{label_name}': {e}", exc_info=False)
        if fig: 
            plt.close(fig)
        plot_output = None

    # Convert dict to string for compatibility
    if isinstance(plot_output, dict) and len(plot_output) > 0:
        # Return first plot path for backward compatibility
        plot_output = list(plot_output.values())[0]
    
    return plot_output

def plot_feature_heatmap_normalized(df: pd.DataFrame, features: List[str], 
                                   group_col: str, figures_dir: Optional[str] = None,
                                   output_format: str = 'file') -> Optional[str]:
    """Plot heatmap with feature-wise normalization within groups."""
    plot_output = None
    if not PLOTTING_ENABLED or not PLOTTING_LIBS_AVAILABLE or not plt or not sns:
        return plot_output
    
    fig = None
    try:
        # Create pivot table
        pivot_data = df.pivot_table(values=features, index=group_col, aggfunc='mean')
        
        # Normalize each feature separately (z-score within feature)
        normalized_data = pivot_data.copy()
        for feature in pivot_data.columns:
            feature_data = pivot_data[feature]
            mean = feature_data.mean()
            std = feature_data.std()
            if std > 0:
                normalized_data[feature] = (feature_data - mean) / std
            else:
                normalized_data[feature] = 0
        
        # Create figure
        fig_height = max(10, len(pivot_data.index) * 0.8)
        fig_width = max(12, len(pivot_data.columns) * 0.8)
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Plot heatmap
        ax = sns.heatmap(normalized_data, 
                        cmap='RdBu_r',
                        center=0,
                        annot=True,
                        fmt='.2f',
                        cbar_kws={'label': 'Normalized Value (z-score)'},
                        linewidths=0.5)
        
        plt.title(f'Normalized Feature Values by {group_col}', fontsize=14, fontweight='bold')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel(group_col, fontsize=12)
        
        # Rotate labels for readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if output_format == 'base64':
            plot_output = _save_plot_to_base64(fig, dpi=90)
        elif output_format == 'file' and figures_dir:
            save_path = os.path.join(figures_dir, f'feature_heatmap_normalized_{group_col}.png')
            success = _save_plot_to_file(fig, save_path, dpi=PLOT_SAVEFIG_DPI)
            if success:
                plot_output = os.path.join(os.path.basename(figures_dir), f'feature_heatmap_normalized_{group_col}.png')
        else:
            plt.close(fig)
            
    except Exception as e:
        logging.error(f"Failed to plot normalized feature heatmap: {e}", exc_info=False)
        if fig:
            plt.close(fig)
            
    return plot_output


def plot_correlation_heatmap(df_features: pd.DataFrame,
                             figures_dir: Optional[str] = None,
                             output_format: str = 'file',
                             method: str = 'pearson',
                             cluster: bool = True) -> Optional[str]:
    """Enhanced correlation heatmap with hierarchical clustering."""
    plot_output = None
    if not PLOTTING_ENABLED or not PLOTTING_LIBS_AVAILABLE or not plt or not sns: 
        return plot_output
    if df_features is None or df_features.empty: 
        return plot_output
    
    try: 
        variances = df_features.var(ddof=0)
        features_to_plot = variances[variances > EPSILON].index.tolist()
    except Exception as var_e: 
        logging.warning(f"Could not calculate variances for heatmap: {var_e}")
        return plot_output
        
    if len(features_to_plot) < 2: 
        return plot_output

    df_plot = df_features[features_to_plot]
    logging.info(f"Generating correlation heatmap for {len(features_to_plot)} features (Format: {output_format})...")

    fig = None
    save_path = None
    plot_filename = None
    
    try:
        # Calculate correlation matrix
        correlation_matrix = df_plot.corr(method=method)
        
        # Perform hierarchical clustering if requested
        if cluster and len(features_to_plot) > 3:
            # Calculate distance matrix - use condensed form
            from scipy.spatial.distance import squareform
            distance_matrix = 1 - np.abs(correlation_matrix)
            # Convert to condensed form
            distance_condensed = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(distance_condensed, method='ward')
            
            # Create dendrogram to get ordering
            dendro = dendrogram(linkage_matrix, no_plot=True)
            cluster_order = dendro['leaves']
            
            # Reorder correlation matrix
            correlation_matrix = correlation_matrix.iloc[cluster_order, cluster_order]
            features_to_plot = [features_to_plot[i] for i in cluster_order]
        
        # Determine figure size based on number of features
        fig_width = max(12, len(features_to_plot) * 0.7)
        fig_height = max(10, len(features_to_plot) * 0.6)
        
        # Create figure with better layout
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        
        # Create custom diverging colormap
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        
        # Plot heatmap with enhancements
        ax = sns.heatmap(correlation_matrix, 
                        mask=mask,
                        annot=True, 
                        fmt=".2f", 
                        cmap=cmap,
                        vmin=-1, 
                        vmax=1, 
                        center=0,
                        square=True,
                        linewidths=0.5,
                        cbar_kws={"shrink": .8, "label": f"{method.capitalize()} Correlation"},
                        annot_kws={"size": 8 if len(features_to_plot) < 20 else 6})
        
        # Customize appearance
        plt.title(f'Feature Correlation Matrix ({method.capitalize()})', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        
        # Add text about clustering
        if cluster and len(features_to_plot) > 3:
            plt.text(0.5, -0.1, 'Features ordered by hierarchical clustering', 
                    transform=ax.transAxes, ha='center', fontsize=9, style='italic')
        
        plt.tight_layout()

        if output_format == 'base64': 
            plot_output = _save_plot_to_base64(fig, dpi=90)
        elif output_format == 'file' and figures_dir:
            plot_filename = f"feature_correlation_heatmap_{method}.png"
            save_path = os.path.join(figures_dir, plot_filename)
            success = _save_plot_to_file(fig, save_path, dpi=PLOT_SAVEFIG_DPI)
            if success: 
                plot_output = os.path.join(os.path.basename(figures_dir), plot_filename)
        else: 
            plt.close(fig)
        fig = None

    except Exception as e:
        logging.error(f"Failed to plot correlation heatmap: {e}", exc_info=False)
        if fig: 
            plt.close(fig)
        plot_output = None

    return plot_output


def plot_feature_importance(feature_names: List[str], 
                           importances: np.ndarray,
                           std_devs: Optional[np.ndarray] = None,
                           title: str = "Feature Importances",
                           figures_dir: Optional[str] = None,
                           output_format: str = 'file',
                           top_n: int = 10) -> Optional[str]:
    """Plot feature importances with error bars."""
    plot_output = None
    if not PLOTTING_ENABLED or not PLOTTING_LIBS_AVAILABLE or not plt: 
        return plot_output
    if not feature_names or importances is None or len(feature_names) != len(importances):
        return plot_output
        
    fig = None
    try:
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        # Create figure
        fig_width = max(10, len(indices) * 0.5)
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        
        # Create bar plot
        x_pos = np.arange(len(indices))
        colors = plt.cm.viridis(importances[indices] / importances[indices].max())
        
        bars = ax.bar(x_pos, importances[indices], color=colors, edgecolor='black', linewidth=1)
        
        # Add error bars if provided
        if std_devs is not None:
            ax.errorbar(x_pos, importances[indices], yerr=std_devs[indices], 
                       fmt='none', color='black', capsize=5, alpha=0.7)
        
        # Customize plot
        ax.set_xticks(x_pos)
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Importance', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, idx) in enumerate(zip(bars, indices)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{importances[idx]:.3f}',
                   ha='center', va='bottom', fontsize=8)
        
        # Add grid
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Save
        if output_format == 'base64':
            plot_output = _save_plot_to_base64(fig, dpi=90)
        elif output_format == 'file' and figures_dir:
            plot_filename = safe_filename(f"{title.lower().replace(' ', '_')}.png")
            save_path = os.path.join(figures_dir, plot_filename)
            success = _save_plot_to_file(fig, save_path, dpi=PLOT_SAVEFIG_DPI)
            if success:
                plot_output = os.path.join(os.path.basename(figures_dir), plot_filename)
        else:
            plt.close(fig)
            
    except Exception as e:
        logging.error(f"Failed to plot feature importances: {e}", exc_info=False)
        if fig:
            plt.close(fig)
            
    return plot_output


def plot_learning_curves(train_sizes: np.ndarray,
                        train_scores: np.ndarray,
                        val_scores: np.ndarray,
                        title: str = "Learning Curves",
                        figures_dir: Optional[str] = None,
                        output_format: str = 'file') -> Optional[str]:
    """Plot learning curves with confidence intervals."""
    plot_output = None
    if not PLOTTING_ENABLED or not PLOTTING_LIBS_AVAILABLE or not plt:
        return plot_output
        
    fig = None
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot with confidence intervals
        ax.plot(train_sizes, train_mean, 'o-', color='#3498db', label='Training score', linewidth=2)
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                       alpha=0.2, color='#3498db')
        
        ax.plot(train_sizes, val_mean, 'o-', color='#e74c3c', label='Cross-validation score', linewidth=2)
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                       alpha=0.2, color='#e74c3c')
        
        # Customize plot
        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('Accuracy Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set y-axis limits
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        # Save
        if output_format == 'base64':
            plot_output = _save_plot_to_base64(fig, dpi=90)
        elif output_format == 'file' and figures_dir:
            plot_filename = "learning_curves.png"
            save_path = os.path.join(figures_dir, plot_filename)
            success = _save_plot_to_file(fig, save_path, dpi=PLOT_SAVEFIG_DPI)
            if success:
                plot_output = os.path.join(os.path.basename(figures_dir), plot_filename)
        else:
            plt.close(fig)
            
    except Exception as e:
        logging.error(f"Failed to plot learning curves: {e}", exc_info=False)
        if fig:
            plt.close(fig)
            
    return plot_output


def run_comprehensive_statistics(df: pd.DataFrame, 
                                features: List[str],
                                grouping_vars: List[str]) -> Dict[str, Any]:
    """Run comprehensive statistical analysis on features."""
    results = {
        'normality_tests': {},
        'correlation_tests': {},
        'group_comparisons': {},
        'chi_square_tests': {},
        'summary_stats': {}
    }
    
    # Normality tests for each feature
    for feat in features:
        if feat in df.columns:
            data = df[feat].dropna()
            if len(data) >= 8:
                try:
                    stat, p_value = normaltest(data)
                    results['normality_tests'][feat] = {
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'is_normal': p_value > 0.05
                    }
                except Exception as e:
                    logging.debug(f"Normality test failed for {feat}: {e}")
    
    # Correlation tests between features
    if len(features) >= 2:
        for i, feat1 in enumerate(features):
            for feat2 in features[i+1:]:
                if feat1 in df.columns and feat2 in df.columns:
                    data1 = df[feat1].dropna()
                    data2 = df[feat2].dropna()
                    # Get common indices
                    common_idx = data1.index.intersection(data2.index)
                    if len(common_idx) >= 3:
                        try:
                            # Pearson correlation
                            r_pearson, p_pearson = stats.pearsonr(data1[common_idx], data2[common_idx])
                            # Spearman correlation
                            r_spearman, p_spearman = stats.spearmanr(data1[common_idx], data2[common_idx])
                            
                            results['correlation_tests'][f"{feat1}_vs_{feat2}"] = {
                                'pearson_r': float(r_pearson),
                                'pearson_p': float(p_pearson),
                                'spearman_r': float(r_spearman),
                                'spearman_p': float(p_spearman),
                                'n_samples': len(common_idx)
                            }
                        except Exception as e:
                            logging.debug(f"Correlation test failed for {feat1} vs {feat2}: {e}")
    
    # Chi-square tests for categorical variables
    if len(grouping_vars) >= 2:
        for i, var1 in enumerate(grouping_vars):
            for var2 in grouping_vars[i+1:]:
                if var1 in df.columns and var2 in df.columns:
                    try:
                        # Create contingency table
                        cont_table = pd.crosstab(df[var1], df[var2])
                        if cont_table.size > 0:
                            chi2, p_value, dof, expected = chi2_contingency(cont_table)
                            
                            # Calculate Cramér's V for effect size
                            n = cont_table.sum().sum()
                            min_dim = min(cont_table.shape[0] - 1, cont_table.shape[1] - 1)
                            cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
                            
                            results['chi_square_tests'][f"{var1}_vs_{var2}"] = {
                                'chi2': float(chi2),
                                'p_value': float(p_value),
                                'dof': int(dof),
                                'cramers_v': float(cramers_v),
                                'n_samples': int(n)
                            }
                    except Exception as e:
                        logging.debug(f"Chi-square test failed for {var1} vs {var2}: {e}")
    
    # Extended summary statistics
    for feat in features:
        if feat in df.columns:
            data = df[feat].dropna()
            if len(data) > 0:
                try:
                    results['summary_stats'][feat] = {
                        'count': int(len(data)),
                        'mean': float(data.mean()),
                        'std': float(data.std()),
                        'min': float(data.min()),
                        'q1': float(data.quantile(0.25)),
                        'median': float(data.median()),
                        'q3': float(data.quantile(0.75)),
                        'max': float(data.max()),
                        'skewness': float(stats.skew(data)),
                        'kurtosis': float(stats.kurtosis(data)),
                        'cv': float(data.std() / data.mean()) if data.mean() != 0 else None
                    }
                except Exception as e:
                    logging.debug(f"Summary stats failed for {feat}: {e}")
    
    return results