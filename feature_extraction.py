# -*- coding: utf-8 -*-
"""
Feature Extraction from FCGR Matrices.
Includes statistical, Haralick texture, Hu moments, and Fractal Dimension features.
Utilizes Numba for FD optimization (native only).
"""
import numpy as np
import logging
from scipy import stats
from math import log2, log
from .config import FCGR_DIM, EPSILON
from .utils import NUMBA_AVAILABLE # Use flag from utils

# --- Conditional Dependency Imports and Availability Flags ---

OPENCV_AVAILABLE = False
cv2 = None
try:
    import cv2 as cv2_
    cv2 = cv2_
    OPENCV_AVAILABLE = True
    logging.debug("OpenCV (cv2) found.")
except ImportError:
    logging.debug("OpenCV (cv2) not found. Hu moments disabled.")

SKIMAGE_AVAILABLE = False
graycomatrix = None; graycoprops = None; shannon_entropy = None
try:
    from skimage.feature import graycomatrix as gcm, graycoprops as gcp
    graycomatrix = gcm; graycoprops = gcp
    from skimage.measure import shannon_entropy as se
    shannon_entropy = se
    SKIMAGE_AVAILABLE = True
    logging.debug("scikit-image found.")
except ImportError:
    logging.debug("scikit-image not found. Haralick texture and Shannon entropy disabled.")


# --- Conditional Numba FD Compilation ---
_boxcount_loop_numba = None
if NUMBA_AVAILABLE:
    try:
        import numba # Import guarded by flag
        @numba.jit(nopython=True, cache=True, fastmath=True)
        def _boxcount_loop_numba_compiled(pixels_row: np.ndarray, pixels_col: np.ndarray, n: int, scales: np.ndarray):
            """Numba-optimized loop for box counting (no logging inside)."""
            n_scales = len(scales)
            counts = np.zeros(n_scales, dtype=np.int64)
            valid_indices = np.zeros(n_scales, dtype=np.bool_)
            max_grid_size = 100_000_000 # Safety limit

            for i in range(n_scales):
                scale = scales[i]
                if scale <= 0: continue # Skip invalid scales
                # Use ceiling division for grid dimensions
                nx = int(np.ceil(n / scale))
                ny = int(np.ceil(n / scale))
                grid_size = nx * ny
                count = 0 # Initialize count for this scale

                if grid_size <= 0 or grid_size >= max_grid_size:
                     if nx > 0 and ny > 0: # Handle scale >= n case
                         count = 1 if len(pixels_row) > 0 else 0
                     else: # Grid too large or invalid
                         count = -1 # Signal invalid scale for later check
                else:
                    occupied = np.zeros((ny, nx), dtype=np.bool_)
                    num_pixels = len(pixels_row)
                    for j in range(num_pixels):
                        # Use integer division for box index calculation
                        r = int(pixels_row[j] // scale)
                        c = int(pixels_col[j] // scale)
                        # Check bounds carefully
                        if 0 <= r < ny and 0 <= c < nx:
                            if not occupied[r, c]:
                                occupied[r, c] = True
                                count += 1
                # Store valid counts (count > 0)
                if count > 0:
                    counts[i] = count
                    valid_indices[i] = True
                # If count is 0 or -1, valid_indices remains False

            return counts[valid_indices], scales[valid_indices]
        _boxcount_loop_numba = _boxcount_loop_numba_compiled
        logging.debug("Numba FD function compiled successfully.")
    except Exception as e:
        logging.error(f"Numba FD compilation failed: {e}. Numba FD disabled.", exc_info=False)
        _boxcount_loop_numba = None


# --- Python Fallback for Box Counting ---
def _boxcount_histogram_python(pixels_row: np.ndarray, pixels_col: np.ndarray, n: int, scales: np.ndarray):
    """Python fallback for box counting using numpy histogram2d."""
    counts = []
    valid_scales_list = []
    # logging.debug("Using Python histogram2d for FD box counting.")
    for scale in scales:
        if scale <= 0: continue
        # Ensure bins cover the entire range [0, n] properly
        # Add epsilon to upper bound to include points exactly at n if needed, though pixel coords should be < n
        bins = [np.arange(0, n + scale, scale), np.arange(0, n + scale, scale)]
        try:
            # Filter pixels to be strictly within [0, n) range
            valid_pix_mask = (pixels_row >= 0) & (pixels_row < n) & (pixels_col >= 0) & (pixels_col < n)
            if not np.any(valid_pix_mask): continue # Skip scale if no valid pixels

            # Calculate 2D histogram
            H, _, _ = np.histogram2d(
                pixels_row[valid_pix_mask],
                pixels_col[valid_pix_mask],
                bins=bins
            )
            # Count number of bins (boxes) with pixels
            count = np.count_nonzero(H)
            if count > 0:
                counts.append(count)
                valid_scales_list.append(scale)
        except Exception as e:
            logging.warning(f"Error during histogram2d for FD scale {scale}: {e}. Skipping.", exc_info=False)
            continue
    return np.array(counts, dtype=np.int64), np.array(valid_scales_list, dtype=np.int64)


# --- Fractal Dimension Calculation ---
def calculate_fractal_dimension(matrix: np.ndarray) -> float:
    """Calculates Fractal Dimension (Box-Counting), using Numba if available/native."""
    default_fd = 0.0
    if matrix is None or matrix.size == 0: return default_fd

    try: # Wrap preprocessing in try/except
        # --- Binarize ---
        threshold = max(np.mean(matrix) * 0.01, EPSILON)
        binary_matrix = matrix > threshold
        if not np.any(binary_matrix):
             logging.debug("FD calc skipped: Matrix below threshold.")
             return default_fd
        pixels_row, pixels_col = np.where(binary_matrix)
        n = matrix.shape[0]
        if n <= 1: return default_fd

        # --- Determine Scales ---
        max_power = int(np.log2(n))
        if max_power <= 1:
             logging.debug(f"FD calc skipped: Matrix size {n}x{n} too small for multiple scales.")
             return default_fd
        # Use powers of 2 for scales: e.g., for 64x64 (max_power=6), scales are 32, 16, 8, 4, 2
        scales_int = 2**np.arange(max_power - 1, 0, -1)
        scales_int = scales_int[scales_int >= 1] # Ensure positive
        if len(scales_int) < 2:
             logging.debug(f"FD calc skipped: Insufficient scales ({len(scales_int)}) for matrix size {n}.")
             return default_fd
    except Exception as e:
        logging.warning(f"FD preprocessing failed: {e}. Returning default.", exc_info=False)
        return default_fd

    # --- Perform Box Counting ---
    counts = np.array([], dtype=np.int64)
    valid_scales = np.array([], dtype=np.int64)
    use_numba_logic = _boxcount_loop_numba is not None

    if use_numba_logic:
        try:
            # logging.debug(f"Attempting Numba FD box counting...")
            # Ensure inputs are contiguous and correct type if Numba is picky
            p_row = np.ascontiguousarray(pixels_row, dtype=np.float64)
            p_col = np.ascontiguousarray(pixels_col, dtype=np.float64)
            scls = np.ascontiguousarray(scales_int, dtype=np.float64)
            counts, valid_scales = _boxcount_loop_numba(p_row, p_col, n, scls)
            counts = counts.astype(np.int64); valid_scales = valid_scales.astype(np.int64)
            # logging.debug(f"Numba FD successful ({len(counts)} points).")
        except Exception as e:
            logging.warning(f"Numba FD failed during execution: {e}. Falling back to Python.", exc_info=False)
            use_numba_logic = False # Ensure fallback

    if not use_numba_logic: # Fallback
        if len(pixels_row) > 0:
            # logging.debug("Using Python histogram for FD box counting.")
            counts, valid_scales = _boxcount_histogram_python(pixels_row, pixels_col, n, scales_int)
        else:
             return default_fd # No pixels found

    # --- Linear Regression ---
    if len(counts) < 2 or len(valid_scales) < 2:
        logging.debug(f"FD calc failed: Insufficient points ({len(counts)}) for regression.")
        return default_fd
    try:
        # Use log base 2
        log_counts = np.log2(counts.astype(np.float64))
        log_scales = np.log2(np.maximum(valid_scales.astype(np.float64), EPSILON))
        # Fit log(count) vs log(scale) -> slope = -FD
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_scales, log_counts)
        fractal_dim = -slope # Dimension is negative of the slope

        if not np.isfinite(fractal_dim):
             logging.warning(f"FD calc resulted in non-finite value ({fractal_dim}). Returning default.")
             return default_fd

        # Clamp result to physically meaningful range [0, 2] for a 2D representation
        fractal_dim_clamped = max(0.0, min(fractal_dim, 2.0))
        if fractal_dim_clamped != fractal_dim:
            logging.debug(f"FD value {fractal_dim:.4f} clamped to {fractal_dim_clamped:.4f}.")

        logging.debug(f"FD calculated: {fractal_dim_clamped:.4f} (R^2 = {r_value**2:.3f})")
        return float(fractal_dim_clamped)
    except ValueError as e: # Catch LinAlgError etc.
        logging.error(f"FD linear regression failed: {e}. Returning default.", exc_info=False)
        return default_fd
    except Exception as e:
        logging.error(f"Unexpected error during FD regression: {e}. Returning default.", exc_info=True)
        return default_fd


# --- Haralick Textures ---
def calculate_haralick_textures(matrix: np.ndarray) -> dict:
    """Calculates Haralick texture features (requires scikit-image)."""
    feature_names = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    default_features = {'contrast': 0.0, 'dissimilarity': 0.0, 'homogeneity': 1.0, 'energy': 1.0, 'correlation': 1.0, 'ASM': 1.0}

    if not SKIMAGE_AVAILABLE or graycomatrix is None or graycoprops is None:
        # logging.debug("Haralick skipped: scikit-image unavailable.")
        return default_features.copy()
    if matrix is None or matrix.size == 0: return default_features.copy()

    try:
        img_min, img_max = np.min(matrix), np.max(matrix)
        # Handle constant image case
        if abs(img_max - img_min) < EPSILON: return default_features.copy()

        # Scale matrix to 0-255 uint8 for GLCM
        # Use np.errstate to suppress potential warnings during scaling if needed
        with np.errstate(divide='ignore', invalid='ignore'):
             scaled_matrix = (matrix - img_min) / (img_max - img_min + EPSILON)
             # Clip to ensure values are in [0, 1] before multiplying
             scaled_matrix = np.nan_to_num(np.clip(scaled_matrix, 0, 1), nan=0.0)
        gray_img = (scaled_matrix * 255).astype(np.uint8)

        if np.all(gray_img == gray_img[0, 0]): return default_features.copy()

        # Calculate GLCM
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(gray_img, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

        # Calculate properties
        features = {}
        for prop in feature_names:
            try:
                prop_values = graycoprops(glcm, prop)
                mean_val = np.nanmean(prop_values)
                # Use default value if mean is NaN or Inf
                features[prop] = default_features.get(prop, 0.0) if not np.isfinite(mean_val) else float(mean_val)
            except Exception as prop_e:
                logging.warning(f"Error calculating Haralick '{prop}': {prop_e}. Using default.", exc_info=False)
                features[prop] = default_features.get(prop, 0.0)
        return features

    except Exception as e:
        logging.error(f"Haralick feature calculation failed entirely: {e}. Returning defaults.", exc_info=False)
        return default_features.copy()


# --- Hu Moments ---
def calculate_hu_moments(matrix: np.ndarray) -> dict:
    """Calculates Hu moments (requires OpenCV)."""
    num_hu_moments = 7
    feature_prefix = 'hu_moment_'
    default_features = {f'{feature_prefix}{i}': 0.0 for i in range(num_hu_moments)}

    if not OPENCV_AVAILABLE or cv2 is None: return default_features.copy()
    if matrix is None or matrix.size == 0: return default_features.copy()

    try:
        img_min, img_max = np.min(matrix), np.max(matrix)
        if abs(img_max - img_min) < EPSILON: return default_features.copy()

        # Scale to 0-255 uint8
        with np.errstate(divide='ignore', invalid='ignore'):
            scaled_matrix = (matrix - img_min) / (img_max - img_min + EPSILON)
            scaled_matrix = np.nan_to_num(np.clip(scaled_matrix, 0, 1), nan=0.0)
        gray_img = (scaled_matrix * 255).astype(np.uint8)

        # Calculate moments
        moments = cv2.moments(gray_img)
        if abs(moments.get('m00', 0.0)) < EPSILON: # Use .get for safety
            logging.debug("Hu moments skipped: m00 is near zero.")
            return default_features.copy()

        hu_m = cv2.HuMoments(moments)

        # Log-scale Hu moments: -sign(h) * log10(|h|)
        hu_log_scaled = np.zeros_like(hu_m)
        for i in range(hu_m.shape[0]):
            val = hu_m[i, 0]
            if abs(val) > EPSILON:
                # Add epsilon inside log10 for absolute numerical safety, though check > EPSILON helps
                hu_log_scaled[i, 0] = -np.sign(val) * np.log10(abs(val) + EPSILON)
            # Else it remains 0.0

        # Replace potential NaN/inf (should be less likely now)
        hu_log_scaled = np.nan_to_num(hu_log_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        features = {f'{feature_prefix}{i}': float(hu_log_scaled[i, 0]) for i in range(num_hu_moments)}
        return features

    except cv2.error as cv_e: # Catch specific cv2 errors
        logging.error(f"OpenCV error during Hu moments: {cv_e}. Returning zeros.", exc_info=False)
        return default_features.copy()
    except Exception as e:
        logging.error(f"Unexpected error during Hu moments: {e}. Returning zeros.", exc_info=True)
        return default_features.copy()


# --- Statistical Features ---
def calculate_statistical_features(matrix: np.ndarray) -> dict:
    """Calculates basic statistical features and Shannon entropy (requires scikit-image)."""
    feature_names = ['mean', 'variance', 'skewness', 'kurtosis', 'shannon_entropy']
    default_features = {'mean': 0.0, 'variance': 0.0, 'skewness': 0.0, 'kurtosis': -3.0, 'shannon_entropy': 0.0}
    if matrix is None or matrix.size == 0: return default_features.copy()

    try:
        flat_matrix = matrix.flatten()
        if flat_matrix.size == 0: return default_features.copy()

        features = {}
        # Use numpy for basic stats for potentially better performance on large arrays
        features['mean'] = float(np.mean(flat_matrix))
        features['variance'] = float(np.var(flat_matrix))

        # Use scipy.stats for skew/kurtosis as they handle edge cases (like constant) well
        is_constant = features['variance'] < EPSILON
        if is_constant:
            features['skewness'] = 0.0
            features['kurtosis'] = -3.0 # Fisher's definition
        else:
            try: features['skewness'] = float(stats.skew(flat_matrix))
            except ValueError: features['skewness'] = 0.0
            try: features['kurtosis'] = float(stats.kurtosis(flat_matrix, fisher=True))
            except ValueError: features['kurtosis'] = -3.0

        # Shannon Entropy
        if SKIMAGE_AVAILABLE and shannon_entropy is not None:
            try:
                # Ensure non-negative probabilities summing to 1
                matrix_non_neg = np.maximum(matrix, 0)
                matrix_sum = np.sum(matrix_non_neg)
                if matrix_sum > EPSILON:
                    prob_matrix = matrix_non_neg / matrix_sum
                    # Ensure no NaN/Inf in prob_matrix before entropy calc
                    prob_matrix = np.nan_to_num(prob_matrix, nan=0.0)
                    entropy = shannon_entropy(prob_matrix)
                    features['shannon_entropy'] = float(entropy) if np.isfinite(entropy) else 0.0
                else: features['shannon_entropy'] = 0.0
            except Exception as e:
                logging.warning(f"Error calculating Shannon entropy: {e}. Setting to 0.0.", exc_info=False)
                features['shannon_entropy'] = default_features['shannon_entropy']
        else: features['shannon_entropy'] = default_features['shannon_entropy']

        # Final check for NaN/inf in all calculated features
        for key in feature_names:
            if key not in features or not np.isfinite(features[key]):
                 features[key] = default_features[key]

        return features

    except Exception as e:
         logging.error(f"Error calculating statistical features: {e}", exc_info=True)
         return default_features.copy()


def extract_all_features(fcgr_matrix: np.ndarray) -> dict:
    """Calculates and aggregates all implemented features for a given FCGR matrix."""
    # Define default values for all possible features this function calculates
    default_values = {
        'mean': 0.0, 'variance': 0.0, 'skewness': 0.0, 'kurtosis': -3.0, 'shannon_entropy': 0.0,
        'contrast': 0.0, 'dissimilarity': 0.0, 'homogeneity': 1.0, 'energy': 1.0, 'correlation': 1.0, 'ASM': 1.0,
        **{f'hu_moment_{i}': 0.0 for i in range(7)},
        'fractal_dimension': 0.0,
        'fcgr_entropy': 0.0,  # New feature
        'quadrant_ratio_AA_GC': 0.0,  # New feature
        'quadrant_ratio_AT_CG': 0.0,  # New feature
        'center_mass_x': 0.5,  # New feature
        'center_mass_y': 0.5   # New feature
    }

    # --- Input Validation ---
    if fcgr_matrix is None or not isinstance(fcgr_matrix, np.ndarray) or fcgr_matrix.ndim != 2:
        logging.warning("Invalid FCGR matrix input for feature extraction. Returning defaults.")
        return default_values.copy()
    if fcgr_matrix.shape[0] != fcgr_matrix.shape[1]:
         logging.warning(f"Non-square matrix ({fcgr_matrix.shape}) passed to feature extraction. Proceeding.")
         # Allow non-square but calculations might behave unexpectedly if not intended
    if fcgr_matrix.shape[0] != FCGR_DIM: # Check against configured dimension
         logging.warning(f"FCGR matrix shape {fcgr_matrix.shape} != expected ({FCGR_DIM}x{FCGR_DIM}). Proceeding.")
    if fcgr_matrix.size == 0 or np.all(np.abs(fcgr_matrix) < EPSILON):
        logging.debug("FCGR matrix is empty or effectively zero. Returning default features.")
        zero_defaults = default_values.copy(); zero_defaults['mean'] = 0.0 # Ensure mean is 0
        return zero_defaults

    # --- Calculate Feature Sets Safely ---
    calculated_features = {}
    
    # Calculate new features
    try:
        # FCGR-specific entropy
        from .analysis import calculate_feature_entropy
        calculated_features['fcgr_entropy'] = calculate_feature_entropy(fcgr_matrix)
        
        # Quadrant analysis
        mid = fcgr_matrix.shape[0] // 2
        if mid > 0:
            # AA quadrant (bottom-left), GC quadrant (top-right)
            aa_sum = np.sum(fcgr_matrix[:mid, :mid])
            gc_sum = np.sum(fcgr_matrix[mid:, mid:])
            total_sum = np.sum(fcgr_matrix) + EPSILON
            
            # AT quadrant (bottom-right), CG quadrant (top-left)  
            at_sum = np.sum(fcgr_matrix[:mid, mid:])
            cg_sum = np.sum(fcgr_matrix[mid:, :mid])
            
            calculated_features['quadrant_ratio_AA_GC'] = float((aa_sum + EPSILON) / (gc_sum + EPSILON))
            calculated_features['quadrant_ratio_AT_CG'] = float((at_sum + EPSILON) / (cg_sum + EPSILON))
        
        # Center of mass
        total = np.sum(fcgr_matrix) + EPSILON
        y_indices, x_indices = np.mgrid[0:fcgr_matrix.shape[0], 0:fcgr_matrix.shape[1]]
        center_x = np.sum(x_indices * fcgr_matrix) / total / fcgr_matrix.shape[1]
        center_y = np.sum(y_indices * fcgr_matrix) / total / fcgr_matrix.shape[0]
        calculated_features['center_mass_x'] = float(center_x)
        calculated_features['center_mass_y'] = float(center_y)
        
    except Exception as e:
        logging.error(f"New feature calculation failed: {e}", exc_info=False)
    
    # Define functions to call
    calculation_map = {
        'stats': calculate_statistical_features,
        'haralick': calculate_haralick_textures,
        'hu': calculate_hu_moments,
        'fd': calculate_fractal_dimension
    }
    # Execute each calculation within a try/except block
    for name, func in calculation_map.items():
        try:
            result = func(fcgr_matrix)
            if isinstance(result, dict):
                calculated_features.update(result)
            elif name == 'fd': # FD returns a float
                calculated_features['fractal_dimension'] = result
            else:
                 logging.warning(f"Unexpected return type ({type(result)}) from feature function '{name}'")
        except Exception as e:
             logging.error(f"{name.capitalize()} feature calculation failed: {e}", exc_info=True)
             # Defaults will be used later

    # --- Final Consolidation ---
    final_features = {}
    for key in default_values: # Iterate through all *expected* keys
        # Get calculated value, fall back to default if not calculated or error occurred
        value = calculated_features.get(key, default_values[key])
        # Ensure the final value is a finite standard Python float
        if isinstance(value, (np.floating, np.integer, int)):
            value = float(value) # Convert numpy types
        if not isinstance(value, float) or not np.isfinite(value):
            # Log if replacing a calculated non-finite value
            # if key in calculated_features and calculated_features[key] is not None:
            #    logging.debug(f"Replacing non-finite calculated value for '{key}' ({value}) with default ({default_values[key]}).")
            final_features[key] = default_values[key] # Use default if non-finite or wrong type
        else:
            final_features[key] = value

    logging.debug(f"Feature extraction complete for one matrix.")
    return final_features