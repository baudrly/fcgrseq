# -*- coding: utf-8 -*-
"""
Utility Functions for FCGR Analyzer.
Includes environment detection (Pyodide/Native) and conditional behavior.
"""
import os
import sys
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings
import numpy as np
import pandas as pd
import shutil

# --- Environment Detection ---
IS_PYODIDE = 'pyodide' in sys.modules or 'js' in sys.modules
_ENV_MSG_SUFFIX = "(Pyodide Environment)" if IS_PYODIDE else "(Native Environment)"

# --- Conditional Imports and Availability Flags ---

# TensorFlow
TENSORFLOW_AVAILABLE = False
tf = None
mixed_precision = None
if not IS_PYODIDE:
    try:
        import tensorflow as tf_
        tf = tf_
        from tensorflow.keras import mixed_precision as mp_
        mixed_precision = mp_
        TENSORFLOW_AVAILABLE = True
        logging.debug(f"TensorFlow found {_ENV_MSG_SUFFIX}")
    except ImportError:
        logging.debug(f"TensorFlow not found {_ENV_MSG_SUFFIX}")
    except Exception as e:
        logging.warning(f"TensorFlow import failed unexpectedly: {e} {_ENV_MSG_SUFFIX}")
else:
    logging.debug(f"TensorFlow import skipped {_ENV_MSG_SUFFIX}")

# Numba
NUMBA_AVAILABLE = False
numba = None
if not IS_PYODIDE:
    try:
        import numba as nb_
        numba = nb_
        NUMBA_AVAILABLE = True
        logging.debug(f"Numba found {_ENV_MSG_SUFFIX}")
    except ImportError:
        logging.debug(f"Numba not found {_ENV_MSG_SUFFIX}")
    except Exception as e:
        logging.warning(f"Numba import failed unexpectedly: {e} {_ENV_MSG_SUFFIX}")
else:
    logging.debug(f"Numba import skipped {_ENV_MSG_SUFFIX}")

# Joblib
JOBLIB_AVAILABLE = False
Memory = None
if not IS_PYODIDE:
    try:
        from joblib import Memory as jm_
        Memory = jm_
        JOBLIB_AVAILABLE = True
        logging.debug(f"Joblib found {_ENV_MSG_SUFFIX}")
    except ImportError:
        logging.debug(f"Joblib not found {_ENV_MSG_SUFFIX}")
    except Exception as e:
        logging.warning(f"Joblib import failed unexpectedly: {e} {_ENV_MSG_SUFFIX}")
else:
    logging.debug(f"Joblib features disabled {_ENV_MSG_SUFFIX}")


# --- Logging Setup ---
_logging_configured = False

def setup_logging(log_level=logging.INFO):
    """Configures application-wide logging, avoiding reconfiguration."""
    global _logging_configured
    if _logging_configured and logging.getLogger().hasHandlers():
        # Only update level if already configured
        current_level = logging.getLogger().getEffectiveLevel()
        if current_level != log_level:
            logging.getLogger().setLevel(log_level)
            logging.info(f"Log level updated to {logging.getLevelName(log_level)}")
        else:
            logging.debug("Logging already configured with the same level.")
        return

    log_env = "Pyodide" if IS_PYODIDE else "Native"
    log_format = f'%(asctime)s - %(levelname)s [{log_env}] - [%(name)s.%(funcName)s] %(message)s' # Use logger name
    date_format = '%Y-%m-%d %H:%M:%S'

    # Use basicConfig with force=True to handle potential re-init scenarios
    logging.basicConfig(level=log_level, format=log_format, datefmt=date_format, force=True)

    # Get the root logger after basicConfig has run
    root_logger = logging.getLogger()

    # Suppress verbose logging from libraries known to be chatty
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    if TENSORFLOW_AVAILABLE: logging.getLogger("tensorflow").setLevel(logging.ERROR)
    if NUMBA_AVAILABLE: logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # Filter specific warnings
    warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.metrics._classification')
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    logging.info(f"Logging configured (Level: {logging.getLevelName(log_level)})")
    _logging_configured = True


# --- Network Session Setup ---
def setup_requests_session(retries, backoff_factor, status_forcelist):
    """Creates a requests session with retry logic."""
    session = requests.Session()
    try:
        # Ensure status_forcelist is a tuple or frozenset of integers
        safe_status_list = tuple(int(s) for s in status_forcelist if isinstance(s, (int, str)) and str(s).isdigit())

        retry_strategy = Retry(
            total=int(retries),
            read=int(retries),
            connect=int(retries),
            backoff_factor=float(backoff_factor),
            status_forcelist=safe_status_list,
            allowed_methods=frozenset(['GET', 'POST'])
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        logging.debug("Requests session configured with retries.")
    except Exception as e:
        logging.warning(f"Failed to configure full requests retry session: {e}. Using basic session.")
    return session


# --- GPU and Mixed Precision Setup ---
def setup_gpu_and_mixed_precision(enable_mixed_precision_if_supported):
    """Configures GPU and mixed precision (Native TensorFlow only)."""
    if IS_PYODIDE or not TENSORFLOW_AVAILABLE or not tf or not mixed_precision:
        logging.debug("GPU/Mixed Precision setup skipped (Pyodide/TensorFlow/MP unavailable).")
        return False, False

    gpu_available = False
    mixed_precision_enabled = False
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_to_use = gpus[0]
            tf.config.experimental.set_memory_growth(gpu_to_use, True)
            # Optional: tf.config.set_visible_devices(gpu_to_use, 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            logging.info(f"Using GPU: {len(logical_gpus)} Logical GPU(s) from {gpu_to_use.name}.")
            gpu_available = True

            if enable_mixed_precision_if_supported:
                try:
                    gpu_details = tf.config.experimental.get_device_details(gpu_to_use)
                    compute_capability = gpu_details.get('compute_capability')
                    # Ensure compute_capability is a tuple (major, minor)
                    if isinstance(compute_capability, tuple) and len(compute_capability) == 2 and compute_capability >= (7, 0):
                         policy = mixed_precision.Policy('mixed_float16')
                         mixed_precision.set_global_policy(policy)
                         mixed_precision_enabled = True
                         logging.info(f"GPU CC {compute_capability}. Enabling mixed precision ('mixed_float16').")
                    elif compute_capability:
                         logging.info(f"GPU CC {compute_capability}. Mixed precision requires >= 7.0, disabled.")
                    else:
                         logging.info("Could not determine GPU Compute Capability. Mixed precision disabled.")
                except Exception as mp_e:
                     logging.warning(f"Mixed precision setup failed: {mp_e}. Continuing without.")
                     mixed_precision_enabled = False
        else:
            logging.info("No compatible GPU found by TensorFlow. Using CPU.")
            gpu_available = False

    except tf.errors.NotFoundError as e: # Catch specific TF errors if possible
        logging.error(f"GPU setup failed: TensorFlow error occurred: {e}. Using CPU.", exc_info=False)
        gpu_available = False; mixed_precision_enabled = False
    except RuntimeError as e:
        logging.error(f"GPU setup failed during runtime configuration: {e}. Using CPU.", exc_info=False)
        gpu_available = False; mixed_precision_enabled = False
    except Exception as e:
        logging.error(f"Unexpected error during GPU setup: {e}. Using CPU.", exc_info=True)
        gpu_available = False; mixed_precision_enabled = False

    # Ensure TF uses CPU if GPU setup failed
    if not gpu_available:
        try:
             tf.config.set_visible_devices([], 'GPU')
        except Exception as tf_config_e:
             logging.warning(f"Could not explicitly set visible devices to CPU: {tf_config_e}")

    return gpu_available, mixed_precision_enabled


# --- Caching Setup ---
def setup_joblib_cache(cache_dir):
    """Initializes joblib Memory caching (Native only)."""
    if IS_PYODIDE:
        logging.info("Filesystem caching disabled in Pyodide environment.")
        return None
    if not JOBLIB_AVAILABLE or not Memory:
        logging.warning("joblib library not available/imported. Native filesystem caching DISABLED.")
        return None

    memory_instance = None
    if not cache_dir:
        logging.warning("Native cache directory not specified. Filesystem caching DISABLED.")
        return None

    try:
        # Resolve potential relative paths like '.' or '~'
        resolved_cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
        os.makedirs(resolved_cache_dir, exist_ok=True)
        if not os.access(resolved_cache_dir, os.W_OK):
             raise OSError(f"Cache directory not writable: {resolved_cache_dir}")

        memory_instance = Memory(resolved_cache_dir, verbose=0)
        logging.info(f"Native filesystem caching ENABLED at: {resolved_cache_dir}")

    except OSError as e:
        logging.error(f"Failed cache directory setup for '{cache_dir}': {e}. Caching DISABLED.")
    except Exception as e:
        logging.error(f"Failed native cache initialization for '{cache_dir}': {e}. Caching DISABLED.")

    return memory_instance


# --- Pandoc Check ---
def check_pandoc_exists():
    """Checks for pandoc executable (Native only)."""
    if IS_PYODIDE:
        logging.debug("Pandoc check skipped in Pyodide environment.")
        return False

    try:
        pandoc_path = shutil.which("pandoc")
        if pandoc_path:
            logging.info(f"Pandoc found at: {pandoc_path}")
            return True
        else:
            logging.warning("Pandoc executable not found in system PATH. PDF report generation will be skipped.")
            return False
    except Exception as e:
        logging.error(f"Error checking for pandoc: {e}", exc_info=False)
        return False


# --- Filename Sanitization ---
def safe_filename(input_string, max_length=100):
    """Creates a filesystem-safe filename from an input string."""
    import re
    if not isinstance(input_string, str):
        input_string = str(input_string)
    # Replace non-alphanumeric/hyphen/underscore/dot with underscore
    safe_str = re.sub(r'[^\w\-_\.]', '_', input_string)
    # Remove leading/trailing whitespace, dots, underscores
    safe_str = safe_str.strip('._ ')
    # Replace multiple consecutive underscores with single underscore
    safe_str = re.sub(r'_+', '_', safe_str)
    # Handle empty or dot-only strings
    if not safe_str or all(c == '.' for c in safe_str):
        safe_str = "default_filename"
    # Truncate if necessary, trying to preserve extension
    if len(safe_str) > max_length:
        try:
            name, ext = os.path.splitext(safe_str)
            # Basic check for plausible extension (e.g., .png, .txt, .csv)
            if ext and len(ext) > 1 and len(ext) < 10 and '.' in ext:
                cutoff = max_length - len(ext)
                if cutoff > 0: # Ensure there's space for the name part
                    safe_str = name[:cutoff] + ext
                else: # Not enough space even for extension, just truncate
                     safe_str = safe_str[:max_length]
            else: # No plausible extension, just truncate
                 safe_str = safe_str[:max_length]
        except Exception: # Fallback on any splitting error
             safe_str = safe_str[:max_length]
    # Final check for empty string
    if not safe_str: safe_str = "default_filename"
    return safe_str


# --- JSON Serialization Helper ---
def convert_numpy_for_json(obj):
    """Converts numpy types to standard Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float32, np.float64)):
        if np.isnan(obj): return None
        if np.isinf(obj): return str(obj) # 'Infinity' or '-Infinity'
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist() # Use tolist() for potentially nested arrays
    elif isinstance(obj, pd.Series):
        try: return obj.to_dict()
        except Exception: return {str(k): convert_numpy_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, pd.DataFrame):
        try: return obj.to_dict(orient='records')
        except Exception: return obj.to_string()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.void, bytes)):
        try: return obj.decode('utf-8', errors='replace') # Try decoding bytes
        except Exception: return str(obj) # Fallback to string representation
    elif obj is None:
        return None
    # Raise error for unhandled types to make issues explicit
    # raise TypeError(f"Object of type {type(obj)} with value {obj!r} is not JSON serializable")
    # Or return string representation as a less strict fallback:
    # return str(obj)
    return obj # Let default handler try


# --- Config Import (Example) ---
# If needed within utils itself, import carefully after definitions
# from .config import SOME_CONSTANT