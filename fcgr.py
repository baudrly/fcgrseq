# -*- coding: utf-8 -*-
"""
Frequency Chaos Game Representation (FCGR) Generation.
Includes optimized Numba implementation (native only) and Python fallback.
"""
import numpy as np
import logging
from .config import FCGR_K, FCGR_DIM, EPSILON
from .utils import NUMBA_AVAILABLE # Use the flag from utils

# --- Conditional Numba compilation ---
_fcgr_loop_numba = None
if NUMBA_AVAILABLE:
    try:
        import numba # Import numba here, guarded by flag
        @numba.jit(nopython=True, fastmath=True, cache=True)
        def _fcgr_loop_numba_compiled(sequence_bytes: np.ndarray, k: int, dim: int, mask: int):
            """Numba-optimized loop to calculate FCGR counts."""
            fcgr_counts = np.zeros((dim, dim), dtype=np.float32)
            x_coord, y_coord = 0, 0
            valid_kmer_count = 0
            current_kmer_len = 0
            A_BYTE, T_BYTE, C_BYTE, G_BYTE = 65, 84, 67, 71 # b'A', b'T', b'C', b'G'

            for i in range(len(sequence_bytes)):
                base_byte = sequence_bytes[i]
                val = -1
                xb = 0
                yb = 0
                if base_byte == A_BYTE:   # A (00)
                    val = 0; xb = 0; yb = 0
                elif base_byte == T_BYTE: # T (01)
                    val = 1; xb = 1; yb = 0
                elif base_byte == C_BYTE: # C (10)
                    val = 2; xb = 0; yb = 1
                elif base_byte == G_BYTE: # G (11)
                    val = 3; xb = 1; yb = 1
                # else: val remains -1 for invalid chars

                if val == -1: # Reset on invalid char
                    x_coord, y_coord = 0, 0
                    current_kmer_len = 0
                    continue

                # Update coordinates
                x_coord = ((x_coord << 1) | xb) & mask
                y_coord = ((y_coord << 1) | yb) & mask
                current_kmer_len += 1

                # Increment count if a full k-mer is formed
                if current_kmer_len >= k:
                    # Use y_coord for row, x_coord for column
                    fcgr_counts[y_coord, x_coord] += 1.0
                    valid_kmer_count += 1

            return fcgr_counts, valid_kmer_count
        _fcgr_loop_numba = _fcgr_loop_numba_compiled # Assign compiled function
        logging.debug("Numba FCGR function compiled successfully.")
    except Exception as e:
         logging.error(f"Numba FCGR compilation failed: {e}. Numba FCGR disabled.", exc_info=False)
         _fcgr_loop_numba = None # Ensure it's None on failure


# --- Python Fallback Implementation ---
def generate_fcgr_python(sequence: str, k: int) -> np.ndarray:
    """
    Generates the FCGR matrix using pure Python. Fallback implementation.
    """
    dim = 1 << k
    mask = dim - 1
    # Initialize with float32 which is often sufficient and saves memory
    fcgr = np.zeros((dim, dim), dtype=np.float32)
    x, y = 0, 0          # CGR coordinates
    vk = 0               # Valid k-mer count
    ck = 0               # Current k-mer length
    base_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    get_val = base_map.get # Micro-optimization for lookup

    for base in sequence:
        val = get_val(base, -1) # Returns -1 if base is not A, T, C, G

        if val == -1: # Reset on invalid base
            x = 0
            y = 0
            ck = 0
            continue

        # Extract bits for coordinate updates
        xb = val & 1
        yb = val >> 1

        # Update coordinates
        x = ((x << 1) | xb) & mask
        y = ((y << 1) | yb) & mask
        ck += 1

        # Increment count if a full k-mer is formed
        if ck >= k:
            fcgr[y, x] += 1.0 # Increment count (y=row, x=col)
            vk += 1

    # Normalize the counts to get frequencies
    if vk > 0:
        fcgr /= vk
    else:
        # Only warn if sequence was long enough to contain k-mers
        if len(sequence) >= k:
            logging.warning(f"No valid {k}-mers found in sequence (len={len(sequence)}) for Python FCGR.")

    return fcgr.astype(np.float32) # Ensure correct dtype


# --- Entry Point Function ---
def generate_fcgr(sequence: str, k: int) -> np.ndarray:
    """
    Generates the FCGR matrix for a given sequence and k-mer size.
    Uses Numba implementation if available and successfully compiled, otherwise falls back to Python.

    Args:
        sequence: DNA sequence string (expects cleaned sequence with only ATGC).
        k: The k-mer size.

    Returns:
        A 2^k x 2^k numpy array (float32) representing the normalized FCGR.
        Returns a zero matrix if the sequence has no valid k-mers or on error.
    """
    dim = 1 << k
    fcgr_matrix = np.zeros((dim, dim), dtype=np.float32) # Default return value

    # Check sequence length early
    if len(sequence) < k:
        # logging.debug(f"Sequence length {len(sequence)} is less than k={k}. Returning zero matrix.")
        return fcgr_matrix

    # --- Select implementation ---
    use_numba_logic = _fcgr_loop_numba is not None

    if use_numba_logic:
        try:
            # Encode sequence to ASCII bytes for Numba.
            sequence_bytes = sequence.encode('ascii')
            # Pass as numpy array for potential Numba optimization
            sequence_np_bytes = np.frombuffer(sequence_bytes, dtype=np.uint8)

            # logging.debug(f"Attempting Numba FCGR k={k}...")
            fcgr_counts, valid_kmer_count = _fcgr_loop_numba(sequence_np_bytes, k, dim, (dim - 1)) # Pass mask

            if valid_kmer_count > 0:
                # Perform division carefully, ensuring float division
                fcgr_matrix = (fcgr_counts / float(valid_kmer_count)).astype(np.float32)
                # logging.debug(f"Numba FCGR successful ({valid_kmer_count} k-mers).")
                return fcgr_matrix
            else:
                # No need to warn again if len < k, handled above
                logging.warning(f"Numba FCGR (k={k}): No valid k-mers found in sequence (len={len(sequence)}).")
                return fcgr_matrix # Return zeros
        except UnicodeEncodeError:
            logging.error(f"Numba FCGR skipped: Sequence contains non-ASCII characters. Ensure sequence is cleaned first. Sequence start: {sequence[:50]}... Falling back to Python.")
            # Fall through to Python implementation below
        except Exception as e:
            logging.error(f"Numba FCGR execution failed unexpectedly: {e}. Falling back to Python.", exc_info=False)
            # Fall through to Python implementation below
    # else:
         # logging.debug("Using Python FCGR implementation.")

    # --- Python Fallback Implementation ---
    try:
        fcgr_matrix = generate_fcgr_python(sequence, k)
    except Exception as e:
         logging.error(f"Python FCGR calculation failed unexpectedly: {e}. Returning zero matrix.", exc_info=True)
         fcgr_matrix = np.zeros((dim, dim), dtype=np.float32) # Ensure return zeros on error

    return fcgr_matrix.astype(np.float32) # Final dtype check