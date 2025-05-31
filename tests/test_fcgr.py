# -*- coding: utf-8 -*-
"""
Unit tests for the fcgr module.
Tests both Numba and Python implementations.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

# Module containing functions to test
from fcgr_analyzer import fcgr as fcg
from fcgr_analyzer.utils import NUMBA_AVAILABLE
from fcgr_analyzer.config import FCGR_K as CFG_FCGR_K # Use configured K

# Test sequences from conftest
from .conftest import TEST_SEQ_VALID_MIN_LEN, TEST_SEQ_INVALID_CHARS, TEST_SEQ_EMPTY, TEST_SEQ_NON_ASCII

# --- Helper for expected coordinates ---
def get_cgr_coords_manual(kmer):
    """Calculates theoretical CGR coordinates (x, y) for a k-mer (manual for clarity)."""
    x_coord, y_coord = 0, 0
    base_map = {'A': (0,0), 'T': (1,0), 'C': (0,1), 'G': (1,1)} # (x_bit, y_bit)
    for base in kmer:
        xb, yb = base_map[base]
        x_coord = (x_coord << 1) | xb
        y_coord = (y_coord << 1) | yb
    return x_coord, y_coord


# --- Test Python Implementation (generate_fcgr_python) ---
@pytest.mark.parametrize("k_val, seq_str, expected_kmers_counts", [
    (2, "ATGCATGC", {("AT",0,0):2, ("TG",2,0):2, ("GC",2,1):2, ("CA",0,1):1}), # AT(x0,y0), TG(x2,y0), GC(x2,y1), CA(x0,y1)
    (3, "AAAA", {("AAA",0,0):2}), # A=(0,0)
    (3, "ATATAT", {("ATA",0,0):2, ("TAT",2,0):2}), # A=(0,0) T=(1,0) -> ATA (0,0), TAT(x=2,y=0)
    (1, "ACGT", {("A",0,0):1, ("C",0,1):1, ("G",1,1):1, ("T",1,0):1})
])
def test_generate_fcgr_python_basic_known(k_val, seq_str, expected_kmers_counts):
    dim = 1 << k_val
    fcgr_py = fcg.generate_fcgr_python(seq_str, k_val)

    assert fcgr_py.shape == (dim, dim)
    assert fcgr_py.dtype == np.float32
    if sum(expected_kmers_counts.values()) > 0:
        assert_allclose(np.sum(fcgr_py), 1.0, atol=1e-6)
    else:
        assert np.sum(fcgr_py) == 0.0

    total_valid_kmers = sum(c for k,c in expected_kmers_counts.items() if c > 0)
    
    for kmer_tuple, count in expected_kmers_counts.items():
        kmer_str, expected_x, expected_y = kmer_tuple
        actual_x, actual_y = get_cgr_coords_manual(kmer_str) # Verify helper
        assert actual_x == expected_x
        assert actual_y == expected_y
        
        if total_valid_kmers > 0:
            assert_allclose(fcgr_py[actual_y, actual_x], count / total_valid_kmers, atol=1e-7)
        else:
            assert_allclose(fcgr_py[actual_y, actual_x], 0.0, atol=1e-7)


def test_generate_fcgr_python_invalid_chars():
    k = 3
    dim = 1 << k
    # N resets, X resets. Expected: ATG (1), GCA(1), CAT(1), ATG(1) ... total 4 from ATGCATGCAT
    # TTT (1) from TTTX. Total 5 valid k-mers.
    seq = "ATGNGCATGCATXTTT"
    fcgr_py = fcg.generate_fcgr_python(seq, k)
    assert fcgr_py.shape == (dim, dim)
    assert_allclose(np.sum(fcgr_py), 1.0, atol=1e-6) # Should still sum to 1

    # Check specific k-mers
    x_atg, y_atg = get_cgr_coords_manual("ATG")
    x_gca, y_gca = get_cgr_coords_manual("GCA")
    x_cat, y_cat = get_cgr_coords_manual("CAT")
    x_ttt, y_ttt = get_cgr_coords_manual("TTT")
    
    # Expected counts: ATG (2/9), GCA(2/9), CAT(2/9), TTT(1/9)
    # Valid k-mers: ATG, GCA, CAT, GCA, CAT, ATG, CAT, ATG, TTT = 9
    # ATG: 3, GCA: 2, CAT: 3, TTT: 1
    assert_allclose(fcgr_py[y_atg, x_atg], 3/9)
    assert_allclose(fcgr_py[y_gca, x_gca], 2/9)
    assert_allclose(fcgr_py[y_cat, x_cat], 3/9)
    assert_allclose(fcgr_py[y_ttt, x_ttt], 1/9)


@pytest.mark.parametrize("seq_str, k_val", [
    (TEST_SEQ_EMPTY, 3),
    ("AT", 3), # Shorter than k
])
def test_generate_fcgr_python_empty_or_short(seq_str, k_val):
    dim = 1 << k_val
    fcgr_py = fcg.generate_fcgr_python(seq_str, k_val)
    assert fcgr_py.shape == (dim, dim)
    assert_array_equal(fcgr_py, np.zeros((dim, dim), dtype=np.float32))
    assert np.sum(fcgr_py) == 0.0


# --- Test Numba Implementation (_fcgr_loop_numba_compiled) ---
@pytest.mark.skipif(not NUMBA_AVAILABLE or fcg._fcgr_loop_numba is None, reason="Numba not available or Numba FCGR not compiled")
def test_fcgr_loop_numba_basic():
    k = 2
    dim = 1 << k
    mask = dim - 1
    seq_str = "ATGC"
    seq_bytes = np.frombuffer(seq_str.encode('ascii'), dtype=np.uint8)
    
    counts_nb, valid_kmers_nb = fcg._fcgr_loop_numba(seq_bytes, k, dim, mask)
    
    assert counts_nb.shape == (dim, dim)
    assert valid_kmers_nb == 3 # AT, TG, GC
    
    # AT: y=0, x=0
    # TG: y=0, x=2 (T=(1,0), G=(1,1) -> y=01, x=01 -> y=1, x=1 by manual; Numba: T(01) G(11) -> y=01, x=11 -> y=1, x=3 if bits are (yb, xb))
    # Numba internal bits: A (00->xb=0,yb=0), T (01->xb=1,yb=0), C (10->xb=0,yb=1), G (11->xb=1,yb=1)
    # AT: A(0,0), T(1,0) -> y_coord = (0<<1)|0 = 0, x_coord = (0<<1)|1 = 1. Count for [0,1] for kmer AT
    # TG: T(1,0), G(1,1) -> y_coord = (0<<1)|1 = 1, x_coord = (1<<1)|1 = 3. Count for [1,3] for kmer TG
    # GC: G(1,1), C(0,1) -> y_coord = (1<<1)|1 = 3, x_coord = (1<<1)|0 = 2. Count for [3,2] for kmer GC
    assert counts_nb[0, 1] == 1.0 # AT
    assert counts_nb[1, 3] == 1.0 # TG
    assert counts_nb[3, 2] == 1.0 # GC
    assert np.sum(counts_nb) == 3.0


@pytest.mark.skipif(not NUMBA_AVAILABLE or fcg._fcgr_loop_numba is None, reason="Numba not available or Numba FCGR not compiled")
def test_fcgr_loop_numba_invalid_chars_reset():
    k = 2
    dim = 1 << k
    mask = dim - 1
    seq_str = "ATNGA" # AT, then reset, then GA
    seq_bytes = np.frombuffer(seq_str.encode('ascii'), dtype=np.uint8)
    
    counts_nb, valid_kmers_nb = fcg._fcgr_loop_numba(seq_bytes, k, dim, mask)
    assert valid_kmers_nb == 2
    assert counts_nb[0, 1] == 1.0 # AT
    assert counts_nb[1, 0] == 1.0 # GA: G(1,1), A(0,0) -> y=(1<<1)|0 = 2, x=(1<<1)|0 = 2
    assert np.sum(counts_nb) == 2.0


# --- Test Main Entry Point (generate_fcgr) ---
# This implicitly tests fallback logic if Numba part is complex to mock perfectly for failure modes

def test_generate_fcgr_default_k_matches_python(mocker):
    # Test with default k from config, ensuring it matches pure Python for a clean sequence
    k = CFG_FCGR_K
    seq = TEST_SEQ_VALID_MIN_LEN[:50] # Use a shorter part for faster test

    # Force Python if Numba is available, to get a reference
    mocker.patch('fcgr_analyzer.fcgr._fcgr_loop_numba', None) # Simulate Numba not compiled
    fcgr_ref_py = fcg.generate_fcgr(seq, k)
    
    # Now, let Numba run if it's available (remove the patch for _fcgr_loop_numba)
    mocker.stopall() # Resets all mocks for this test
    if NUMBA_AVAILABLE and fcg._fcgr_loop_numba is not None: # If Numba truly available and compiled
        fcgr_actual = fcg.generate_fcgr(seq, k)
        assert_allclose(fcgr_actual, fcgr_ref_py, atol=1e-7, rtol=0,
                        err_msg=f"FCGR from main entry (Numba if available) differs from pure Python for k={k}")
    else: # If Numba not available, generate_fcgr directly uses Python
        fcgr_actual = fcg.generate_fcgr(seq, k)
        assert_allclose(fcgr_actual, fcgr_ref_py, atol=1e-7, rtol=0,
                        err_msg=f"FCGR from main entry (Python) differs from direct Python call for k={k}")


def test_generate_fcgr_numba_fails_fallback_to_python(mocker):
    # This test only makes sense if Numba is initially considered available
    if not NUMBA_AVAILABLE or fcg._fcgr_loop_numba is None:
        pytest.skip("Numba not available/compiled, cannot test Numba failure fallback.")

    k = 3
    seq = "ATGCATGC"
    
    # Mock the Numba compiled function to raise an error
    mocker.patch('fcgr_analyzer.fcgr._fcgr_loop_numba', side_effect=RuntimeError("Simulated Numba Failure"))
    # Spy on the Python implementation
    spy_generate_fcgr_python = mocker.spy(fcg, 'generate_fcgr_python')

    fcgr_result = fcg.generate_fcgr(seq, k)

    # Check that Python implementation was called after Numba failed
    spy_generate_fcgr_python.assert_called_once_with(seq, k)
    
    # Check that the result is a valid FCGR (produced by Python)
    dim = 1 << k
    assert fcgr_result.shape == (dim, dim)
    assert_allclose(np.sum(fcgr_result), 1.0, atol=1e-6)


def test_generate_fcgr_non_ascii_input_fallback(mocker, caplog):
    # This test only makes sense if Numba is initially considered available
    if not NUMBA_AVAILABLE or fcg._fcgr_loop_numba is None:
        pytest.skip("Numba not available/compiled, cannot test non-ASCII fallback.")

    k = 3
    seq_non_ascii = TEST_SEQ_NON_ASCII # "ATGC你好"
    
    # Spy on Python implementation
    spy_generate_fcgr_python = mocker.spy(fcg, 'generate_fcgr_python')

    fcgr_result = fcg.generate_fcgr(seq_non_ascii, k)
    
    # Numba version (if attempted) would raise UnicodeEncodeError.
    # The main function should catch this and fall back to Python.
    assert "Numba FCGR skipped: Sequence contains non-ASCII characters" in caplog.text
    spy_generate_fcgr_python.assert_called_once_with(seq_non_ascii, k)
    
    # The Python version will process "ATGC" from "ATGC你好"
    fcgr_expected_python = fcg.generate_fcgr_python("ATGC", k)
    assert_allclose(fcgr_result, fcgr_expected_python, atol=1e-7)


@pytest.mark.parametrize("seq, k_val", [
    (TEST_SEQ_EMPTY, CFG_FCGR_K),
    ("AG", CFG_FCGR_K), # Too short for default k
])
def test_generate_fcgr_empty_or_too_short_main(seq, k_val):
    dim = 1 << k_val
    fcgr_main = fcg.generate_fcgr(seq, k_val)
    assert fcgr_main.shape == (dim, dim)
    assert_array_equal(fcgr_main, np.zeros((dim, dim), dtype=np.float32))
    assert np.sum(fcgr_main) == 0.0
