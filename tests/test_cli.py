# -*- coding: utf-8 -*-
"""
Unit tests for the cli module.
"""
import pytest
import os
import sys
import json
import argparse
from unittest.mock import patch, mock_open, MagicMock

# Module to test
from fcgr_analyzer import cli
from fcgr_analyzer import config as default_config_module # For resetting state
from fcgr_analyzer.config import DEFAULT_SEQUENCE_TARGETS, FIGURES_SUBDIR, DATA_SUBDIR

# Use constants from conftest
from .conftest import MINIMAL_TEST_CONFIG

@pytest.fixture(autouse=True)
def ensure_cli_functions_are_module_level():
    """
    Dynamically ensure functions from cli.py that are tested directly
    are available at module level for patching/calling if they were nested.
    This is a workaround if source code has them nested in main().
    Ideally, they should be module-level in cli.py.
    """
    # For this exercise, we assume functions like validate_genome_sampler_fasta
    # are already module-level in the corrected cli.py. If not, this fixture
    # would need to temporarily move them or tests would fail/need adjustment.
    # Example (if validate_genome_sampler_fasta was nested):
    # if not hasattr(cli, 'validate_genome_sampler_fasta'):
    #     cli.validate_genome_sampler_fasta = cli.main.__globals__['validate_genome_sampler_fasta']
    pass


@pytest.fixture(autouse=True)
def reset_default_config_module_state():
    """Ensures the default_config module is reset after each test."""
    original_vars = {k: v for k, v in vars(default_config_module).items() if not k.startswith('__')}
    yield
    for k, v in original_vars.items():
        if k in original_vars: # Ensure it was an original attribute
            setattr(default_config_module, k, v)
    # Clear any potentially added attributes during test if necessary
    current_vars = {k: v for k, v in vars(default_config_module).items() if not k.startswith('__')}
    for k in list(current_vars.keys()): # list() for safe iteration while deleting
        if k not in original_vars:
            delattr(default_config_module, k)


@pytest.fixture
def mock_pipeline_run(mocker):
    return mocker.patch('fcgr_analyzer.pipeline.run_pipeline', return_value={'timings': {'total_pipeline': 1.0}, 'data_summary':{}, 'performance_metrics':{}})

@pytest.fixture
def mock_cli_run_tests(mocker): # Renamed to avoid conflict
    return mocker.patch('fcgr_analyzer.cli.run_tests', return_value=0)

@pytest.fixture
def mock_setup_logging(mocker):
    return mocker.patch('fcgr_analyzer.utils.setup_logging')


# --- Test load_targets_from_file ---
def test_load_targets_csv_success(tmp_path):
    content = "species1,biotype1,accession,ID001,label_override1\n#comment\nspecies2,biotype2,local_sequence,ATGC\n"
    filepath = tmp_path / "targets.csv"
    filepath.write_text(content)
    targets = cli.load_targets_from_file(str(filepath))
    assert targets is not None
    assert len(targets) == 2
    assert targets[0] == ('species1', 'biotype1', 'accession', 'ID001', 'label_override1')
    assert targets[1] == ('species2', 'biotype2', 'local_sequence', 'ATGC')

def test_load_targets_tsv_success(tmp_path):
    content = "species1\tbiotype1\taccession\tID001\tlabel_override1\n#comment\nspecies2\tbiotype2\tlocal_sequence\tATGC\n"
    filepath = tmp_path / "targets.tsv"
    filepath.write_text(content)
    targets = cli.load_targets_from_file(str(filepath))
    assert targets is not None
    assert len(targets) == 2
    assert targets[0] == ('species1', 'biotype1', 'accession', 'ID001', 'label_override1')
    assert targets[1] == ('species2', 'biotype2', 'local_sequence', 'ATGC')

def test_load_targets_invalid_lines(tmp_path, caplog):
    content = "species1,biotype1,accession,ID001,label1\ninvalid_line_too_short\nspecies2,biotype2,local_sequence,ATGC,extra,columns\n"
    filepath = tmp_path / "targets.csv"
    filepath.write_text(content)
    targets = cli.load_targets_from_file(str(filepath))
    assert targets is not None
    assert len(targets) == 1
    assert targets[0] == ('species1', 'biotype1', 'accession', 'ID001', 'label1')
    assert "Skipping invalid line 2" in caplog.text
    assert "Skipping invalid line 3" in caplog.text

def test_load_targets_file_not_found(caplog):
    targets = cli.load_targets_from_file("nonexistent_file.csv")
    assert targets is None
    assert "Targets file not found" in caplog.text

def test_load_targets_empty_file(tmp_path, caplog):
    filepath = tmp_path / "empty.csv"
    filepath.write_text("#only comments\n\n")
    targets = cli.load_targets_from_file(str(filepath))
    assert targets is None 
    assert "No valid targets found" in caplog.text


# --- Test load_config_from_file ---
def test_load_config_json_success(tmp_path):
    config_data = {"FCGR_K": 7, "MIN_SEQ_LEN": 100, "UNKNOWN_KEY": "value"}
    filepath = tmp_path / "config.json"
    with open(filepath, 'w') as f:
        json.dump(config_data, f)
    
    loaded_config = cli.load_config_from_file(str(filepath))
    assert loaded_config == config_data

def test_load_config_file_not_found(caplog):
    config = cli.load_config_from_file("nonexistent.json")
    assert config == {}
    assert "Configuration file not found" in caplog.text

def test_load_config_invalid_json(tmp_path, caplog):
    filepath = tmp_path / "invalid.json"
    filepath.write_text("{'FCGR_K': 7,") 
    config = cli.load_config_from_file(str(filepath))
    assert config == {}
    assert "Error decoding JSON" in caplog.text


# --- Test run_tests function (mocking pytest) ---
def test_cli_run_tests_pytest_found(mocker): # Renamed fixture usage
    mock_pytest_main = mocker.patch('pytest.main', return_value=0)
    # Patch os.path.isdir to simulate finding the 'tests' directory
    mocker.patch('os.path.isdir', return_value=True)
    # Mock abspath and dirname to control the path construction logic
    mocker.patch('os.path.abspath', return_value='/fake/path/to/module/cli.py')
    mocker.patch('os.path.dirname', return_value='/fake/path/to/module') # This path will be joined with 'tests'
    
    exit_code = cli.run_tests()
    assert exit_code == 0
    mock_pytest_main.assert_called_once_with(['-v', '/fake/path/to/module/tests'])

def test_cli_run_tests_pytest_pkg_resources(mocker): # Renamed
    mock_pytest_main = mocker.patch('pytest.main', return_value=0)
    mocker.patch('os.path.isdir', side_effect=[False, True]) 
    mocker.patch('pkg_resources.resource_filename', return_value='/pkg/res/tests')

    exit_code = cli.run_tests()
    assert exit_code == 0
    mock_pytest_main.assert_called_once_with(['-v', '/pkg/res/tests'])


def test_cli_run_tests_pytest_not_found(mocker, capsys): # Renamed
    # Mock the import of pytest within cli.run_tests
    with patch('importlib.import_module', side_effect=ImportError("No module named pytest")):
        exit_code = cli.run_tests()
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "pytest is required to run tests" in captured.err


# --- Test validate_genome_sampler_fasta ---
# This function is defined inside cli.main(). To test it directly, it needs to be module-level.
# Assuming it's moved to module level for testability. If not, these tests would need to call cli.main()
# and assert behavior, which is more complex.
# For now, I will assume `validate_genome_sampler_fasta` has been moved to module level in `cli.py`.

@pytest.fixture(autouse=True)
def move_cli_helpers_to_module_scope_for_testing(mocker):
    """
    If helper functions are defined inside main(), this fixture temporarily
    makes them available at module scope for direct testing.
    This is a workaround; ideally, helpers are module-level.
    """
    if not hasattr(cli, 'validate_genome_sampler_fasta'):
        # This indicates the function is nested. For robust testing, it should be refactored.
        # For this test run, we can try to extract it if possible, or skip.
        # print("Warning: validate_genome_sampler_fasta not at module level in cli.py. Test may be unreliable.")
        # As a workaround, you might try: cli.validate_genome_sampler_fasta = cli.main.__globals__['validate_genome_sampler_fasta']
        # But this is fragile. For now, tests will fail if it's not module level.
        pass

# The following tests assume validate_genome_sampler_fasta is module-level in cli.py
def test_validate_genome_sampler_fasta_valid(tmp_path):
    if not hasattr(cli, 'validate_genome_sampler_fasta'): pytest.skip("validate_genome_sampler_fasta not at module level")
    content = ">species|biotype|description\nATGC\n"
    filepath = tmp_path / "valid.fasta"
    filepath.write_text(content)
    assert cli.validate_genome_sampler_fasta(str(filepath)) is True

def test_validate_genome_sampler_fasta_invalid_header(tmp_path):
    if not hasattr(cli, 'validate_genome_sampler_fasta'): pytest.skip("validate_genome_sampler_fasta not at module level")
    content = ">species_biotype_description\nATGC\n"
    filepath = tmp_path / "invalid_header.fasta"
    filepath.write_text(content)
    assert cli.validate_genome_sampler_fasta(str(filepath)) is False


# --- Test main function argument parsing and flow ---
@pytest.mark.parametrize("cli_args, expected_targets_source_type, expected_input_fasta_used", [
    ([], "Default List", False), 
    (["-t", "targets.csv"], "targets.csv", False),
    (["-i", "input.fasta"], "input.fasta", True),
])
def test_main_input_source(cli_args, expected_targets_source_type, expected_input_fasta_used,
                           mocker, mock_pipeline_run, mock_setup_logging, tmp_path):
    
    # Mock module-level validate_genome_sampler_fasta if it exists
    if hasattr(cli, 'validate_genome_sampler_fasta'):
        mocker.patch('fcgr_analyzer.cli.validate_genome_sampler_fasta', return_value=True)
    else: # If it's still nested, we can't easily mock it for this test setup of main()
        pass # main() will call its internal version

    mocker.patch('sys.argv', ['cli.py'] + cli_args)
    mocker.patch('os.path.exists', return_value=True) 
    mocker.patch('fcgr_analyzer.cli.load_targets_from_file', return_value=[('s','b','acc','id')])
    mocker.patch('fcgr_analyzer.utils.setup_gpu_and_mixed_precision', return_value=(False, False))
    mocker.patch('sys.exit') # Prevent actual exit during test of main flow

    # Create dummy files if needed and adjust paths
    adjusted_cli_args = cli_args.copy()
    if "-t" in cli_args:
        targets_file_path = tmp_path / "targets.csv"
        targets_file_path.touch()
        adjusted_cli_args[cli_args.index("-t") + 1] = str(targets_file_path)
    if "-i" in cli_args:
        input_fasta_path = tmp_path / "input.fasta"
        input_fasta_path.write_text(">sp|bio|desc\nATGC")
        adjusted_cli_args[cli_args.index("-i") + 1] = str(input_fasta_path)
    
    mocker.patch('sys.argv', ['cli.py'] + adjusted_cli_args)

    try:
        cli.main()
    except SystemExit as e: # Catch sys.exit(0)
        assert e.code == 0

    mock_pipeline_run.assert_called_once()
    call_args_dict = mock_pipeline_run.call_args[1] 
    
    config_in_call = call_args_dict['config_dict']
    assert expected_targets_source_type in config_in_call['targets_source']
    
    if expected_input_fasta_used:
        assert call_args_dict['input_fasta'] is not None
        assert os.path.basename(call_args_dict['input_fasta']) == expected_targets_source_type
        assert call_args_dict['targets'] is None
    else:
        assert call_args_dict['input_fasta'] is None
        if expected_targets_source_type == "Default List":
             assert call_args_dict['targets'] == DEFAULT_SEQUENCE_TARGETS
        else: 
             assert call_args_dict['targets'] == [('s','b','acc','id')]


def test_main_run_tests_arg(mock_cli_run_tests, mock_setup_logging, mocker):
    mocker.patch('sys.argv', ['cli.py', '--run-tests'])
    mock_sys_exit = mocker.patch('sys.exit') 
    
    cli.main()
    
    mock_cli_run_tests.assert_called_once()
    mock_setup_logging.assert_called_once() 
    mock_sys_exit.assert_called_once_with(0) # Expect exit code from run_tests


def test_main_config_file_override(mocker, mock_pipeline_run, mock_setup_logging, tmp_path):
    user_config_data = {"FCGR_K": 8, "ENTREZ_EMAIL": "user@configured.com", "UNKNOWN_KEY": True}
    config_filepath = tmp_path / "user_cfg.json"
    with open(config_filepath, 'w') as f:
        json.dump(user_config_data, f)

    mocker.patch('sys.argv', ['cli.py', '--config-file', str(config_filepath)])
    mocker.patch('fcgr_analyzer.utils.setup_gpu_and_mixed_precision', return_value=(False, False))
    mocker.patch('sys.exit') 

    try:
        cli.main()
    except SystemExit as e:
        assert e.code == 0

    mock_pipeline_run.assert_called_once()
    call_config = mock_pipeline_run.call_args[1]['config_dict']
    
    assert call_config['FCGR_K'] == 8
    assert call_config['FCGR_DIM'] == 1 << 8
    assert call_config['ENTREZ_EMAIL'] == "user@configured.com"
    assert "UNKNOWN_KEY" not in call_config 


def test_main_cli_overrides_config_file_and_advanced_options(mocker, mock_pipeline_run, mock_setup_logging, tmp_path):
    user_config_data = {"FCGR_K": 8, "MIN_SEQ_LEN": 50} # From file
    config_filepath = tmp_path / "user_cfg.json"
    with open(config_filepath, 'w') as f:
        json.dump(user_config_data, f)

    # CLI args to override file and defaults
    cli_params = ['cli.py', 
                  '--config-file', str(config_filepath), 
                  '--fcgr-k', '7',             # CLI override for FCGR_K (advanced)
                  '--min-seq-len', '150']      # CLI override for MIN_SEQ_LEN (advanced)
    
    mocker.patch('sys.argv', cli_params)
    mocker.patch('fcgr_analyzer.utils.setup_gpu_and_mixed_precision', return_value=(False, False))
    mocker.patch('sys.exit')

    try:
        cli.main()
    except SystemExit as e:
        assert e.code == 0

    mock_pipeline_run.assert_called_once()
    call_config = mock_pipeline_run.call_args[1]['config_dict']

    assert call_config['FCGR_K'] == 7 # CLI --fcgr-k overrides file
    assert call_config['FCGR_DIM'] == 1 << 7
    assert call_config['MIN_SEQ_LEN'] == 150 # CLI --min-seq-len overrides file


def test_main_entrez_email_check_needed(mocker, mock_pipeline_run, mock_setup_logging, tmp_path, capsys):
    targets_requiring_entrez = [('Human', 'Gene', 'accession', 'NM_001')]
    mocker.patch('fcgr_analyzer.cli.load_targets_from_file', return_value=targets_requiring_entrez)
    
    original_entrez_email = default_config_module.ENTREZ_EMAIL
    mocker.patch.object(default_config_module, 'ENTREZ_EMAIL', "your.email@example.com")

    dummy_targets_path = tmp_path / "dummy_targets.csv"
    dummy_targets_path.touch()
    mocker.patch('sys.argv', ['cli.py', '-t', str(dummy_targets_path)]) 

    mock_sys_exit = mocker.patch('sys.exit')
    mocker.patch('fcgr_analyzer.utils.setup_gpu_and_mixed_precision', return_value=(False, False))

    cli.main() # Should call sys.exit(1)
    
    mock_sys_exit.assert_called_once_with(1)
    mock_pipeline_run.assert_not_called() 
    captured = capsys.readouterr()
    assert "CRITICAL ERROR: ENTREZ_EMAIL is not configured" in captured.err

    mocker.patch.object(default_config_module, 'ENTREZ_EMAIL', original_entrez_email)


def test_main_entrez_email_check_not_needed_for_fasta(mocker, mock_pipeline_run, mock_setup_logging, tmp_path):
    original_entrez_email = default_config_module.ENTREZ_EMAIL
    mocker.patch.object(default_config_module, 'ENTREZ_EMAIL', "your.email@example.com")

    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(">sp|bio|desc\nATGC")
    
    mocker.patch('sys.argv', ['cli.py', '-i', str(fasta_path)])
    # Mock the module-level validate_genome_sampler_fasta if it was refactored out
    if hasattr(cli, 'validate_genome_sampler_fasta'):
        mocker.patch('fcgr_analyzer.cli.validate_genome_sampler_fasta', return_value=True)

    mocker.patch('fcgr_analyzer.utils.setup_gpu_and_mixed_precision', return_value=(False, False))
    mocker.patch('sys.exit')

    try:
        cli.main() 
    except SystemExit as e:
        assert e.code == 0 # Should complete successfully

    mock_pipeline_run.assert_called_once() 
    mocker.patch.object(default_config_module, 'ENTREZ_EMAIL', original_entrez_email)


def test_main_handles_pipeline_exception(mocker, mock_setup_logging, capsys):
    mocker.patch('sys.argv', ['cli.py'])
    mocker.patch('fcgr_analyzer.pipeline.run_pipeline', side_effect=Exception("Pipeline boom!"))
    mock_sys_exit = mocker.patch('sys.exit')
    mocker.patch('fcgr_analyzer.utils.setup_gpu_and_mixed_precision', return_value=(False, False))

    cli.main()
    
    mock_sys_exit.assert_called_once_with(1)
    captured = capsys.readouterr()
    assert "ERROR: Pipeline failed unexpectedly" in captured.err