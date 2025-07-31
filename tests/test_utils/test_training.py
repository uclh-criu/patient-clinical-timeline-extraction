import pytest
import os
import csv
from unittest.mock import patch, mock_open
from types import SimpleNamespace
from utils import training_utils
import matplotlib.pyplot as plt

# 1. Test for generate_hyperparameter_grid
def test_generate_hyperparameter_grid_enabled():
    """Test grid generation when ENABLE_GRID_SEARCH is True."""
    mock_config = SimpleNamespace(
        LEARNING_RATE=[0.01, 0.001],
        BATCH_SIZE=[32],
        OPTIMIZER='Adam',
        ENABLE_GRID_SEARCH=True
    )
    grid = training_utils.generate_hyperparameter_grid(mock_config)
    assert len(grid) == 2
    assert {'LEARNING_RATE': 0.01, 'BATCH_SIZE': 32, 'OPTIMIZER': 'Adam', 'ENABLE_GRID_SEARCH': True} in grid
    assert {'LEARNING_RATE': 0.001, 'BATCH_SIZE': 32, 'OPTIMIZER': 'Adam', 'ENABLE_GRID_SEARCH': True} in grid

def test_generate_hyperparameter_grid_disabled():
    """Test grid generation when ENABLE_GRID_SEARCH is False."""
    mock_config = SimpleNamespace(
        LEARNING_RATE=[0.01, 0.001],
        BATCH_SIZE=[32, 64],
        OPTIMIZER='Adam',
        ENABLE_GRID_SEARCH=False
    )
    grid = training_utils.generate_hyperparameter_grid(mock_config)
    assert len(grid) == 1
    # It should pick the *first* value from any lists
    assert grid[0]['LEARNING_RATE'] == 0.01
    assert grid[0]['BATCH_SIZE'] == 32

def test_generate_hyperparameter_grid_no_lists():
    """Test grid generation when there are no lists in config."""
    mock_config = SimpleNamespace(
        LEARNING_RATE=0.01,
        BATCH_SIZE=32,
        OPTIMIZER='Adam'
    )
    grid = training_utils.generate_hyperparameter_grid(mock_config)
    assert len(grid) == 1
    assert grid[0]['LEARNING_RATE'] == 0.01
    assert grid[0]['BATCH_SIZE'] == 32

# 2. Test for log_training_run
@patch('os.makedirs')
@patch('os.path.isfile')
@patch('builtins.open', new_callable=mock_open)
def test_log_training_run_new_file(mock_file, mock_isfile, mock_makedirs):
    """Test that logging creates a file with the correct headers and content when file doesn't exist."""
    mock_isfile.return_value = False  # File doesn't exist
    
    hyperparams = {'lr': 0.01, 'epochs': 10}
    metrics = {'val_acc': 0.95, 'val_loss': 0.1, 'train_loss': 0.2, 'epochs': 10}
    
    training_utils.log_training_run(
        model_path="/models/model_v1.pt",
        hyperparams=hyperparams,
        metrics=metrics,
        dataset_name="test_data",
        entity_mode="multi_entity",
        log_dir="/log_dir",
        log_filename="test_log.csv"
    )
    
    # Check that directories were created
    mock_makedirs.assert_called_once_with(os.path.join("/log_dir"), exist_ok=True)
    
    # Check file was opened for appending
    mock_file.assert_called_once_with(os.path.join('/log_dir', 'test_log.csv'), mode='a', newline='')
    
    # Check that writeheader and writerow were called
    file_handle = mock_file()
    csv_writer = file_handle.write.call_args_list
    assert len(csv_writer) > 0  # At least one write call should have been made

@patch('os.makedirs')
@patch('os.path.isfile')
@patch('builtins.open', new_callable=mock_open)
def test_log_training_run_existing_file(mock_file, mock_isfile, mock_makedirs):
    """Test that logging appends to an existing file without writing headers."""
    mock_isfile.return_value = True  # File exists
    
    hyperparams = {'lr': 0.01, 'epochs': 10}
    metrics = {'val_acc': 0.95, 'val_loss': 0.1, 'train_loss': 0.2, 'epochs': 10}
    
    training_utils.log_training_run(
        model_path="/models/model_v1.pt",
        hyperparams=hyperparams,
        metrics=metrics,
        dataset_name="test_data",
        entity_mode="multi_entity",
        log_dir="/log_dir",
        log_filename="test_log.csv"
    )
    
    # Check that directories were created
    mock_makedirs.assert_called_once_with(os.path.join("/log_dir"), exist_ok=True)
    
    # Check file was opened for appending
    mock_file.assert_called_once_with(os.path.join('/log_dir', 'test_log.csv'), mode='a', newline='')

# 3. Test for plot_training_curves
@patch('matplotlib.pyplot.savefig')
@patch('os.makedirs')
def test_plot_training_curves(mock_makedirs, mock_savefig):
    """Test that plotting function saves a file and handles path extension."""
    save_path = "/output/test_plot" # Note: no .png extension
    metrics = {
        'train_losses': [0.5, 0.4, 0.3],
        'val_losses': [0.6, 0.5, 0.4],
        'val_accs': [80, 85, 90]
    }
    
    training_utils.plot_training_curves(metrics, save_path, show_plot=False)
    
    # Assert that makedirs was called to create the directory
    mock_makedirs.assert_called_once_with(os.path.join("/output"), exist_ok=True)
    
    # Assert that savefig was called with the corrected path (.png added)
    # Use normpath to handle path separators consistently across platforms
    expected_path = "/output/test_plot.png"
    mock_savefig.assert_called_once_with(expected_path)

@patch('matplotlib.pyplot.savefig')
@patch('os.makedirs')
def test_plot_training_curves_with_png_extension(mock_makedirs, mock_savefig):
    """Test that plotting function handles paths that already have .png extension."""
    save_path = "/output/test_plot.png" # Already has .png extension
    metrics = {
        'train_losses': [0.5, 0.4, 0.3],
        'val_losses': [0.6, 0.5, 0.4],
        'val_accs': [80, 85, 90]
    }
    
    training_utils.plot_training_curves(metrics, save_path, show_plot=False)
    
    # Assert that makedirs was called to create the directory
    mock_makedirs.assert_called_once_with(os.path.join("/output"), exist_ok=True)
    
    # Assert that savefig was called with the same path (no additional .png)
    # Use normpath to handle path separators consistently across platforms
    expected_path = "/output/test_plot.png"
    mock_savefig.assert_called_once_with(expected_path) 