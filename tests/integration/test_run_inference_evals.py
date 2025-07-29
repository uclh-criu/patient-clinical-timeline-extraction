import pytest
import os
import sys
import pandas as pd
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the module to test
import run_inference_evals

class TestRunInferenceEvals:
    @patch('run_inference_evals.get_data_path')
    @patch('run_inference_evals.pd.read_csv')
    @patch('run_inference_evals.create_extractor')
    @patch('run_inference_evals.load_and_prepare_data')
    @patch('run_inference_evals.run_extraction')
    @patch('run_inference_evals.calculate_and_report_metrics')
    @patch('run_inference_evals.os.makedirs')
    @patch('run_inference_evals.sys.exit')
    def test_data_loading_and_extraction_diagnosis_only_mode(
        self, mock_exit, mock_makedirs, mock_metrics, mock_extraction, 
        mock_load_data, mock_create_extractor, mock_read_csv, mock_get_data_path,
        diagnosis_only_config
    ):
        """Test the data loading and extraction part in diagnosis_only mode."""
        # Configure mocks
        mock_get_data_path.return_value = diagnosis_only_config.DATA_PATH
        
        mock_extractor = MagicMock()
        mock_extractor.name = 'Naive (Proximity)'
        mock_extractor.load.return_value = True
        mock_create_extractor.return_value = mock_extractor
        
        prepared_data = [{'note_id': 1, 'patient_id': 'P001', 'note': 'test', 'entities': ([], [])}]
        gold_standard = [{'note_id': 1, 'diagnosis': 'test', 'date': '2023-01-01'}]
        mock_load_data.return_value = (prepared_data, gold_standard)
        
        predictions = [{'note_id': 1, 'patient_id': 'P001', 'entity_label': 'test', 'date': '2023-01-01'}]
        mock_extraction.return_value = predictions
        
        metrics = {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        mock_metrics.return_value = metrics
        
        # Create a test config that merges the diagnosis_only_config with additional settings
        test_config = SimpleNamespace(
            **{k: getattr(diagnosis_only_config, k) for k in dir(diagnosis_only_config) if not k.startswith('_')},
            EXTRACTION_METHOD='naive',
            INFERENCE_SAMPLES=10,
            PROXIMITY_MAX_DISTANCE=200,
            DEVICE='cpu'
        )
        
        # Mock the DataFrame and to_csv to avoid actual file operations
        mock_df = MagicMock()
        mock_read_csv.return_value = mock_df
        
        # Call the function with our test config
        with patch.object(sys, 'modules', {'config': test_config}):
            # We'll use a try/except block to catch the SystemExit that might be raised
            try:
                run_inference_evals.run_inference_and_evaluate()
            except SystemExit:
                pass  # Ignore the SystemExit
        
        # Verify that the key functions were called
        mock_get_data_path.assert_called_once()
        mock_create_extractor.assert_called_once_with('naive', test_config)
        mock_load_data.assert_called_once()
        mock_extraction.assert_called_once_with(mock_extractor, prepared_data)
        mock_metrics.assert_called_once()
        mock_makedirs.assert_called()
    
    @patch('run_inference_evals.get_data_path')
    @patch('run_inference_evals.pd.read_csv')
    @patch('run_inference_evals.create_extractor')
    @patch('run_inference_evals.load_and_prepare_data')
    @patch('run_inference_evals.run_extraction')
    @patch('run_inference_evals.calculate_entity_metrics')
    @patch('run_inference_evals.calculate_and_report_metrics')
    @patch('run_inference_evals.os.makedirs')
    @patch('run_inference_evals.sys.exit')
    def test_data_loading_and_extraction_multi_entity_mode(
        self, mock_exit, mock_makedirs, mock_rel_metrics, mock_entity_metrics, 
        mock_extraction, mock_load_data, mock_create_extractor, mock_read_csv,
        mock_get_data_path, multi_entity_config
    ):
        """Test the data loading and extraction part in multi_entity mode."""
        # Configure mocks
        mock_get_data_path.return_value = multi_entity_config.DATA_PATH
        
        mock_extractor = MagicMock()
        mock_extractor.name = 'Naive (Proximity)'
        mock_extractor.load.return_value = True
        mock_create_extractor.return_value = mock_extractor
        
        prepared_data = [{'note_id': 1, 'patient_id': 'P001', 'note': 'test', 'entities': ([], [])}]
        entity_gold = [{'note_id': 1, 'entity_label': 'test', 'entity_category': 'disorder'}]
        rel_gold = [{'note_id': 1, 'entity_label': 'test', 'date': '2023-01-01'}]
        pa_likelihood = {}
        mock_load_data.return_value = (prepared_data, entity_gold, rel_gold, pa_likelihood)
        
        predictions = [{'note_id': 1, 'patient_id': 'P001', 'entity_label': 'test', 'entity_category': 'disorder', 'date': '2023-01-01'}]
        mock_extraction.return_value = predictions
        
        entity_metrics = {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        mock_entity_metrics.return_value = entity_metrics
        
        rel_metrics = {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        mock_rel_metrics.return_value = rel_metrics
        
        # Create a test config that merges the multi_entity_config with additional settings
        test_config = SimpleNamespace(
            **{k: getattr(multi_entity_config, k) for k in dir(multi_entity_config) if not k.startswith('_')},
            EXTRACTION_METHOD='naive',
            INFERENCE_SAMPLES=10,
            PROXIMITY_MAX_DISTANCE=200,
            DEVICE='cpu'
        )
        
        # Mock the DataFrame and to_csv to avoid actual file operations
        mock_df = MagicMock()
        mock_read_csv.return_value = mock_df
        
        # Call the function with our test config
        with patch.object(sys, 'modules', {'config': test_config}):
            # We'll use a try/except block to catch the SystemExit that might be raised
            try:
                run_inference_evals.run_inference_and_evaluate()
            except SystemExit:
                pass  # Ignore the SystemExit
        
        # Verify that the key functions were called
        mock_get_data_path.assert_called_once()
        mock_create_extractor.assert_called_once_with('naive', test_config)
        mock_load_data.assert_called_once()
        mock_extraction.assert_called_once_with(mock_extractor, prepared_data)
        mock_entity_metrics.assert_called_once()
        mock_rel_metrics.assert_called_once()
        mock_makedirs.assert_called() 